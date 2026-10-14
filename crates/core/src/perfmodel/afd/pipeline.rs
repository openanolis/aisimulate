// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ordered stages with one server per stage and `m` circulating microbatches.

use serde::{Deserialize, Serialize};

use super::model::AfdLayerTimes;
use crate::common::error::AicError;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AfdPipelineLayout {
    /// A, A→F, F and F→A can each work independently.
    #[default]
    FourStage,
    /// A→F immediately follows A on one resource; F→A follows F on another.
    TwoStage,
}

impl AfdPipelineLayout {
    /// Heterogeneous source layer sequence. Empty stages are bypassed rather
    /// than queued on an unrelated resource (dense layers use A only).
    pub(crate) fn plan_latency_ms(
        self,
        templates: &[[f64; 5]],
        layers: &[usize],
        microbatches: u32,
        factor: f64,
        prefix: f64,
        suffix: f64,
    ) -> Result<f64, AicError> {
        if microbatches == 0
            || layers.is_empty()
            || !factor.is_finite()
            || factor <= 0.0
            || [prefix, suffix].iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(AicError::InvalidEngineConfig(
                "invalid AFD layer pipeline geometry".into(),
            ));
        }
        let mut inline = [0.0; 4];
        let mut heap = Vec::new();
        let m = microbatches as usize;
        let completed = if m <= 4 {
            &mut inline[..m]
        } else {
            heap.resize(m, 0.0);
            &mut heap[..]
        };
        let mut available = [0.0_f64; 4];
        for (layer_index, &template) in layers.iter().enumerate() {
            let d = templates.get(template).ok_or_else(|| {
                AicError::InvalidEngineConfig("invalid AFD layer template".into())
            })?;
            let stages = match self {
                Self::FourStage => [d[0], d[1] * factor, d[2], d[3] * factor + d[4]],
                Self::TwoStage => [d[0] + d[1] * factor, d[2] + d[3] * factor + d[4], 0.0, 0.0],
            };
            if stages.iter().any(|t| !t.is_finite() || *t < 0.0) {
                return Err(AicError::InvalidEngineConfig(
                    "invalid AFD stage duration".into(),
                ));
            }
            for completion in completed.iter_mut() {
                let mut ready = *completion;
                for (r, &duration) in stages.iter().enumerate() {
                    let duration = duration
                        + if r == 0 && layer_index == 0 {
                            prefix
                        } else {
                            0.0
                        };
                    if duration == 0.0 {
                        continue;
                    }
                    available[r] = available[r].max(ready) + duration;
                    ready = available[r];
                }
                *completion = ready;
            }
        }
        // Logits and other one-time work are on A after the final return.
        for &ready in completed.iter() {
            available[0] = available[0].max(ready) + suffix;
        }
        let step = completed.iter().copied().fold(available[0], f64::max);
        if !step.is_finite() || step <= 0.0 {
            return Err(AicError::InvalidEngineConfig(
                "invalid AFD pass latency".into(),
            ));
        }
        Ok(step)
    }

    /// Apply the transfer factor once, before grouping stages.
    pub fn step_latency_ms(
        self,
        times: &AfdLayerTimes,
        microbatches: u32,
        comm_overhead_factor: f64,
    ) -> Result<f64, AicError> {
        if !comm_overhead_factor.is_finite() || comm_overhead_factor <= 0.0 {
            return Err(AicError::InvalidEngineConfig(
                "AFD comm_overhead_factor must be positive and finite".into(),
            ));
        }
        self.plan_latency_ms(
            &[[
                times.attention_ms,
                times.a_to_f_ms,
                times.ffn_ms,
                times.f_to_a_ms,
                0.0,
            ]],
            &vec![0; times.num_layers as usize],
            microbatches,
            comm_overhead_factor,
            0.0,
            0.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_execute_once_on_the_a_resource() {
        // m=2, L=2: routed returns are 8/11 then 14/17. The A head
        // starts at 14 and 19, costs 5 each, and completes the pass at 24.
        assert_eq!(
            AfdPipelineLayout::FourStage
                .plan_latency_ms(&[[1., 1., 3., 1., 0.]], &[0, 0], 2, 1., 2., 5.)
                .unwrap(),
            24.
        );
        // Amortizing the seven endpoint milliseconds into two layers changes
        // the resource ordering and gives 23.5 instead.
        assert_eq!(
            AfdPipelineLayout::FourStage
                .step_latency_ms(&times([4.5, 1., 3., 1.], 2), 2, 1.)
                .unwrap(),
            23.5
        );
    }

    #[test]
    fn heterogeneous_layers_skip_idle_resources_and_preserve_source_order() {
        let templates = [[1., 1., 10., 1., 0.], [2., 0., 0., 0., 0.]];
        // Routed microbatches return at 13 and 23. The first dense layer
        // completes at 15, independently of the F server still busy to 22.
        assert_eq!(
            AfdPipelineLayout::FourStage
                .plan_latency_ms(&templates, &[0, 1], 2, 1., 0., 0.)
                .unwrap(),
            25.
        );
        assert_eq!(
            AfdPipelineLayout::FourStage
                .plan_latency_ms(&templates, &[1, 0], 2, 1., 0., 0.)
                .unwrap(),
            27.
        );
        assert_eq!(
            AfdPipelineLayout::TwoStage
                .plan_latency_ms(&templates, &[0, 1], 2, 1., 0., 0.)
                .unwrap(),
            26.
        );
        assert_eq!(
            AfdPipelineLayout::TwoStage
                .plan_latency_ms(&templates, &[1, 0], 2, 1., 0., 0.)
                .unwrap(),
            28.
        );
    }

    #[test]
    fn return_restore_and_one_time_work_are_not_transfer_scaled() {
        // One routed layer costs 1+2+3+4+5, dense costs 7, prefix/suffix
        // cost 2+3: 27. Doubling transfers adds only 2+4, giving 33.
        for layout in [AfdPipelineLayout::FourStage, AfdPipelineLayout::TwoStage] {
            let templates = [[1., 2., 3., 4., 5.], [7., 0., 0., 0., 0.]];
            assert_eq!(
                layout
                    .plan_latency_ms(&templates, &[0, 1], 1, 1., 2., 3.)
                    .unwrap(),
                27.
            );
            assert_eq!(
                layout
                    .plan_latency_ms(&templates, &[0, 1], 1, 2., 2., 3.)
                    .unwrap(),
                33.
            );
        }
    }

    fn times(stages: [f64; 4], layers: u32) -> AfdLayerTimes {
        AfdLayerTimes {
            attention_ms: stages[0],
            a_to_f_ms: stages[1],
            ffn_ms: stages[2],
            f_to_a_ms: stages[3],
            num_layers: layers,
        }
    }

    /// Four unit stages finish the first microbatch at 4 and others one
    /// unit apart. With two merged stages, completions are two units apart.
    /// Feedback permits a two-stage pipeline to stay busy with m>=2.
    #[test]
    fn hand_timeline_for_common_depths() {
        for (layout, single, double) in [
            (
                AfdPipelineLayout::FourStage,
                [4., 5., 6., 7.],
                [8., 9., 10., 11.],
            ),
            (
                AfdPipelineLayout::TwoStage,
                [4., 6., 8., 10.],
                [8., 10., 14., 18.],
            ),
        ] {
            for m in 1..=4 {
                assert_eq!(
                    layout.step_latency_ms(&times([1.; 4], 1), m, 1.).unwrap(),
                    single[m as usize - 1]
                );
                assert_eq!(
                    layout.step_latency_ms(&times([1.; 4], 2), m, 1.).unwrap(),
                    double[m as usize - 1]
                );
            }
        }
    }

    /// For [2,4,3,5], first-layer completions are 14,19,24,29.
    /// For m=2, second-layer completions are 28,33 (feedback causes a gap).
    /// Merging yields [6,8], whose m>=2 cadence is 8 after the first 14.
    #[test]
    fn unequal_stages_and_feedback() {
        for (layout, expected) in [
            (AfdPipelineLayout::FourStage, [28., 33., 39., 49.]),
            (AfdPipelineLayout::TwoStage, [28., 38., 54., 70.]),
        ] {
            for m in 1..=4 {
                assert_eq!(
                    layout
                        .step_latency_ms(&times([2., 4., 3., 5.], 2), m, 1.)
                        .unwrap(),
                    expected[m as usize - 1]
                );
            }
        }
    }

    #[test]
    fn zero_transfers_compute_and_one_way_transfer_bottlenecks() {
        // Only one nonzero server: every job consumes exactly that duration.
        for stages in [
            [5., 0., 0., 0.],
            [0., 5., 0., 0.],
            [0., 0., 5., 0.],
            [0., 0., 0., 5.],
        ] {
            for layout in [AfdPipelineLayout::FourStage, AfdPipelineLayout::TwoStage] {
                for m in [1, 2, 3, 4, 7] {
                    assert_eq!(
                        layout.step_latency_ms(&times(stages, 3), m, 1.).unwrap(),
                        15. * f64::from(m)
                    );
                }
            }
        }
    }

    #[test]
    fn invalid_times_and_counts_reject() {
        let layout = AfdPipelineLayout::FourStage;
        for bad in [-1., f64::NAN, f64::INFINITY] {
            assert!(
                layout
                    .step_latency_ms(&times([1., bad, 1., 1.], 2), 2, 1.)
                    .is_err()
            );
        }
        assert!(layout.step_latency_ms(&times([0.; 4], 2), 2, 1.).is_err());
        assert!(layout.step_latency_ms(&times([1.; 4], 0), 2, 1.).is_err());
        assert!(layout.step_latency_ms(&times([1.; 4], 2), 0, 1.).is_err());
    }
}
