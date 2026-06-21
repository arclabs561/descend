//! Training-loop primitives: optimizers, learning-rate schedules, gradient
//! clipping, weight averaging, gradient accumulation, and early stopping.
//!
//! `descend` provides the moving parts of a training loop as small, composable
//! pieces that operate on plain parameter and gradient slices, so they drop into
//! a hand-written loop without committing to a tensor framework. It does not
//! define tensors, autodiff, or a `Module` abstraction; bring your own (ndarray,
//! candle, burn) and pass gradients in.
//!
//! # Modules
//! - [`optimizer`]: first-order optimizers (SGD, Adam, Adagrad, RMSprop, Lion,
//!   LAMB) behind the [`Optimizer`] trait.
//! - [`schedule`]: learning-rate schedules (constant, linear warmup, cosine,
//!   warmup-cosine, step decay, inverse-sqrt, one-cycle, sequential) and
//!   [`ScheduledOptimizer`], which wraps an optimizer with a schedule.
//! - [`gradient`]: gradient clipping by global norm ([`clip_grad_norm`]) or by
//!   value ([`clip_grad_value`]).
//! - [`ema`]: exponential moving average of weights ([`WeightAverager`]).
//! - [`accumulator`]: gradient accumulation across micro-batches
//!   ([`GradientAccumulator`]).
//! - [`early_stopping`]: patience-based stopping ([`EarlyStopping`],
//!   [`StopMode`]).
//! - [`metrics`]: running training metrics ([`TrainingMetrics`],
//!   [`MetricSummary`]).
//! - `riemannian` (feature `riemannian`): Riemannian SGD/Adam steps and geodesic
//!   distance for optimization on manifolds.

pub mod accumulator;
pub mod early_stopping;
pub mod ema;
pub mod gradient;
pub mod metrics;
pub mod optimizer;
#[cfg(feature = "riemannian")]
pub mod riemannian;
pub mod schedule;

pub use accumulator::GradientAccumulator;
pub use early_stopping::{EarlyStopping, StopMode};
pub use ema::WeightAverager;
pub use gradient::{clip_grad_norm, clip_grad_value};
pub use metrics::{MetricSummary, TrainingMetrics};
pub use optimizer::{Adagrad, Adam, Lamb, Lion, Optimizer, RmsProp, Sgd};
#[cfg(feature = "riemannian")]
pub use riemannian::{
    RiemannianAdamState, geodesic_distance, riemannian_adam_step, riemannian_sgd_step,
};
pub use schedule::{
    ConstantLr, CosineAnnealing, InverseSqrt, LinearWarmup, LrSchedule, OneCycleLr,
    ScheduledOptimizer, SequentialSchedule, StepDecay, WarmupCosine,
};
