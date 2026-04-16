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
pub use riemannian::{RiemannianAdamState, geodesic_distance, riemannian_adam_step, riemannian_sgd_step};
pub use schedule::{
    ConstantLr, CosineAnnealing, InverseSqrt, LinearWarmup, LrSchedule, OneCycleLr,
    ScheduledOptimizer, SequentialSchedule, StepDecay, WarmupCosine,
};
