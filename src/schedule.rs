use crate::optimizer::Optimizer;

/// Learning rate schedule that computes LR as a function of step count.
pub trait LrSchedule {
    /// Compute the learning rate at a given step.
    fn lr_at(&self, step: usize, base_lr: f32) -> f32;
}

/// Constant learning rate (identity schedule).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConstantLr;

impl LrSchedule for ConstantLr {
    fn lr_at(&self, _step: usize, base_lr: f32) -> f32 {
        base_lr
    }
}

/// Linear warmup from 0 to base_lr, then constant.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinearWarmup {
    pub warmup_steps: usize,
}

impl LrSchedule for LinearWarmup {
    fn lr_at(&self, step: usize, base_lr: f32) -> f32 {
        if self.warmup_steps == 0 {
            return base_lr;
        }
        if step >= self.warmup_steps {
            base_lr
        } else {
            base_lr * (step as f32 / self.warmup_steps as f32)
        }
    }
}

/// Cosine annealing from base_lr to eta_min over t_max steps.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CosineAnnealing {
    pub t_max: usize,
    pub eta_min: f32,
}

impl LrSchedule for CosineAnnealing {
    fn lr_at(&self, step: usize, base_lr: f32) -> f32 {
        if step >= self.t_max {
            return self.eta_min;
        }
        let progress = step as f32 / self.t_max as f32;
        self.eta_min
            + (base_lr - self.eta_min) * (1.0 + (progress * std::f32::consts::PI).cos()) / 2.0
    }
}

/// Linear warmup followed by cosine annealing.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WarmupCosine {
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub eta_min: f32,
}

impl LrSchedule for WarmupCosine {
    fn lr_at(&self, step: usize, base_lr: f32) -> f32 {
        if step < self.warmup_steps {
            if self.warmup_steps == 0 {
                return base_lr;
            }
            return base_lr * (step as f32 / self.warmup_steps as f32);
        }

        let cosine_steps = self.total_steps - self.warmup_steps;
        if cosine_steps == 0 || step >= self.total_steps {
            return self.eta_min;
        }

        let progress = (step - self.warmup_steps) as f32 / cosine_steps as f32;
        self.eta_min
            + (base_lr - self.eta_min) * (1.0 + (progress * std::f32::consts::PI).cos()) / 2.0
    }
}

/// Step decay: multiply LR by gamma every step_size steps.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StepDecay {
    pub step_size: usize,
    pub gamma: f32,
}

impl LrSchedule for StepDecay {
    fn lr_at(&self, step: usize, base_lr: f32) -> f32 {
        let n = (step / self.step_size) as u32;
        base_lr * self.gamma.powi(n as i32)
    }
}

/// Wraps an optimizer with a learning rate schedule.
#[derive(Debug, Clone)]
pub struct ScheduledOptimizer<O, S> {
    optimizer: O,
    schedule: S,
    base_lr: f32,
    current_step: usize,
}

impl<O: Optimizer, S: LrSchedule> ScheduledOptimizer<O, S> {
    pub fn new(optimizer: O, schedule: S) -> Self {
        let base_lr = optimizer.lr();
        Self {
            optimizer,
            schedule,
            base_lr,
            current_step: 0,
        }
    }

    /// Current step count.
    pub fn current_step(&self) -> usize {
        self.current_step
    }
}

impl<O: Optimizer, S: LrSchedule> Optimizer for ScheduledOptimizer<O, S> {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        let scheduled_lr = self.schedule.lr_at(self.current_step, self.base_lr);
        self.optimizer.set_lr(scheduled_lr);
        self.optimizer.step(params, grads);
        self.current_step += 1;
    }

    fn reset(&mut self) {
        self.optimizer.reset();
        self.current_step = 0;
    }

    fn lr(&self) -> f32 {
        self.optimizer.lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.base_lr = lr;
        self.optimizer.set_lr(lr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::Sgd;

    #[test]
    fn test_constant_lr() {
        let sched = ConstantLr;
        assert_eq!(sched.lr_at(0, 0.1), 0.1);
        assert_eq!(sched.lr_at(100, 0.1), 0.1);
        assert_eq!(sched.lr_at(10000, 0.5), 0.5);
    }

    #[test]
    fn test_linear_warmup() {
        let sched = LinearWarmup { warmup_steps: 100 };
        assert_eq!(sched.lr_at(0, 1.0), 0.0);
        assert!((sched.lr_at(50, 1.0) - 0.5).abs() < 1e-6);
        assert!((sched.lr_at(100, 1.0) - 1.0).abs() < 1e-6);
        assert!((sched.lr_at(200, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_annealing() {
        let sched = CosineAnnealing {
            t_max: 100,
            eta_min: 0.0,
        };

        // At step 0: base_lr
        assert!((sched.lr_at(0, 1.0) - 1.0).abs() < 1e-6);

        // At step t_max: eta_min
        assert!((sched.lr_at(100, 1.0) - 0.0).abs() < 1e-6);

        // At midpoint: ~0.5
        let mid = sched.lr_at(50, 1.0);
        assert!((mid - 0.5).abs() < 1e-4, "midpoint was {}", mid);
    }

    #[test]
    fn test_warmup_cosine() {
        let sched = WarmupCosine {
            warmup_steps: 10,
            total_steps: 110,
            eta_min: 0.0,
        };

        // Warmup phase
        assert_eq!(sched.lr_at(0, 1.0), 0.0);
        assert!((sched.lr_at(5, 1.0) - 0.5).abs() < 1e-6);
        assert!((sched.lr_at(10, 1.0) - 1.0).abs() < 1e-4);

        // Cosine phase
        let at_end = sched.lr_at(110, 1.0);
        assert!(at_end.abs() < 1e-4, "end was {}", at_end);
    }

    #[test]
    fn test_step_decay() {
        let sched = StepDecay {
            step_size: 10,
            gamma: 0.5,
        };

        assert!((sched.lr_at(0, 1.0) - 1.0).abs() < 1e-6);
        assert!((sched.lr_at(9, 1.0) - 1.0).abs() < 1e-6);
        assert!((sched.lr_at(10, 1.0) - 0.5).abs() < 1e-6);
        assert!((sched.lr_at(20, 1.0) - 0.25).abs() < 1e-6);
        assert!((sched.lr_at(30, 1.0) - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_scheduled_optimizer() {
        let sgd = Sgd::new(0.1);
        let sched = LinearWarmup { warmup_steps: 10 };
        let mut opt = ScheduledOptimizer::new(sgd, sched);

        let mut params = vec![5.0];

        // Step 0: lr should be 0.0 (warmup start)
        opt.step(&mut params, &[1.0]);
        // params unchanged at step 0 since lr=0
        assert!((params[0] - 5.0).abs() < 1e-6);
        assert_eq!(opt.current_step(), 1);

        // Steps 1-9: LR ramps up linearly
        for _ in 1..10 {
            opt.step(&mut params, &[1.0]);
        }
        assert_eq!(opt.current_step(), 10);

        // Last step used lr_at(9, 0.1) = 0.09. One more step uses lr_at(10, 0.1) = 0.1.
        opt.step(&mut params, &[1.0]);
        assert!((opt.lr() - 0.1).abs() < 1e-6);
        assert_eq!(opt.current_step(), 11);
    }
}
