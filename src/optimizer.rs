/// Core optimizer trait operating on raw `&mut [f32]` slices.
pub trait Optimizer {
    /// Apply one optimization step given current parameters and their gradients.
    fn step(&mut self, params: &mut [f32], grads: &[f32]);

    /// Reset all internal state (moments, velocity, step counters).
    fn reset(&mut self);

    /// Current learning rate.
    fn lr(&self) -> f32;

    /// Set the learning rate.
    fn set_lr(&mut self, lr: f32);
}

/// Stochastic gradient descent with optional momentum, weight decay, and Nesterov.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Sgd {
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    nesterov: bool,
    velocity: Option<Vec<f32>>,
}

impl Sgd {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            velocity: None,
        }
    }

    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        assert_eq!(params.len(), grads.len());
        let n = params.len();

        if self.velocity.is_none() && self.momentum != 0.0 {
            self.velocity = Some(vec![0.0; n]);
        }

        for i in 0..n {
            let mut g = grads[i];

            // L2 weight decay
            if self.weight_decay != 0.0 {
                g += self.weight_decay * params[i];
            }

            if self.momentum != 0.0 {
                let v = self.velocity.as_mut().unwrap();
                v[i] = self.momentum * v[i] + g;
                if self.nesterov {
                    g += self.momentum * v[i];
                } else {
                    g = v[i];
                }
            }

            params[i] -= self.lr * g;
        }
    }

    fn reset(&mut self) {
        self.velocity = None;
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Adam optimizer with optional decoupled weight decay (AdamW) and variance rectification (RAdam).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    decoupled: bool,
    /// When true, use RAdam variance rectification (Liu et al., 2020).
    rectified: bool,
    m: Option<Vec<f32>>,
    v: Option<Vec<f32>>,
    t: usize,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            decoupled: true,
            rectified: false,
            m: None,
            v: None,
            t: 0,
        }
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_decoupled(mut self, decoupled: bool) -> Self {
        self.decoupled = decoupled;
        self
    }

    /// Enable RAdam variance rectification. Eliminates the need for LR warmup.
    pub fn with_rectified(mut self, rectified: bool) -> Self {
        self.rectified = rectified;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        assert_eq!(params.len(), grads.len());
        let n = params.len();

        if self.m.is_none() {
            self.m = Some(vec![0.0; n]);
            self.v = Some(vec![0.0; n]);
        }

        self.t += 1;
        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        // RAdam: compute SMA length to decide whether variance is tractable.
        let rho_inf = 2.0 / (1.0 - self.beta2) - 1.0;
        let beta2_t = self.beta2.powi(self.t as i32);
        let rho_t = rho_inf - 2.0 * self.t as f32 * beta2_t / (1.0 - beta2_t);

        for i in 0..n {
            let mut g = grads[i];

            // L2 regularization (coupled weight decay)
            if !self.decoupled && self.weight_decay != 0.0 {
                g += self.weight_decay * params[i];
            }

            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g;
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g * g;

            let m_hat = m[i] / bias_correction1;

            let update = if self.rectified && rho_t <= 5.0 {
                // Variance is not tractable yet; use SGD-like step.
                m_hat
            } else {
                let v_hat = v[i] / bias_correction2;
                let mut step = m_hat / (v_hat.sqrt() + self.epsilon);
                if self.rectified {
                    // Apply variance rectification term.
                    let rect = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf
                        / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
                        .sqrt();
                    step *= rect;
                }
                step
            };

            // Decoupled weight decay (AdamW)
            if self.decoupled && self.weight_decay != 0.0 {
                params[i] -= self.lr * (update + self.weight_decay * params[i]);
            } else {
                params[i] -= self.lr * update;
            }
        }
    }

    fn reset(&mut self) {
        self.m = None;
        self.v = None;
        self.t = 0;
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Adagrad optimizer with per-parameter adaptive learning rates.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Adagrad {
    lr: f32,
    epsilon: f32,
    weight_decay: f32,
    initial_accumulator_value: f32,
    sum_of_squares: Option<Vec<f32>>,
}

impl Adagrad {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            epsilon: 1e-10,
            weight_decay: 0.0,
            initial_accumulator_value: 0.0,
            sum_of_squares: None,
        }
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_initial_accumulator_value(mut self, value: f32) -> Self {
        self.initial_accumulator_value = value;
        self
    }
}

impl Optimizer for Adagrad {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        assert_eq!(params.len(), grads.len());
        let n = params.len();

        if self.sum_of_squares.is_none() {
            self.sum_of_squares = Some(vec![self.initial_accumulator_value; n]);
        }

        let sos = self.sum_of_squares.as_mut().unwrap();

        for i in 0..n {
            let mut g = grads[i];

            // L2 weight decay
            if self.weight_decay != 0.0 {
                g += self.weight_decay * params[i];
            }

            sos[i] += g * g;
            params[i] -= self.lr * g / (sos[i].sqrt() + self.epsilon);
        }
    }

    fn reset(&mut self) {
        self.sum_of_squares = None;
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// LAMB optimizer (layerwise adaptive moments with trust ratio).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Lamb {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    trust_ratio_clip: Option<f32>,
    m: Option<Vec<f32>>,
    v: Option<Vec<f32>>,
    t: usize,
}

impl Lamb {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-6,
            weight_decay: 0.0,
            trust_ratio_clip: None,
            m: None,
            v: None,
            t: 0,
        }
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_trust_ratio_clip(mut self, clip: f32) -> Self {
        self.trust_ratio_clip = Some(clip);
        self
    }
}

impl Optimizer for Lamb {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        assert_eq!(params.len(), grads.len());
        let n = params.len();

        if self.m.is_none() {
            self.m = Some(vec![0.0; n]);
            self.v = Some(vec![0.0; n]);
        }

        self.t += 1;
        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        // Compute update vector
        let mut update = vec![0.0f32; n];
        for i in 0..n {
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grads[i];
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grads[i] * grads[i];

            let m_hat = m[i] / bias_correction1;
            let v_hat = v[i] / bias_correction2;

            update[i] = m_hat / (v_hat.sqrt() + self.epsilon);

            // Decoupled weight decay
            if self.weight_decay != 0.0 {
                update[i] += self.weight_decay * params[i];
            }
        }

        // Compute trust ratio
        let param_norm: f32 = params.iter().map(|p| p * p).sum::<f32>().sqrt();
        let update_norm: f32 = update.iter().map(|u| u * u).sum::<f32>().sqrt();

        let mut trust_ratio = if param_norm > 0.0 && update_norm > 0.0 {
            param_norm / update_norm
        } else {
            1.0
        };

        if let Some(clip) = self.trust_ratio_clip {
            trust_ratio = trust_ratio.min(clip);
        }

        for i in 0..n {
            params[i] -= self.lr * trust_ratio * update[i];
        }
    }

    fn reset(&mut self) {
        self.m = None;
        self.v = None;
        self.t = 0;
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// RMSprop optimizer (Hinton, unpublished).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RmsProp {
    lr: f32,
    alpha: f32,
    epsilon: f32,
    weight_decay: f32,
    momentum: f32,
    centered: bool,
    square_avg: Option<Vec<f32>>,
    grad_avg: Option<Vec<f32>>,
    momentum_buf: Option<Vec<f32>>,
}

impl RmsProp {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            alpha: 0.99,
            epsilon: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
            square_avg: None,
            grad_avg: None,
            momentum_buf: None,
        }
    }

    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn with_centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }
}

impl Optimizer for RmsProp {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        assert_eq!(params.len(), grads.len());
        let n = params.len();

        if self.square_avg.is_none() {
            self.square_avg = Some(vec![0.0; n]);
            if self.centered {
                self.grad_avg = Some(vec![0.0; n]);
            }
            if self.momentum > 0.0 {
                self.momentum_buf = Some(vec![0.0; n]);
            }
        }

        let sq = self.square_avg.as_mut().unwrap();

        for i in 0..n {
            let mut g = grads[i];
            if self.weight_decay != 0.0 {
                g += self.weight_decay * params[i];
            }

            sq[i] = self.alpha * sq[i] + (1.0 - self.alpha) * g * g;

            let avg = if self.centered {
                let ga = self.grad_avg.as_mut().unwrap();
                ga[i] = self.alpha * ga[i] + (1.0 - self.alpha) * g;
                (sq[i] - ga[i] * ga[i]).sqrt() + self.epsilon
            } else {
                sq[i].sqrt() + self.epsilon
            };

            if self.momentum > 0.0 {
                let mb = self.momentum_buf.as_mut().unwrap();
                mb[i] = self.momentum * mb[i] + g / avg;
                params[i] -= self.lr * mb[i];
            } else {
                params[i] -= self.lr * g / avg;
            }
        }
    }

    fn reset(&mut self) {
        self.square_avg = None;
        self.grad_avg = None;
        self.momentum_buf = None;
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Lion optimizer -- sign-based, memory efficient (Chen et al., 2023).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Lion {
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    m: Option<Vec<f32>>,
}

impl Lion {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
            m: None,
        }
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for Lion {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        assert_eq!(params.len(), grads.len());
        let n = params.len();

        if self.m.is_none() {
            self.m = Some(vec![0.0; n]);
        }

        let m = self.m.as_mut().unwrap();

        for i in 0..n {
            let g = grads[i];

            // Compute update direction from interpolation of momentum and gradient.
            let update = (self.beta1 * m[i] + (1.0 - self.beta1) * g).signum();

            // Update momentum for next step.
            m[i] = self.beta2 * m[i] + (1.0 - self.beta2) * g;

            // Apply weight decay and sign update.
            params[i] -= self.lr * (update + self.weight_decay * params[i]);
        }
    }

    fn reset(&mut self) {
        self.m = None;
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Number of state buffers an optimizer allocates per parameter.
pub trait StateSize {
    fn buffers_per_param(&self) -> usize;
}

impl StateSize for Adam {
    fn buffers_per_param(&self) -> usize {
        2 // m, v
    }
}

impl StateSize for Lion {
    fn buffers_per_param(&self) -> usize {
        1 // m only
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic_grads(params: &[f32]) -> Vec<f32> {
        // f(x) = sum(x_i^2), grad = 2*x_i
        params.iter().map(|&x| 2.0 * x).collect()
    }

    /// Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1,1).
    fn rosenbrock_grads(params: &[f32]) -> Vec<f32> {
        let x = params[0];
        let y = params[1];
        let dx = -2.0 * (1.0 - x) + 200.0 * (y - x * x) * (-2.0 * x);
        let dy = 200.0 * (y - x * x);
        vec![dx, dy]
    }

    fn rosenbrock_value(params: &[f32]) -> f32 {
        let x = params[0];
        let y = params[1];
        (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
    }

    #[test]
    fn test_sgd_convergence() {
        let mut opt = Sgd::new(0.1);
        let mut params = vec![5.0];

        for _ in 0..100 {
            let grads = quadratic_grads(&params);
            opt.step(&mut params, &grads);
        }

        assert!(
            params[0].abs() < 1e-6,
            "SGD did not converge: {}",
            params[0]
        );
    }

    #[test]
    fn test_adam_convergence() {
        let mut opt = Adam::new(0.1);
        let mut params = vec![5.0];

        for _ in 0..200 {
            let grads = quadratic_grads(&params);
            opt.step(&mut params, &grads);
        }

        assert!(
            params[0].abs() < 1e-3,
            "Adam did not converge: {}",
            params[0]
        );
    }

    #[test]
    fn test_adagrad_convergence() {
        let mut opt = Adagrad::new(1.0);
        let mut params = vec![5.0];

        for _ in 0..200 {
            let grads = quadratic_grads(&params);
            opt.step(&mut params, &grads);
        }

        assert!(
            params[0].abs() < 1e-2,
            "Adagrad did not converge: {}",
            params[0]
        );
    }

    #[test]
    fn test_weight_decay() {
        // L2 weight decay adds weight_decay * param to the gradient,
        // so even with zero external gradient, params shrink.
        let mut opt = Sgd::new(0.1).with_weight_decay(0.1);
        let mut params = vec![5.0];
        let grads = vec![0.0];

        for _ in 0..50 {
            opt.step(&mut params, &grads);
        }

        assert!(
            params[0] < 5.0,
            "Weight decay did not shrink params: {}",
            params[0]
        );
        assert!(params[0] > 0.0, "Params should remain positive");
    }

    #[test]
    fn test_lazy_init() {
        let mut opt = Adam::new(0.01);
        assert!(opt.m.is_none());
        assert!(opt.v.is_none());

        let mut params = vec![1.0, 2.0];
        let grads = vec![0.1, 0.2];
        opt.step(&mut params, &grads);

        assert!(opt.m.is_some());
        assert_eq!(opt.m.as_ref().unwrap().len(), 2);

        // Second step reuses state
        opt.step(&mut params, &grads);
        assert_eq!(opt.t, 2);
    }

    #[test]
    fn test_adamw_decoupled() {
        let mut adamw = Adam::new(0.01).with_weight_decay(0.1).with_decoupled(true);
        let mut adam_l2 = Adam::new(0.01).with_weight_decay(0.1).with_decoupled(false);

        let mut params_w = vec![5.0, 3.0];
        let mut params_l2 = vec![5.0, 3.0];
        let grads = vec![1.0, 1.0];

        for _ in 0..20 {
            adamw.step(&mut params_w, &grads);
            adam_l2.step(&mut params_l2, &grads);
        }

        // Decoupled and L2 should produce different results
        assert!(
            (params_w[0] - params_l2[0]).abs() > 1e-6,
            "AdamW and Adam+L2 should differ"
        );
    }

    #[test]
    fn test_reset() {
        let mut opt = Adam::new(0.01);
        let mut params = vec![1.0];
        let grads = vec![0.1];

        opt.step(&mut params, &grads);
        assert!(opt.m.is_some());
        assert_eq!(opt.t, 1);

        opt.reset();
        assert!(opt.m.is_none());
        assert!(opt.v.is_none());
        assert_eq!(opt.t, 0);
    }

    #[test]
    fn test_radam_convergence() {
        // RAdam should converge on quadratic without warmup.
        let mut opt = Adam::new(0.1).with_rectified(true);
        let mut params = vec![5.0];

        for _ in 0..300 {
            let grads = quadratic_grads(&params);
            opt.step(&mut params, &grads);
        }

        assert!(
            params[0].abs() < 1e-2,
            "RAdam did not converge: {}",
            params[0]
        );
    }

    #[test]
    fn test_radam_early_steps_stable() {
        let mut opt = Adam::new(0.1).with_rectified(true);
        let mut params = vec![5.0, -3.0, 10.0];

        for _ in 0..10 {
            let grads = quadratic_grads(&params);
            opt.step(&mut params, &grads);
            for &p in &params {
                assert!(p.is_finite(), "RAdam produced non-finite value: {}", p);
            }
        }
    }

    #[test]
    fn test_rmsprop_convergence() {
        let mut opt = RmsProp::new(0.01);
        let mut params = vec![5.0];

        for _ in 0..2000 {
            let grads = quadratic_grads(&params);
            opt.step(&mut params, &grads);
        }

        assert!(
            params[0].abs() < 0.1,
            "RMSprop did not converge: {}",
            params[0]
        );
    }

    #[test]
    fn test_rmsprop_centered() {
        let mut centered = RmsProp::new(0.01).with_centered(true);
        let mut non_centered = RmsProp::new(0.01).with_centered(false);

        let mut params_c = vec![5.0, 3.0];
        let mut params_n = vec![5.0, 3.0];

        for _ in 0..50 {
            let gc = quadratic_grads(&params_c);
            let gn = quadratic_grads(&params_n);
            centered.step(&mut params_c, &gc);
            non_centered.step(&mut params_n, &gn);
        }

        // Centered and non-centered should produce different trajectories.
        assert!(
            (params_c[0] - params_n[0]).abs() > 1e-6,
            "Centered and non-centered RMSprop should differ"
        );
    }

    #[test]
    fn test_lion_convergence() {
        // Lion uses sign-based updates (fixed step size = lr), so it needs
        // many steps. With lr=0.001, each step moves exactly 0.001.
        let mut opt = Lion::new(0.01);
        let mut params = vec![5.0];

        for _ in 0..1000 {
            let grads = quadratic_grads(&params);
            opt.step(&mut params, &grads);
        }

        assert!(
            params[0].abs() < 0.5,
            "Lion did not converge: {}",
            params[0]
        );
    }

    #[test]
    fn test_lion_memory_efficiency() {
        let adam = Adam::new(0.01);
        let lion = Lion::new(0.01);

        assert_eq!(adam.buffers_per_param(), 2);
        assert_eq!(lion.buffers_per_param(), 1);
    }

    #[test]
    fn test_rosenbrock_adam() {
        let mut opt = Adam::new(0.001);
        let mut params = vec![-1.0, 1.0];

        for _ in 0..10_000 {
            let grads = rosenbrock_grads(&params);
            opt.step(&mut params, &grads);
        }

        let val = rosenbrock_value(&params);
        assert!(
            val < 0.01,
            "Adam on Rosenbrock: f={}, params=({}, {})",
            val,
            params[0],
            params[1]
        );
    }

    #[test]
    fn test_rosenbrock_sgd_momentum() {
        let mut opt = Sgd::new(0.0001).with_momentum(0.9);
        let mut params = vec![-1.0, 1.0];

        for _ in 0..50_000 {
            let grads = rosenbrock_grads(&params);
            opt.step(&mut params, &grads);
        }

        let val = rosenbrock_value(&params);
        assert!(
            val < 0.1,
            "SGD+momentum on Rosenbrock: f={}, params=({}, {})",
            val,
            params[0],
            params[1]
        );
    }
}
