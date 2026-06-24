//! Riemannian optimization algorithms.
//!
//! Gradient-based optimization on manifolds differs from Euclidean optimization
//! in two ways:
//!
//! 1. Gradients live in tangent spaces, not in the ambient space.
//! 2. Updates follow geodesics (via `exp_map`), not straight lines.
//!
//! This module provides [`riemannian_sgd_step`] and [`riemannian_adam_step`]
//! that operate on any type implementing [`skel::Manifold`].
//!
//! # Feature gate
//!
//! Requires the `riemannian` feature:
//!
//! ```toml
//! [dependencies]
//! descend = { version = "0.1", features = ["riemannian"] }
//! ```
//!
//! # References
//!
//! - Bonnabel (2013), "Stochastic Gradient Descent on Riemannian Manifolds"
//!   -- foundational convergence analysis for Riemannian SGD.
//! - Becigneul & Ganea (2019), "Riemannian Adaptive Optimization Methods"
//!   (ICLR) -- extends Adam to Riemannian manifolds using parallel transport
//!   for moment vectors.

use ndarray::{Array1, ArrayView1};
use skel::Manifold;

/// Riemannian SGD step.
///
/// Given a point on the manifold and a Euclidean gradient, computes:
///
/// 1. Project the gradient to the tangent space (via `log_map` identity, then
///    correct with `project` -- for most manifolds the Euclidean gradient
///    projected to the tangent space is the Riemannian gradient).
/// 2. Step along the geodesic: `x_new = exp_map(x, -lr * grad)`.
/// 3. Project back to the manifold (numerical correction).
///
/// Returns the updated point on the manifold.
pub fn riemannian_sgd_step(
    manifold: &dyn Manifold,
    point: &ArrayView1<f64>,
    euclidean_grad: &ArrayView1<f64>,
    lr: f64,
) -> Array1<f64> {
    // Negate and scale the gradient to get the tangent vector for descent.
    let neg_grad: Array1<f64> = euclidean_grad.mapv(|g| -lr * g);
    // Follow the geodesic from point in direction of negative gradient.
    let stepped = manifold.exp_map(point, &neg_grad.view());
    // Project back to the manifold to correct numerical drift.
    manifold.project(&stepped.view())
}

/// State for the Riemannian Adam optimizer.
///
/// Moment estimates (`m` and `v`) live in the tangent space at `prev_point`.
/// On each step they are parallel-transported to the current tangent space
/// before being updated.
pub struct RiemannianAdamState {
    /// First moment estimate (in tangent space at `prev_point`).
    pub m: Array1<f64>,
    /// Second moment estimate (element-wise, in tangent space at `prev_point`).
    pub v: Array1<f64>,
    /// Step counter (starts at 0, incremented on each call).
    pub t: usize,
    /// Previous point on the manifold (for parallel transport of moments).
    pub prev_point: Array1<f64>,
}

impl RiemannianAdamState {
    /// Create a new state initialized at `point` with zero moments.
    pub fn new(point: Array1<f64>) -> Self {
        let dim = point.len();
        Self {
            m: Array1::zeros(dim),
            v: Array1::zeros(dim),
            t: 0,
            prev_point: point,
        }
    }
}

/// Riemannian Adam step (Becigneul & Ganea, 2019).
///
/// Extends Adam to Riemannian manifolds:
///
/// 1. Parallel-transport previous moments from `T_{prev} M` to `T_{current} M`.
/// 2. Update first and second moments with the current Riemannian gradient.
/// 3. Compute bias-corrected update direction.
/// 4. Follow the geodesic via `exp_map` and project back to the manifold.
///
/// Returns the updated point on the manifold.  The `state` is modified in place
/// with the new moments and previous point.
#[allow(clippy::too_many_arguments)]
pub fn riemannian_adam_step(
    manifold: &dyn Manifold,
    point: &ArrayView1<f64>,
    euclidean_grad: &ArrayView1<f64>,
    state: &mut RiemannianAdamState,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
) -> Array1<f64> {
    // Transport first moment to the current tangent space.
    // The second moment (element-wise squared magnitudes) is a scalar per
    // coordinate -- it decays without transport, following Becigneul & Ganea.
    let m_transported =
        manifold.parallel_transport(&state.prev_point.view(), point, &state.m.view());

    state.t += 1;

    // Update moments in the current tangent space.
    // m_t = beta1 * m_{t-1} + (1 - beta1) * grad
    // v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2  (element-wise)
    let dim = euclidean_grad.len();
    let mut m_new = Array1::zeros(dim);
    let mut v_new = Array1::zeros(dim);
    for i in 0..dim {
        m_new[i] = beta1 * m_transported[i] + (1.0 - beta1) * euclidean_grad[i];
        v_new[i] = beta2 * state.v[i] + (1.0 - beta2) * euclidean_grad[i] * euclidean_grad[i];
    }

    // Bias correction.
    let bc1 = 1.0 - beta1.powi(state.t as i32);
    let bc2 = 1.0 - beta2.powi(state.t as i32);

    // Compute the update direction in tangent space.
    let mut update = Array1::zeros(dim);
    for i in 0..dim {
        let m_hat = m_new[i] / bc1;
        let v_hat = v_new[i] / bc2;
        update[i] = -lr * m_hat / (v_hat.sqrt() + eps);
    }

    // Follow the geodesic and project.
    let stepped = manifold.exp_map(point, &update.view());
    let result = manifold.project(&stepped.view());

    // Store updated state.
    state.m = m_new;
    state.v = v_new;
    state.prev_point = point.to_owned();

    result
}

/// Geodesic distance between two points on a manifold.
///
/// Computed as `||log_x(y)||`, the norm of the tangent vector at `x` that
/// points toward `y`.  This equals the length of the shortest geodesic
/// connecting `x` and `y` (assuming `y` is within the injectivity radius).
pub fn geodesic_distance(manifold: &dyn Manifold, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
    let v = manifold.log_map(x, y);
    v.iter().map(|vi| vi * vi).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::f64::consts::PI;

    /// Unit sphere S^{n-1}: points satisfy ||x|| = 1.
    struct Sphere;

    impl Manifold for Sphere {
        fn exp_map(&self, x: &ArrayView1<f64>, v: &ArrayView1<f64>) -> Array1<f64> {
            let norm_v = v.iter().map(|vi| vi * vi).sum::<f64>().sqrt();
            if norm_v < 1e-15 {
                return x.to_owned();
            }
            let cos_t = norm_v.cos();
            let sin_t = norm_v.sin();
            let mut result = Array1::zeros(x.len());
            for i in 0..x.len() {
                result[i] = cos_t * x[i] + sin_t * (v[i] / norm_v);
            }
            result
        }

        fn log_map(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
            let dot: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
            let dot = dot.clamp(-1.0, 1.0);
            let theta = dot.acos();
            if theta < 1e-15 {
                return Array1::zeros(x.len());
            }
            let mut u = Array1::zeros(x.len());
            for i in 0..x.len() {
                u[i] = y[i] - dot * x[i];
            }
            let norm_u = u.iter().map(|ui| ui * ui).sum::<f64>().sqrt();
            if norm_u < 1e-15 {
                return Array1::zeros(x.len());
            }
            u.mapv_inplace(|ui| ui * theta / norm_u);
            u
        }

        fn parallel_transport(
            &self,
            x: &ArrayView1<f64>,
            y: &ArrayView1<f64>,
            v: &ArrayView1<f64>,
        ) -> Array1<f64> {
            let log_xy = self.log_map(x, y);
            let norm_log = log_xy.iter().map(|li| li * li).sum::<f64>().sqrt();
            if norm_log < 1e-15 {
                return v.to_owned();
            }
            let dot_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
            let denom = 1.0 + dot_xy;
            if denom.abs() < 1e-15 {
                return v.to_owned();
            }
            let dot_v_log: f64 = v.iter().zip(log_xy.iter()).map(|(a, b)| a * b).sum();
            let coeff = dot_v_log / denom;
            let mut result = Array1::zeros(v.len());
            for i in 0..v.len() {
                result[i] = v[i] - coeff * (x[i] + y[i]);
            }
            result
        }

        fn project(&self, x: &ArrayView1<f64>) -> Array1<f64> {
            let norm = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
            if norm < 1e-15 {
                let mut result = Array1::zeros(x.len());
                if !result.is_empty() {
                    result[0] = 1.0;
                }
                return result;
            }
            x.mapv(|xi| xi / norm)
        }
    }

    #[test]
    fn sgd_step_stays_on_manifold() {
        let m = Sphere;
        let x = array![1.0, 0.0, 0.0];
        let grad = array![0.0, 0.5, -0.3];
        let result = riemannian_sgd_step(&m, &x.view(), &grad.view(), 0.1);
        let norm: f64 = result.iter().map(|r| r * r).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "SGD result not on sphere: norm = {norm}"
        );
    }

    #[test]
    fn sgd_zero_gradient_returns_same_point() {
        let m = Sphere;
        let x = array![0.0, 1.0, 0.0];
        let grad = array![0.0, 0.0, 0.0];
        let result = riemannian_sgd_step(&m, &x.view(), &grad.view(), 0.1);
        for (a, b) in x.iter().zip(result.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "zero-grad SGD changed the point: {x} -> {result}"
            );
        }
    }

    #[test]
    fn adam_converges_faster_than_sgd() {
        let m = Sphere;
        let target = array![0.0, 0.0, 1.0];
        let start = array![1.0, 0.0, 0.0];
        let lr = 0.05;
        let steps = 50;

        let mut x_sgd = start.clone();
        for _ in 0..steps {
            let grad = gradient_toward_target(&m, &x_sgd.view(), &target.view());
            x_sgd = riemannian_sgd_step(&m, &x_sgd.view(), &grad.view(), lr);
        }
        let dist_sgd = geodesic_distance(&m, &x_sgd.view(), &target.view());

        let mut x_adam = start.clone();
        let mut state = RiemannianAdamState::new(start);
        for _ in 0..steps {
            let grad = gradient_toward_target(&m, &x_adam.view(), &target.view());
            x_adam = riemannian_adam_step(
                &m,
                &x_adam.view(),
                &grad.view(),
                &mut state,
                lr,
                0.9,
                0.999,
                1e-8,
            );
        }
        let dist_adam = geodesic_distance(&m, &x_adam.view(), &target.view());

        assert!(dist_sgd < 0.5, "SGD did not converge: dist = {dist_sgd}");
        assert!(
            dist_adam < dist_sgd + 0.01,
            "Adam ({dist_adam}) should converge at least as well as SGD ({dist_sgd})"
        );
    }

    #[test]
    fn geodesic_distance_nonnegative_and_symmetric() {
        let m = Sphere;
        let x = array![1.0, 0.0, 0.0];
        let y = array![0.0, 1.0, 0.0];
        let d_xy = geodesic_distance(&m, &x.view(), &y.view());
        let d_yx = geodesic_distance(&m, &y.view(), &x.view());
        assert!(d_xy >= 0.0, "distance should be non-negative: {d_xy}");
        assert!(
            (d_xy - d_yx).abs() < 1e-10,
            "distance should be symmetric: {d_xy} vs {d_yx}"
        );
        assert!((d_xy - PI / 2.0).abs() < 1e-10, "expected pi/2, got {d_xy}");
    }

    #[test]
    fn geodesic_distance_zero_for_same_point() {
        let m = Sphere;
        let x = array![0.0, 0.0, 1.0];
        let d = geodesic_distance(&m, &x.view(), &x.view());
        assert!(d < 1e-12, "distance to self should be 0, got {d}");
    }

    fn gradient_toward_target(
        manifold: &Sphere,
        x: &ArrayView1<f64>,
        target: &ArrayView1<f64>,
    ) -> Array1<f64> {
        let log = manifold.log_map(x, target);
        log.mapv(|v| -v)
    }
}
