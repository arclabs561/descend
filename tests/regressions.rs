//! Regression tests for gradient clipping, accumulation, and optimizer state.

use descend::{GradientAccumulator, Lamb, Lion, Optimizer, clip_grad_norm};

/// `clip_grad_norm` returns the original norm. Each element and the norm of
/// `[1e20, 1e20]` (about 1.414e20) are finite f32 values, though the squared
/// sum overflows f32, so the clipped gradient keeps its direction at norm 1.
#[test]
fn clip_grad_norm_rescales_finite_gradients_whose_square_overflows_f32() {
    let mut g = vec![1e20f32, 1e20];
    let norm = clip_grad_norm(&mut g, 1.0);
    assert!(
        (norm / 1.414_213_6e20 - 1.0).abs() < 1e-6,
        "returned norm {norm}"
    );
    let after = (g[0] * g[0] + g[1] * g[1]).sqrt();
    assert!((after - 1.0).abs() < 1e-6, "clipped grads {g:?}");
    assert_eq!(g[0], g[1]);
}

/// The accumulator averages over `micro_steps` calls. After it returns an
/// average, the next call starts a new window instead of folding one more
/// gradient into the previous average.
#[test]
fn accumulator_starts_a_new_window_after_returning_an_average() {
    let mut acc = GradientAccumulator::new(2);
    assert!(acc.accumulate(&[1.0]).is_none());
    assert_eq!(acc.accumulate(&[3.0]).map(<[f32]>::to_vec), Some(vec![2.0]));
    assert!(acc.accumulate(&[10.0]).is_none());
    assert_eq!(acc.count(), 1);
    assert_eq!(
        acc.accumulate(&[20.0]).map(<[f32]>::to_vec),
        Some(vec![15.0])
    );
}

/// Calling `reset` after consuming an average still gives a fresh window.
#[test]
fn accumulator_reset_after_an_average_still_starts_a_new_window() {
    let mut acc = GradientAccumulator::new(2);
    acc.accumulate(&[1.0]);
    acc.accumulate(&[3.0]);
    acc.reset();
    assert!(acc.accumulate(&[10.0]).is_none());
    assert_eq!(
        acc.accumulate(&[20.0]).map(<[f32]>::to_vec),
        Some(vec![15.0])
    );
}

/// Sgd, Adam, Adagrad, and RmsProp panic with "parameter count changed; call
/// reset before reusing an optimizer". Lamb and Lion keep per-parameter state
/// the same way and must not reuse it for a different parameter count.
#[test]
#[should_panic(expected = "parameter count changed")]
fn lamb_rejects_a_parameter_count_change_without_reset() {
    let mut opt = Lamb::new(0.1);
    opt.step(&mut [1.0, 2.0], &[0.5, 0.25]);
    opt.step(&mut [1.0], &[0.5]);
}

#[test]
#[should_panic(expected = "parameter count changed")]
fn lion_rejects_a_parameter_count_change_without_reset() {
    let mut opt = Lion::new(0.1);
    opt.step(&mut [1.0, 2.0], &[0.5, 0.25]);
    opt.step(&mut [1.0], &[0.5]);
}

#[test]
fn lamb_and_lion_accept_a_new_parameter_count_after_reset() {
    let mut lamb = Lamb::new(0.1);
    lamb.step(&mut [1.0, 2.0], &[0.5, 0.25]);
    lamb.reset();
    lamb.step(&mut [1.0], &[0.5]);

    let mut lion = Lion::new(0.1);
    lion.step(&mut [1.0, 2.0], &[0.5, 0.25]);
    lion.reset();
    lion.step(&mut [1.0], &[0.5]);
}
