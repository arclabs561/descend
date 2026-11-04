# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `clip_grad_norm` accumulates the norm in `f64`. Finite gradients whose
  squared sum overflows `f32`, such as `[1e20, 1e20]`, were zeroed (the norm
  came out infinite, so the scale was zero); they are now rescaled to
  `max_norm`, and the returned norm is the finite original norm.
  `torch.nn.utils.clip_grad_norm_` 2.15.0 on float32 tensors still zeroes
  this input.
- `GradientAccumulator::accumulate` starts a new window on the call after it
  returns an average. Without an explicit `reset`, that call used to fold one
  more gradient into the previous average and return the result as a new
  average.
- `Lamb` and `Lion` panic with "parameter count changed; call reset before
  reusing an optimizer" when the parameter count changes, as `Sgd`, `Adam`,
  `Adagrad` and `RmsProp` do. A smaller count used to reuse stale moment
  state silently, and a larger one panicked on an index out of bounds.

## [0.1.2] - 2026-11-03

### Fixed

- Riemannian SGD and Adam project the Euclidean gradient onto the tangent
  space before the exponential map. A gradient with a normal component no
  longer sends a sphere point to its antipode.
- `Lion` leaves a parameter unchanged when its interpolated update is zero
  (`sign(0) = 0`); before, `f32::signum(0.0) = 1.0` stepped it.
- `WarmupCosine` no longer underflows when `total_steps < warmup_steps`; the
  learning rate holds at `eta_min` after warmup.
- `SequentialSchedule` keeps counting the final schedule's local step from the
  previous milestone past its own milestone, instead of restarting at zero.
- RAdam (`Adam::with_rectified(true)`) computes the rectification term in
  `f64`, which removes f32 cancellation for `beta2` near one.
- Optimizers panic with a message to call `reset` when the parameter count
  changes between steps.
- `OneCycleLr` now reaches `max_lr` at step `pct_start * total_steps - 1` and
  `min_lr` at step `total_steps - 1`, matching PyTorch's `OneCycleLR`; before,
  `min_lr` was reached only at step `total_steps`, which a `total_steps`-step
  run never executes.

## [0.1.1] - 2026-06-24

### Added

- `riemannian` feature: Riemannian SGD and Adam (`descend::riemannian`) for
  first-order optimization on any `skel::Manifold` (the optimizers `skel`
  documents as living here).
- Crate-level documentation, a `LICENSE` file, and a `CONTRIBUTING.md`.

## [0.1.0] - 2026-04-15

### Added

- Add repository settings
- Add RAdam, RMSprop, Lion, InverseSqrt, OneCycleLr, SequentialSchedule, EMA, gradient accumulation, Rosenbrock tests, training_loop example

### Changed

- Rename gradstep -> descend
- Initial crate -- training infrastructure primitives
