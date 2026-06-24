# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
