# descend

[![crates.io](https://img.shields.io/crates/v/descend.svg)](https://crates.io/crates/descend)
[![Documentation](https://docs.rs/descend/badge.svg)](https://docs.rs/descend)
[![CI](https://github.com/arclabs561/descend/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/descend/actions/workflows/ci.yml)

Training infrastructure primitives: optimizers, LR schedules, gradient clipping, early stopping.

## Quickstart

```toml
[dependencies]
descend = "0.1"
```

```rust
use descend::{Adam, Optimizer, WarmupCosine, ScheduledOptimizer, LrSchedule};

// Adam optimizer
let mut opt = Adam::new(1e-3, 0.9, 0.999, 1e-8, 0.0);
let grads = vec![vec![0.1, -0.2]];
let params = vec![vec![1.0, 2.0]];
let updated = opt.step(&params, &grads);

// LR schedule
let schedule = WarmupCosine::new(1000, 1e-3, 100);
let lr = schedule.lr(500);
```

## Features

- Optimizers: SGD, Adam, AdaGrad, RMSProp, Lion, LAMB
- LR schedules: linear warmup, cosine annealing, one-cycle, step decay, inverse sqrt
- Gradient clipping: by norm and by value
- Early stopping with patience and delta
- Gradient accumulation
- Exponential moving average

## License

MIT OR Apache-2.0
