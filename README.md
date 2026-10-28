# descend

[![crates.io](https://img.shields.io/crates/v/descend.svg)](https://crates.io/crates/descend)
[![Documentation](https://docs.rs/descend/badge.svg)](https://docs.rs/descend)

Training loop primitives for gradient-based optimization.

`descend` updates `f32` parameter slices with gradients you supply. Use it in
training loops where you manage the parameters and gradient computation.

## Quickstart

```toml
[dependencies]
descend = "0.1"
```

```rust
use descend::{Adam, Optimizer, WarmupCosine, LrSchedule};

// Adam: new takes the learning rate; builders set the rest.
let mut opt = Adam::new(1e-3).with_betas(0.9, 0.999);
let mut params = vec![1.0, 2.0];
let grads = vec![0.1, -0.2];
opt.step(&mut params, &grads); // updates params in place
assert!(params[0] < 1.0 && params[1] > 2.0);

// Warmup + cosine LR schedule.
let schedule = WarmupCosine { warmup_steps: 100, total_steps: 1000, eta_min: 0.0 };
let lr = schedule.lr_at(500, 1e-3);
opt.set_lr(lr); // use the scheduled rate for the next step
```

## Features

- Optimizers: SGD, Adam, AdaGrad, RMSProp, Lion, LAMB
- LR schedules: linear warmup, cosine annealing, one-cycle, step decay, inverse sqrt
- Gradient clipping: by norm and by value
- Early stopping with patience and delta
- Gradient accumulation
- Exponential moving average
- Riemannian SGD and Adam over `skel::Manifold` (feature `riemannian`)

Enable the optional `serde` feature to serialize supported optimizer and schedule types.

## License

MIT OR Apache-2.0
