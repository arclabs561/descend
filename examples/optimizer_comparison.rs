//! Optimizer comparison on a convex bowl with a known minimum.
//!
//! `descend`'s optimizers operate on plain `&mut [f32]` parameter slices behind
//! the [`Optimizer`](descend::Optimizer) trait, with no tensor framework,
//! autodiff, or `Module` abstraction. This example exploits that: it drops
//! several optimizers into the *same* hand-written loop as boxed trait objects,
//! minimizes `f(x) = sum (x_i - t_i)^2` (global minimum `f = 0` at `x = t`), and
//! asserts each one reaches the known optimum. So the example doubles as a
//! smoke test for the optimizer roster.
//!
//! Run with: `cargo run --example optimizer_comparison`

use descend::{Adagrad, Adam, Lion, Optimizer, RmsProp, Sgd};

/// Center of the bowl. The global minimum is `f(target) = 0`.
const TARGET: [f32; 3] = [3.0, -2.0, 0.5];

fn loss(params: &[f32]) -> f32 {
    params
        .iter()
        .zip(TARGET)
        .map(|(&x, t)| (x - t).powi(2))
        .sum()
}

fn grads(params: &[f32]) -> Vec<f32> {
    params
        .iter()
        .zip(TARGET)
        .map(|(&x, t)| 2.0 * (x - t))
        .collect()
}

/// Run a boxed optimizer on the bowl from a fixed start and report the final
/// parameters and loss.
fn run(mut opt: Box<dyn Optimizer>, steps: usize) -> (Vec<f32>, f32) {
    let mut params = vec![0.0f32; TARGET.len()];
    for _ in 0..steps {
        let g = grads(&params);
        opt.step(&mut params, &g);
    }
    let final_loss = loss(&params);
    (params, final_loss)
}

fn main() {
    // (name, optimizer, steps, loss tolerance). Step budgets differ because
    // sign-based (Lion) and slowly-adapting (RMSprop, Adagrad) methods need
    // more iterations to reach the same accuracy on this bowl.
    let configs: Vec<(&str, Box<dyn Optimizer>, usize, f32)> = vec![
        ("SGD", Box::new(Sgd::new(0.1)), 200, 1e-4),
        (
            "SGD+mom",
            Box::new(Sgd::new(0.05).with_momentum(0.9)),
            500,
            1e-4,
        ),
        ("Adam", Box::new(Adam::new(0.1)), 500, 1e-3),
        ("Adagrad", Box::new(Adagrad::new(1.0)), 2000, 1e-2),
        ("RMSprop", Box::new(RmsProp::new(0.01)), 2000, 5e-2),
        ("Lion", Box::new(Lion::new(0.01)), 3000, 5e-2),
    ];

    println!("Minimizing f(x) = sum (x_i - t_i)^2,  t = {TARGET:?}");
    println!("global minimum: f = 0 at x = t\n");
    println!(
        "{:<8} | {:>6} | {:>12} | params",
        "optimizer", "steps", "final loss"
    );
    println!("{:-<8}-+-{:-<6}-+-{:-<12}-+--------", "", "", "");

    for (name, opt, steps, tol) in configs {
        let (params, final_loss) = run(opt, steps);
        println!(
            "{name:<8} | {steps:>6} | {final_loss:>12.3e} | [{:.4}, {:.4}, {:.4}]",
            params[0], params[1], params[2]
        );
        assert!(
            final_loss < tol,
            "{name} did not reach the known minimum: loss={final_loss}, tol={tol}"
        );
    }

    println!("\nAll optimizers converged to the known minimum at x = {TARGET:?}.");
}
