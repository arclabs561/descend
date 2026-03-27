//! Complete training loop minimizing the Rosenbrock function.
//!
//! Demonstrates: Adam + WarmupCosine schedule + gradient clipping + EMA +
//! early stopping + training metrics.

use descend::{
    Adam, EarlyStopping, Optimizer, ScheduledOptimizer, StopMode, TrainingMetrics, WarmupCosine,
    WeightAverager, clip_grad_norm,
};

fn rosenbrock_value(params: &[f32]) -> f32 {
    let x = params[0];
    let y = params[1];
    (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
}

fn rosenbrock_grads(params: &[f32]) -> Vec<f32> {
    let x = params[0];
    let y = params[1];
    let dx = -2.0 * (1.0 - x) + 200.0 * (y - x * x) * (-2.0 * x);
    let dy = 200.0 * (y - x * x);
    vec![dx, dy]
}

fn main() {
    let total_steps = 5000;
    let warmup_steps = 500;

    let adam = Adam::new(0.001);
    let schedule = WarmupCosine {
        warmup_steps,
        total_steps,
        eta_min: 1e-6,
    };
    let mut opt = ScheduledOptimizer::new(adam, schedule);

    let mut params = vec![-1.5, 2.0];
    let mut ema = WeightAverager::new(0.999);
    let mut early_stop = EarlyStopping::new(500, 1e-8, StopMode::Min);
    let mut metrics = TrainingMetrics::new();

    println!("Step  | Loss         | LR          | Params");
    println!("------+--------------+-------------+-------------------");

    for step in 0..total_steps {
        let mut grads = rosenbrock_grads(&params);
        clip_grad_norm(&mut grads, 10.0);

        opt.step(&mut params, &grads);
        ema.update(&params);

        let loss = rosenbrock_value(&params) as f64;
        metrics.record("loss", loss);

        if step % 500 == 0 || step == total_steps - 1 {
            println!(
                "{:5} | {:12.6} | {:11.2e} | ({:.4}, {:.4})",
                step,
                loss,
                opt.lr(),
                params[0],
                params[1]
            );
        }

        if early_stop.should_stop(loss) {
            println!("Early stopping at step {}", step);
            break;
        }
    }

    let summary = metrics.epoch_summary();
    let loss_summary = &summary["loss"];
    println!();
    println!(
        "Loss -- mean: {:.6}, min: {:.6}, max: {:.6}, last: {:.6}",
        loss_summary.mean, loss_summary.min, loss_summary.max, loss_summary.last
    );

    let ema_params = ema.get();
    let ema_loss = rosenbrock_value(ema_params);
    println!(
        "EMA params: ({:.4}, {:.4}), loss: {:.6}",
        ema_params[0], ema_params[1], ema_loss
    );
}
