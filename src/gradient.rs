/// Clip gradients by global norm. Returns the original norm before clipping.
pub fn clip_grad_norm(grads: &mut [f32], max_norm: f32) -> f32 {
    let norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();

    if norm > max_norm {
        let scale = max_norm / norm;
        for g in grads.iter_mut() {
            *g *= scale;
        }
    }

    norm
}

/// Clip each gradient element to `[-max_value, max_value]`.
pub fn clip_grad_value(grads: &mut [f32], max_value: f32) {
    for g in grads.iter_mut() {
        *g = g.clamp(-max_value, max_value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_norm() {
        let mut grads = vec![3.0, 4.0]; // norm = 5.0
        let original_norm = clip_grad_norm(&mut grads, 1.0);

        assert!((original_norm - 5.0).abs() < 1e-6);

        let clipped_norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!(
            (clipped_norm - 1.0).abs() < 1e-6,
            "clipped norm was {}",
            clipped_norm
        );

        // Direction preserved
        assert!((grads[0] / grads[1] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_clip_norm_no_op() {
        let mut grads = vec![0.3, 0.4]; // norm = 0.5
        let original = grads.clone();
        let norm = clip_grad_norm(&mut grads, 1.0);

        assert!((norm - 0.5).abs() < 1e-6);
        assert_eq!(grads, original);
    }

    #[test]
    fn test_clip_value() {
        let mut grads = vec![-5.0, 3.0, 0.5, -0.1, 10.0];
        clip_grad_value(&mut grads, 2.0);

        for &g in &grads {
            assert!((-2.0..=2.0).contains(&g), "grad {} out of range", g);
        }
        assert_eq!(grads[2], 0.5); // within range, unchanged
        assert_eq!(grads[3], -0.1); // within range, unchanged
    }
}
