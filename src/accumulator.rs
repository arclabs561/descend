/// Gradient accumulation for simulating large batches.
///
/// Accumulates gradients over `micro_steps` calls, then returns the averaged
/// result. Useful when the desired batch size exceeds GPU memory.
pub struct GradientAccumulator {
    accum: Vec<f32>,
    micro_steps: usize,
    count: usize,
}

impl GradientAccumulator {
    pub fn new(micro_steps: usize) -> Self {
        assert!(micro_steps > 0, "micro_steps must be positive");
        Self {
            accum: Vec::new(),
            micro_steps,
            count: 0,
        }
    }

    /// Accumulate gradients. Returns `Some(&averaged_grads)` when `micro_steps` reached.
    pub fn accumulate(&mut self, grads: &[f32]) -> Option<&[f32]> {
        if self.accum.is_empty() {
            self.accum = vec![0.0; grads.len()];
        }

        assert_eq!(self.accum.len(), grads.len());

        for (a, &g) in self.accum.iter_mut().zip(grads.iter()) {
            *a += g;
        }
        self.count += 1;

        if self.count >= self.micro_steps {
            let scale = 1.0 / self.micro_steps as f32;
            for a in self.accum.iter_mut() {
                *a *= scale;
            }
            // Return the averaged grads; caller must call reset() after consuming.
            Some(&self.accum)
        } else {
            None
        }
    }

    /// Reset accumulator.
    pub fn reset(&mut self) {
        for a in self.accum.iter_mut() {
            *a = 0.0;
        }
        self.count = 0;
    }

    /// Current accumulation count.
    pub fn count(&self) -> usize {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_accumulator_basic() {
        let mut acc = GradientAccumulator::new(4);

        assert!(acc.accumulate(&[1.0, 2.0]).is_none());
        assert!(acc.accumulate(&[3.0, 4.0]).is_none());
        assert!(acc.accumulate(&[5.0, 6.0]).is_none());
        let result = acc.accumulate(&[7.0, 8.0]);

        assert!(result.is_some());
        let avg = result.unwrap();
        // (1+3+5+7)/4 = 4.0, (2+4+6+8)/4 = 5.0
        assert!((avg[0] - 4.0).abs() < 1e-6);
        assert!((avg[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_accumulator_partial() {
        let mut acc = GradientAccumulator::new(3);

        assert!(acc.accumulate(&[1.0]).is_none());
        assert_eq!(acc.count(), 1);
        assert!(acc.accumulate(&[2.0]).is_none());
        assert_eq!(acc.count(), 2);
        // Not yet at micro_steps=3
    }

    #[test]
    fn test_gradient_accumulator_reset() {
        let mut acc = GradientAccumulator::new(2);

        acc.accumulate(&[1.0, 2.0]);
        acc.accumulate(&[3.0, 4.0]);
        assert_eq!(acc.count(), 2);

        acc.reset();
        assert_eq!(acc.count(), 0);

        // After reset, can accumulate fresh
        assert!(acc.accumulate(&[10.0, 20.0]).is_none());
        let result = acc.accumulate(&[30.0, 40.0]);
        assert!(result.is_some());
        let avg = result.unwrap();
        assert!((avg[0] - 20.0).abs() < 1e-6);
        assert!((avg[1] - 30.0).abs() < 1e-6);
    }
}
