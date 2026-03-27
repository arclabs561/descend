/// Exponential moving average of model parameters.
///
/// Maintains a shadow copy of parameters that is updated as:
/// `shadow = decay * shadow + (1 - decay) * params`
///
/// Typical decay values: 0.999 to 0.9999 for stable training.
pub struct WeightAverager {
    shadow: Vec<f32>,
    decay: f32,
    initialized: bool,
}

impl WeightAverager {
    pub fn new(decay: f32) -> Self {
        Self {
            shadow: Vec::new(),
            decay,
            initialized: false,
        }
    }

    /// Update shadow with current parameters.
    pub fn update(&mut self, params: &[f32]) {
        if !self.initialized {
            self.shadow = params.to_vec();
            self.initialized = true;
            return;
        }

        assert_eq!(
            self.shadow.len(),
            params.len(),
            "Parameter size changed after initialization"
        );

        for (s, &p) in self.shadow.iter_mut().zip(params.iter()) {
            *s = self.decay * *s + (1.0 - self.decay) * p;
        }
    }

    /// Get averaged parameters.
    pub fn get(&self) -> &[f32] {
        &self.shadow
    }

    /// Copy averaged parameters into the given slice.
    pub fn copy_to(&self, dst: &mut [f32]) {
        assert_eq!(dst.len(), self.shadow.len());
        dst.copy_from_slice(&self.shadow);
    }

    /// Reset to uninitialized state.
    pub fn reset(&mut self) {
        self.shadow.clear();
        self.initialized = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_basic() {
        let mut ema = WeightAverager::new(0.9);

        ema.update(&[1.0, 2.0]);
        // First update: shadow = params
        assert_eq!(ema.get(), &[1.0, 2.0]);

        ema.update(&[3.0, 4.0]);
        // shadow = 0.9 * [1, 2] + 0.1 * [3, 4] = [1.2, 2.2]
        assert!((ema.get()[0] - 1.2).abs() < 1e-6);
        assert!((ema.get()[1] - 2.2).abs() < 1e-6);
    }

    #[test]
    fn test_ema_decay_1() {
        let mut ema = WeightAverager::new(1.0);

        ema.update(&[1.0]);
        ema.update(&[5.0]);
        ema.update(&[10.0]);

        // decay=1.0: shadow never changes after init
        assert!((ema.get()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ema_decay_0() {
        let mut ema = WeightAverager::new(0.0);

        ema.update(&[1.0]);
        ema.update(&[5.0]);
        ema.update(&[10.0]);

        // decay=0.0: shadow = latest params
        assert!((ema.get()[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_ema_reset() {
        let mut ema = WeightAverager::new(0.9);
        ema.update(&[1.0, 2.0]);

        ema.reset();
        assert!(ema.get().is_empty());
        assert!(!ema.initialized);
    }
}
