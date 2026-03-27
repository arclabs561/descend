/// Whether to minimize or maximize the tracked metric.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StopMode {
    Min,
    Max,
}

/// Early stopping tracker with patience and minimum delta.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EarlyStopping {
    patience: usize,
    min_delta: f64,
    mode: StopMode,
    best_value: Option<f64>,
    counter: usize,
    stopped: bool,
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: f64, mode: StopMode) -> Self {
        Self {
            patience,
            min_delta,
            mode,
            best_value: None,
            counter: 0,
            stopped: false,
        }
    }

    /// Record a metric value. Returns true if training should stop.
    pub fn should_stop(&mut self, metric: f64) -> bool {
        if self.stopped {
            return true;
        }

        if self.is_improvement(metric) {
            self.best_value = Some(metric);
            self.counter = 0;
        } else {
            self.counter += 1;
            if self.counter >= self.patience {
                self.stopped = true;
                return true;
            }
        }

        false
    }

    /// Whether the given metric value is the best seen so far.
    pub fn is_best(&self, metric: f64) -> bool {
        self.is_improvement(metric)
    }

    /// The best metric value seen, if any.
    pub fn best_value(&self) -> Option<f64> {
        self.best_value
    }

    /// Current patience counter (steps since last improvement).
    pub fn counter(&self) -> usize {
        self.counter
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        self.best_value = None;
        self.counter = 0;
        self.stopped = false;
    }

    fn is_improvement(&self, metric: f64) -> bool {
        match self.best_value {
            None => true,
            Some(best) => match self.mode {
                StopMode::Min => metric < best - self.min_delta,
                StopMode::Max => metric > best + self.min_delta,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patience() {
        let mut es = EarlyStopping::new(3, 0.0, StopMode::Min);

        assert!(!es.should_stop(1.0)); // first value, sets best
        assert!(!es.should_stop(0.9)); // improvement
        assert!(!es.should_stop(0.95)); // worse, counter=1
        assert!(!es.should_stop(0.95)); // worse, counter=2
        assert!(es.should_stop(0.95)); // worse, counter=3, stop
    }

    #[test]
    fn test_min_delta() {
        let mut es = EarlyStopping::new(3, 0.1, StopMode::Min);

        assert!(!es.should_stop(1.0)); // sets best=1.0
        assert!(!es.should_stop(0.95)); // improvement below delta, counter=1
        // 0.95 < 1.0 but not < 1.0 - 0.1 = 0.9, so not a real improvement
        assert!(!es.should_stop(0.85)); // 0.85 < 0.9, real improvement, resets
        assert_eq!(es.counter(), 0);
    }

    #[test]
    fn test_is_best() {
        let mut es = EarlyStopping::new(3, 0.0, StopMode::Max);

        es.should_stop(0.5);
        assert!(!es.is_best(0.3));
        assert!(es.is_best(0.6));
    }

    #[test]
    fn test_reset_early_stopping() {
        let mut es = EarlyStopping::new(3, 0.0, StopMode::Min);

        es.should_stop(1.0);
        es.should_stop(1.1);
        es.should_stop(1.1);
        assert_eq!(es.counter(), 2);

        es.reset();
        assert_eq!(es.counter(), 0);
        assert!(es.best_value().is_none());
    }
}
