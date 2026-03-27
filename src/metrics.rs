use std::collections::HashMap;

/// Summary statistics for a single metric.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MetricSummary {
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub last: f64,
    pub count: usize,
}

/// Accumulator for training metrics across steps/epochs.
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    values: HashMap<String, Vec<f64>>,
}

impl TrainingMetrics {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Record a value for a named metric.
    pub fn record(&mut self, name: &str, value: f64) {
        self.values.entry(name.to_string()).or_default().push(value);
    }

    /// Compute summary statistics for all recorded metrics.
    pub fn epoch_summary(&self) -> HashMap<String, MetricSummary> {
        self.values
            .iter()
            .map(|(name, vals)| {
                let count = vals.len();
                let sum: f64 = vals.iter().sum();
                let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
                let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let summary = MetricSummary {
                    mean: sum / count as f64,
                    min,
                    max,
                    last: vals[count - 1],
                    count,
                };
                (name.clone(), summary)
            })
            .collect()
    }

    /// Compute exponential moving average for a named metric.
    pub fn ema(&self, name: &str, alpha: f64) -> Option<f64> {
        let vals = self.values.get(name)?;
        if vals.is_empty() {
            return None;
        }

        let mut ema = vals[0];
        for &v in &vals[1..] {
            ema = alpha * v + (1.0 - alpha) * ema;
        }
        Some(ema)
    }

    /// Get raw values for a named metric.
    pub fn get(&self, name: &str) -> Option<&[f64]> {
        self.values.get(name).map(|v| v.as_slice())
    }

    /// Clear all recorded metrics.
    pub fn clear(&mut self) {
        self.values.clear();
    }
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_summary() {
        let mut m = TrainingMetrics::new();
        m.record("loss", 1.0);
        m.record("loss", 0.5);
        m.record("loss", 0.3);
        m.record("loss", 0.2);

        let summary = m.epoch_summary();
        let loss = &summary["loss"];

        assert_eq!(loss.count, 4);
        assert!((loss.mean - 0.5).abs() < 1e-6);
        assert!((loss.min - 0.2).abs() < 1e-6);
        assert!((loss.max - 1.0).abs() < 1e-6);
        assert!((loss.last - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_ema() {
        let mut m = TrainingMetrics::new();
        m.record("loss", 1.0);
        m.record("loss", 0.5);
        m.record("loss", 0.3);

        let ema = m.ema("loss", 0.5).unwrap();
        // EMA: start=1.0, then 0.5*0.5 + 0.5*1.0 = 0.75, then 0.5*0.3 + 0.5*0.75 = 0.525
        assert!((ema - 0.525).abs() < 1e-6, "ema was {}", ema);

        assert!(m.ema("nonexistent", 0.5).is_none());
    }

    #[test]
    fn test_clear() {
        let mut m = TrainingMetrics::new();
        m.record("loss", 1.0);
        m.record("acc", 0.5);

        m.clear();

        assert!(m.get("loss").is_none());
        assert!(m.get("acc").is_none());
        assert!(m.epoch_summary().is_empty());
    }
}
