//! Bounded time-series buffers behind every chart and sparkline.

use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct History {
    buf: VecDeque<f64>,
    cap: usize,
}

impl History {
    pub fn new(cap: usize) -> Self {
        Self { buf: VecDeque::with_capacity(cap.max(1)), cap: cap.max(1) }
    }

    pub fn push(&mut self, v: f64) {
        let v = if v.is_finite() { v } else { 0.0 };
        while self.buf.len() >= self.cap {
            self.buf.pop_front();
        }
        self.buf.push_back(v);
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn last(&self) -> f64 {
        self.buf.back().copied().unwrap_or(0.0)
    }

    pub fn max(&self) -> f64 {
        self.buf.iter().copied().fold(0.0, f64::max)
    }

    pub fn iter(&self) -> impl Iterator<Item = &f64> {
        self.buf.iter()
    }

    /// `(x, y)` pairs for `ratatui::Chart`, oldest at x = 0.
    pub fn points(&self) -> Vec<(f64, f64)> {
        self.buf.iter().enumerate().map(|(i, y)| (i as f64, *y)).collect()
    }

    /// Last `n` values as `u64`, for `ratatui::Sparkline`.
    pub fn sparkline(&self, n: usize) -> Vec<u64> {
        let skip = self.buf.len().saturating_sub(n);
        self.buf.iter().skip(skip).map(|v| v.max(0.0) as u64).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_history_is_empty_rather_than_a_flat_line_of_zeros() {
        // The old code pre-filled 120 zeros, so every chart opened with a
        // misleading flat line across the whole window.
        let h = History::new(4);
        assert!(h.is_empty());
        assert!(h.points().is_empty());
        assert_eq!(h.last(), 0.0);
    }

    #[test]
    fn pushing_past_capacity_drops_the_oldest_sample() {
        let mut h = History::new(3);
        for v in [1.0, 2.0, 3.0, 4.0] {
            h.push(v);
        }
        assert_eq!(h.len(), 3);
        assert_eq!(h.iter().copied().collect::<Vec<_>>(), vec![2.0, 3.0, 4.0]);
        assert_eq!(h.last(), 4.0);
        assert_eq!(h.max(), 4.0);
    }

    #[test]
    fn non_finite_samples_are_normalised_to_zero() {
        let mut h = History::new(3);
        h.push(f64::NAN);
        h.push(f64::INFINITY);
        assert_eq!(h.iter().copied().collect::<Vec<_>>(), vec![0.0, 0.0]);
    }

    #[test]
    fn points_are_indexed_from_the_oldest_sample() {
        let mut h = History::new(5);
        h.push(10.0);
        h.push(20.0);
        assert_eq!(h.points(), vec![(0.0, 10.0), (1.0, 20.0)]);
    }

    #[test]
    fn sparkline_returns_at_most_the_requested_width() {
        let mut h = History::new(10);
        for i in 0..10 {
            h.push(i as f64);
        }
        assert_eq!(h.sparkline(3), vec![7, 8, 9]);
        assert_eq!(h.sparkline(100).len(), 10);
    }

    #[test]
    fn zero_capacity_is_clamped_instead_of_panicking() {
        let mut h = History::new(0);
        h.push(1.0);
        assert_eq!(h.len(), 1);
    }
}
