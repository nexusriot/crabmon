//! crabmon — a keyboard-driven terminal system monitor.
//!
//! The binary is a thin shell around this library: `App` owns all state and
//! input handling, `metrics` produces `Snapshot`s, and `ui` renders them. That
//! split is what lets the test-suite drive the whole program headlessly.

pub mod alerts;
pub mod app;
pub mod audit;
pub mod cli;
pub mod clipboard;
pub mod config;
pub mod diff;
pub mod export;
pub mod filter;
pub mod format;
pub mod history;
pub mod metrics;
pub mod record;
pub mod remote;
pub mod sampler;
pub mod serve;
pub mod sort;
pub mod theme;
pub mod tree;
pub mod ui;
pub mod watch;

/// sysinfo needs 200 ms between CPU samples for the numbers to mean anything.
pub const MIN_REFRESH_MS: u64 = 200;
pub const MAX_REFRESH_MS: u64 = 10_000;

pub use app::{Action, App, Mode};
pub use config::Config;
pub use metrics::{MetricSource, Snapshot};

#[cfg(test)]
mod tests {
    #[test]
    fn the_refresh_floor_is_not_below_what_sysinfo_requires() {
        assert!(
            std::time::Duration::from_millis(crate::MIN_REFRESH_MS)
                >= sysinfo::MINIMUM_CPU_UPDATE_INTERVAL
        );
    }
}
