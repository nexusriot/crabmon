//! Sampling on its own thread.
//!
//! `App::tick()` used to call the metric source directly from the event loop,
//! so anything slow in a sample froze the whole interface — not just the
//! numbers, but the keyboard, including the key that quits. An SSH round trip,
//! an `nvidia-smi` that takes its time, or a `statvfs` on a wedged NFS mount
//! were each enough.
//!
//! `ThreadedSource` moves the source onto a worker and hands the loop whatever
//! the most recent completed sample was. The UI is then never slower than a
//! redraw, and a stalled source shows stale numbers rather than a dead program.

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::time::{Duration, Instant};

use crate::metrics::{MetricSource, Snapshot};

struct Update {
    snap: Snapshot,
    label: Option<String>,
    /// Why the inner source could not sample, when it could not. Carried
    /// alongside the label because the worker is the only thing that can ask
    /// it, and the fleet view needs the answer rather than a sentence with
    /// the answer somewhere inside it.
    error: Option<String>,
}

pub struct ThreadedSource {
    requests: Sender<()>,
    updates: Receiver<Update>,
    latest: Snapshot,
    label: Option<String>,
    error: Option<String>,
    /// A sample has been asked for and has not come back.
    pending: bool,
    /// Incremented for every frame that actually arrives, so the caller can
    /// tell a genuinely new sample from the same one handed out again.
    frames: u64,
    /// The worker died; nothing more will ever arrive.
    finished: bool,
}

impl ThreadedSource {
    /// Takes the first sample synchronously, so the interface opens with real
    /// numbers rather than a blank dashboard for one refresh interval.
    pub fn new(mut inner: Box<dyn MetricSource + Send>) -> ThreadedSource {
        let first = inner.snapshot(Duration::ZERO);
        let label = inner.label();
        let error = inner.error();
        let (req_tx, req_rx) = mpsc::channel::<()>();
        let (upd_tx, upd_rx) = mpsc::channel::<Update>();

        std::thread::spawn(move || {
            let mut last = Instant::now();
            // Ends when the source is dropped and `requests` closes.
            while req_rx.recv().is_ok() {
                let dt = last.elapsed();
                last = Instant::now();
                let snap = inner.snapshot(dt);
                let label = inner.label();
                let error = inner.error();
                if upd_tx.send(Update { snap, label, error }).is_err() {
                    return;
                }
            }
        });

        ThreadedSource {
            requests: req_tx,
            updates: upd_rx,
            latest: first,
            label,
            error,
            pending: false,
            frames: 1,
            finished: false,
        }
    }

    /// Take everything the worker has finished, keeping only the newest. A
    /// source that fell behind must not replay a backlog of stale frames.
    fn drain(&mut self) {
        loop {
            match self.updates.try_recv() {
                Ok(update) => {
                    self.latest = update.snap;
                    self.label = update.label;
                    self.error = update.error;
                    self.frames += 1;
                    self.pending = false;
                }
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    self.finished = true;
                    self.pending = false;
                    return;
                }
            }
        }
    }

    pub fn is_sampling(&self) -> bool {
        self.pending
    }
}

impl MetricSource for ThreadedSource {
    fn snapshot(&mut self, _dt: Duration) -> Snapshot {
        self.drain();
        // One outstanding request at a time: asking again while the worker is
        // still busy would queue work it can never catch up on.
        if !self.pending && !self.finished && self.requests.send(()).is_ok() {
            self.pending = true;
        }
        self.latest.clone()
    }

    fn frame_id(&self) -> Option<u64> {
        Some(self.frames)
    }

    fn label(&self) -> Option<String> {
        self.label.clone()
    }
}

/// Paces a headless sampling loop and reports the *real* interval between
/// samples.
///
/// Every rate crabmon publishes is a delta of a monotonic counter divided by
/// the time that passed between the two readings, so the figure handed to
/// `MetricSource::snapshot` has to be that time and not the interval that was
/// asked for. Sampling is not free — on a busy machine, or with `[procs] fds`
/// on, a sample costs a sizeable fraction of a refresh — so a loop that sleeps
/// the whole interval *and then* samples runs slower than it claims, and
/// dividing by the nominal interval inflates every byte-per-second series by
/// `(interval + sample_cost) / interval`. `--serve` already measured the gap
/// between scrapes; `--stream`, `--once` and `--watch` divided by the nominal
/// figure.
///
/// Marking the time just before each sample fixes both halves at once: the
/// reported interval is the distance between two sampling windows, and the
/// sleep is only whatever is left of the interval, so the cadence is the
/// configured one rather than the interval plus the cost of a sample. When a
/// sample costs more than the interval the loop simply runs as fast as it can
/// and reports the longer gap it actually achieved.
#[derive(Debug, Clone)]
pub struct Pacer {
    interval: Duration,
    last: Instant,
}

impl Pacer {
    /// Starts the clock now, i.e. immediately after the priming sample.
    pub fn new(interval: Duration) -> Pacer {
        Pacer::starting_at(interval, Instant::now())
    }

    /// The same, from an explicit origin, so the pacing can be asserted on a
    /// fixed timeline rather than on how long a test itself took.
    pub fn starting_at(interval: Duration, at: Instant) -> Pacer {
        Pacer { interval, last: at }
    }

    /// How much of the interval is left to sleep before the next sample.
    pub fn remaining(&self, now: Instant) -> Duration {
        self.interval.saturating_sub(now.saturating_duration_since(self.last))
    }

    /// Record that a sample is being taken at `now`, and return the real time
    /// since the previous one.
    pub fn mark(&mut self, now: Instant) -> Duration {
        let dt = now.saturating_duration_since(self.last);
        self.last = now;
        dt
    }

    /// Sleep out the rest of the interval, then mark. The returned duration is
    /// what belongs in `MetricSource::snapshot`.
    pub fn wait(&mut self) -> Duration {
        std::thread::sleep(self.remaining(Instant::now()));
        self.mark(Instant::now())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A source that takes `delay` to produce each frame, counting up.
    struct SlowSource {
        delay: Duration,
        n: u64,
    }

    impl MetricSource for SlowSource {
        fn snapshot(&mut self, _dt: Duration) -> Snapshot {
            std::thread::sleep(self.delay);
            self.n += 1;
            Snapshot { taken_at_unix: self.n, ..Default::default() }
        }
        fn label(&self) -> Option<String> {
            Some(format!("slow {}", self.n))
        }
    }

    fn wait_for(source: &mut ThreadedSource, want: u64) -> Snapshot {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let snap = source.snapshot(Duration::ZERO);
            if snap.taken_at_unix >= want || Instant::now() > deadline {
                return snap;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn the_first_frame_is_ready_before_the_constructor_returns() {
        // Otherwise the dashboard opens blank for a whole refresh interval.
        let mut src = ThreadedSource::new(Box::new(SlowSource { delay: Duration::ZERO, n: 0 }));
        assert_eq!(src.snapshot(Duration::ZERO).taken_at_unix, 1);
        assert_eq!(src.frame_id(), Some(1));
    }

    #[test]
    fn a_slow_sample_never_blocks_the_caller() {
        // This is the whole point: a source that takes 300 ms must not make the
        // event loop wait 300 ms to redraw or read a keystroke.
        let mut src =
            ThreadedSource::new(Box::new(SlowSource { delay: Duration::from_millis(300), n: 0 }));
        let start = Instant::now();
        for _ in 0..20 {
            src.snapshot(Duration::ZERO);
        }
        assert!(start.elapsed() < Duration::from_millis(200), "took {:?}", start.elapsed());
    }

    #[test]
    fn the_frame_id_only_moves_when_a_new_sample_actually_lands() {
        let mut src =
            ThreadedSource::new(Box::new(SlowSource { delay: Duration::from_millis(80), n: 0 }));
        let before = src.frame_id();
        src.snapshot(Duration::ZERO);
        assert_eq!(src.frame_id(), before, "nothing has come back yet");
        assert!(src.is_sampling());

        wait_for(&mut src, 2);
        assert!(src.frame_id().unwrap() > before.unwrap());
    }

    #[test]
    fn a_backlog_collapses_to_the_newest_frame() {
        let mut src = ThreadedSource::new(Box::new(SlowSource { delay: Duration::ZERO, n: 0 }));
        // Pump several samples through, then check we are not replaying them.
        let latest = wait_for(&mut src, 5);
        assert!(latest.taken_at_unix >= 5);
        let again = src.snapshot(Duration::ZERO);
        assert!(again.taken_at_unix >= latest.taken_at_unix, "never goes backwards");
    }

    #[test]
    fn the_label_follows_the_source_so_an_unreachable_host_still_says_so() {
        let mut src = ThreadedSource::new(Box::new(SlowSource { delay: Duration::ZERO, n: 0 }));
        assert_eq!(src.label().as_deref(), Some("slow 1"));
        wait_for(&mut src, 3);
        assert_ne!(src.label().as_deref(), Some("slow 1"), "the label must track the source");
    }

    /// The bug this exists for: a loop that sleeps the full interval and then
    /// samples takes `interval + sample_cost` per frame, so dividing the
    /// counter delta by the nominal interval overstates every rate.
    #[test]
    fn the_reported_interval_is_the_real_gap_between_samples_not_the_nominal_one() {
        let base = Instant::now();
        let mut p = Pacer::starting_at(Duration::from_millis(1000), base);
        // A sample that took 300 ms lands the next mark 1.3 s after the last
        // one only if the loop does not also sleep the full second...
        assert_eq!(p.remaining(base + Duration::from_millis(300)), Duration::from_millis(700));
        // ...and the interval reported is the distance actually travelled.
        assert_eq!(p.mark(base + Duration::from_millis(1000)), Duration::from_millis(1000));
    }

    /// A sampler slower than the refresh must report the gap it achieved. The
    /// alternative is a figure that says 200 ms when 900 ms passed, i.e. every
    /// rate four and a half times too high.
    #[test]
    fn a_sampler_slower_than_the_interval_reports_how_long_it_actually_took() {
        let base = Instant::now();
        let mut p = Pacer::starting_at(Duration::from_millis(200), base);
        assert_eq!(
            p.remaining(base + Duration::from_millis(900)),
            Duration::ZERO,
            "already overdue; there is nothing left to sleep"
        );
        assert_eq!(p.mark(base + Duration::from_millis(900)), Duration::from_millis(900));
        // ...and the next one is measured from there, not from the schedule.
        assert_eq!(p.mark(base + Duration::from_millis(1150)), Duration::from_millis(250));
    }

    /// The cadence is the configured interval, not the interval plus a sample.
    #[test]
    fn marking_before_each_sample_keeps_the_cadence_on_schedule() {
        let base = Instant::now();
        let mut p = Pacer::starting_at(Duration::from_millis(500), base);
        let mut at = base;
        for _ in 0..4 {
            // Sleep whatever is left, sample at that point, then burn 120 ms
            // "sampling" before the loop comes round again.
            at += p.remaining(at);
            assert_eq!(p.mark(at), Duration::from_millis(500));
            at += Duration::from_millis(120);
        }
        assert_eq!(at.saturating_duration_since(base), Duration::from_millis(2120));
    }

    #[test]
    fn wait_sleeps_about_the_interval_and_reports_about_that_much() {
        let mut p = Pacer::new(Duration::from_millis(80));
        let start = Instant::now();
        let dt = p.wait();
        assert!(dt >= Duration::from_millis(70), "{dt:?}");
        assert!(start.elapsed() >= Duration::from_millis(70));
        assert!(start.elapsed() < Duration::from_secs(2), "slept far too long");
    }

    #[test]
    fn a_synchronous_source_reports_no_frame_id() {
        // `None` means "every call is a fresh frame", which is what every
        // in-process source does.
        struct Sync;
        impl MetricSource for Sync {
            fn snapshot(&mut self, _dt: Duration) -> Snapshot {
                Snapshot::default()
            }
        }
        assert_eq!(Sync.frame_id(), None);
    }
}
