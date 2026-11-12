//! Monitoring several hosts at once.
//!
//! `--remote host` has always been one machine at a time, which is the shape
//! of a tool you reach for when you already know which machine is in trouble.
//! Finding that out is the other half of the job, and a terminal full of
//! `ssh` windows is how it is usually done.
//!
//! Nothing here needed inventing: a snapshot is the single thing the whole UI
//! renders from, `RemoteSource` already streams them over one SSH session, and
//! `ThreadedSource` already keeps a slow or unreachable source from blocking
//! the event loop. A fleet is a vector of those, one worker each, with a
//! selected index — so hosts are sampled concurrently and one unreachable
//! machine costs the others nothing.

use std::time::Duration;

use crate::metrics::{MetricSource, Snapshot};

/// One host's latest state, as the fleet table draws it.
#[derive(Debug, Clone)]
pub struct FleetHost {
    /// What the user wrote: `web-1`, `deploy@db-2`, `localhost`.
    pub target: String,
    /// Position in the order the hosts were named, which is what
    /// `MetricSource::select_host` takes. The table may be sorted otherwise.
    pub index: usize,
    /// The most recent frame, or a default one if none has arrived yet.
    pub snapshot: Snapshot,
    /// Why this host has no current numbers, when it has none.
    pub error: Option<String>,
    /// No frame has ever arrived: the first one is still in flight, or the
    /// host has been unreachable from the start. Distinguished from an error
    /// so a row can say "connecting" rather than drawing a machine at 0%.
    pub waiting: bool,
}

impl FleetHost {
    /// Whether this row has real numbers to draw.
    pub fn is_live(&self) -> bool {
        !self.waiting && self.error.is_none()
    }

    /// A short status word for the row, when it is not drawing numbers.
    ///
    /// An error outranks "connecting": a host that has not answered *and* has
    /// said why has told you the more useful of the two things.
    pub fn status(&self) -> &'static str {
        match (self.error.is_some(), self.waiting) {
            (true, _) => "unreachable",
            (_, true) => "connecting",
            _ => "ok",
        }
    }

    /// Whether any real frame has ever arrived. A host that has gone away
    /// still has its last known numbers, which are worth drawing — faded,
    /// next to the reason they stopped — where a host that never answered has
    /// nothing at all.
    pub fn has_data(&self) -> bool {
        !self.waiting
    }
}

struct Host {
    target: String,
    source: Box<dyn MetricSource>,
    latest: Snapshot,
    seen: bool,
}

/// Several hosts behind one `MetricSource`.
///
/// `snapshot` returns the selected host's latest frame, so every panel and
/// every key in the rest of the program goes on working unchanged against
/// whichever machine is in focus.
pub struct FleetSource {
    hosts: Vec<Host>,
    selected: usize,
    /// Bumped whenever any host produces a new frame, so `App` can tell a
    /// genuinely new sample from the same one handed out again.
    frames: u64,
    /// The selected host and what its own source last called its frame. The
    /// count used to move on every poll, so a host that had gone quiet was
    /// charted as flat measurement instead of showing the `sampling…` notice.
    last_frame: Option<(usize, Option<u64>)>,
}

impl FleetSource {
    /// `sources` is one already-built source per target, in the order the
    /// user named them.
    pub fn new(sources: Vec<(String, Box<dyn MetricSource>)>) -> FleetSource {
        FleetSource {
            hosts: sources
                .into_iter()
                .map(|(target, source)| Host {
                    target,
                    source,
                    latest: Snapshot::default(),
                    seen: false,
                })
                .collect(),
            selected: 0,
            frames: 0,
            last_frame: None,
        }
    }

    pub fn len(&self) -> usize {
        self.hosts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hosts.is_empty()
    }

    /// Poll every host. Returns the selected host's frame.
    ///
    /// Every host is polled on every tick, not just the selected one: the
    /// point of the fleet table is to see the machine that is in trouble
    /// *before* you have selected it, and a row that only updates while it is
    /// highlighted would show the fleet as it was whenever you last looked.
    fn poll(&mut self, dt: Duration) -> Snapshot {
        for host in &mut self.hosts {
            let snap = host.source.snapshot(dt);
            // A source with nothing yet hands back a default frame; taking it
            // would draw a machine idling at zero rather than one not yet
            // heard from.
            if snap.taken_at_unix > 0 || !host.snapshot_is_blank(&snap) {
                host.latest = snap;
                host.seen = true;
            }
        }
        // Switching hosts is new measurement too, even when the host being
        // switched to has been sitting on the same frame for a while. A host
        // whose source has no frame id of its own samples afresh on every
        // call by definition, so there is nothing to deduplicate there.
        let inner = self.hosts.get(self.selected).and_then(|h| h.source.frame_id());
        let current = (self.selected, inner);
        if inner.is_none() || self.last_frame != Some(current) {
            self.frames = self.frames.wrapping_add(1);
        }
        self.last_frame = Some(current);
        self.hosts.get(self.selected).map(|h| h.latest.clone()).unwrap_or_default()
    }
}

impl Host {
    /// Whether a frame carries nothing at all, which is what an unreachable
    /// host's source returns before it has ever succeeded.
    fn snapshot_is_blank(&self, snap: &Snapshot) -> bool {
        snap.taken_at_unix == 0 && snap.procs.is_empty() && snap.host.hostname.is_empty()
    }
}

impl MetricSource for FleetSource {
    fn snapshot(&mut self, dt: Duration) -> Snapshot {
        self.poll(dt)
    }

    fn fleet(&self) -> Vec<FleetHost> {
        self.hosts
            .iter()
            .enumerate()
            .map(|(index, h)| FleetHost {
                target: h.target.clone(),
                index,
                snapshot: h.latest.clone(),
                error: h.source.error(),
                waiting: !h.seen,
            })
            .collect()
    }

    fn select_host(&mut self, index: usize) {
        if index < self.hosts.len() {
            self.selected = index;
        }
    }

    fn selected_host(&self) -> usize {
        self.selected
    }

    fn label(&self) -> Option<String> {
        let host = self.hosts.get(self.selected)?;
        let reachable = self.hosts.iter().filter(|h| h.source.error().is_none()).count();
        Some(format!(
            "fleet {}/{} — {} ({} reachable)",
            self.selected + 1,
            self.hosts.len(),
            host.target,
            reachable
        ))
    }

    fn error(&self) -> Option<String> {
        self.hosts.get(self.selected).and_then(|h| h.source.error())
    }

    fn frame_id(&self) -> Option<u64> {
        Some(self.frames)
    }

    fn is_local(&self) -> bool {
        // A fleet is remote hosts by construction.
        false
    }
}

/// Split what the user wrote after `--remote` into targets.
///
/// Commas, because that is how every other tool spells a host list and because
/// an SSH target can contain almost anything else — `deploy@db-2.internal:2222`
/// has an `@` and a `:` in it already.
pub fn parse_targets(spec: &str) -> Vec<String> {
    spec.split(',').map(str::trim).filter(|t| !t.is_empty()).map(String::from).collect()
}

/// A named list of hosts, from `[[fleet]]`, so a fleet is `--remote @db`
/// rather than the same comma-separated list retyped every time.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct FleetGroup {
    pub name: String,
    pub hosts: Vec<String>,
}

/// `parse_targets`, with each `@name` replaced by that group's hosts.
///
/// `@` because it cannot start an SSH target — `user@host` has one, but not
/// first — so a group can never shadow a machine of the same name. A host
/// named twice, directly or through two groups, is watched once, where it
/// was first named: two workers on one machine would be two rows that
/// always agree, and two recorders truncating the same file.
pub fn expand_targets(spec: &str, groups: &[FleetGroup]) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    for item in parse_targets(spec) {
        let named = match item.strip_prefix('@') {
            Some(name) => {
                let group = groups.iter().find(|g| g.name == name).ok_or_else(|| {
                    let known: Vec<String> =
                        groups.iter().map(|g| format!("@{}", g.name)).collect();
                    match known.is_empty() {
                        true => format!("no [[fleet]] group named {name}"),
                        false => {
                            format!("no [[fleet]] group named {name}; known: {}", known.join(", "))
                        }
                    }
                })?;
                let hosts: Vec<String> = group
                    .hosts
                    .iter()
                    .map(|h| h.trim())
                    .filter(|h| !h.is_empty())
                    .map(String::from)
                    .collect();
                if hosts.is_empty() {
                    return Err(format!("[[fleet]] group {name} names no hosts"));
                }
                hosts
            }
            None => vec![item],
        };
        for host in named {
            if !out.contains(&host) {
                out.push(host);
            }
        }
    }
    Ok(out)
}

/// What the fleet table is sorted by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FleetSort {
    /// The order the hosts were named in.
    #[default]
    Named,
    Host,
    Cpu,
    Mem,
    Swap,
    Load,
    Disk,
    Net,
    Procs,
    /// Rules firing, with a host that is not answering counted worst of all:
    /// "which machine needs me" as one key press.
    Alerts,
}

impl FleetSort {
    pub const ALL: [FleetSort; 10] = [
        FleetSort::Named,
        FleetSort::Host,
        FleetSort::Cpu,
        FleetSort::Mem,
        FleetSort::Swap,
        FleetSort::Load,
        FleetSort::Disk,
        FleetSort::Net,
        FleetSort::Procs,
        FleetSort::Alerts,
    ];

    pub fn name(self) -> &'static str {
        match self {
            FleetSort::Named => "as named",
            FleetSort::Host => "host",
            FleetSort::Cpu => "cpu",
            FleetSort::Mem => "mem",
            FleetSort::Swap => "swap",
            FleetSort::Load => "load",
            FleetSort::Disk => "disk",
            FleetSort::Net => "net",
            FleetSort::Procs => "procs",
            FleetSort::Alerts => "alerts",
        }
    }

    pub fn next(self) -> FleetSort {
        let i = Self::ALL.iter().position(|k| *k == self).unwrap_or(0);
        Self::ALL[(i + 1) % Self::ALL.len()]
    }

    pub fn prev(self) -> FleetSort {
        let i = Self::ALL.iter().position(|k| *k == self).unwrap_or(0);
        Self::ALL[(i + Self::ALL.len() - 1) % Self::ALL.len()]
    }

    /// Whether a key reads best biggest-first, which is every figure: the
    /// busiest host is the one being looked for. Names read A to Z.
    pub fn default_desc(self) -> bool {
        !matches!(self, FleetSort::Named | FleetSort::Host)
    }
}

/// Order `hosts` by `key`. `alerts(host)` is how many rules are firing on
/// it, which only the caller knows.
///
/// A host with no numbers sorts after every host that has some, whichever
/// way the figures are sorted — a blank row at the top of "busiest CPU"
/// would read as the busiest — except under `Alerts`, where a host that is
/// not answering is the worst of all and goes first. The sort is stable, so
/// ties keep the order the hosts were named in.
pub fn sort_hosts(
    hosts: &mut [FleetHost],
    key: FleetSort,
    desc: bool,
    show_virtual: bool,
    alerts: impl Fn(&FleetHost) -> usize,
) {
    use std::cmp::Ordering;
    let figure = |h: &FleetHost| -> Option<f64> {
        let s = &h.snapshot;
        if !h.has_data() {
            return None;
        }
        Some(match key {
            FleetSort::Cpu => s.cpu.avg(),
            FleetSort::Mem => s.mem.ratio(),
            FleetSort::Swap => s.mem.swap_ratio(),
            FleetSort::Load => s.host.load[0],
            FleetSort::Disk => s.disk_util_max().unwrap_or(0.0),
            FleetSort::Net => {
                let (rx, tx) = s.net_totals(show_virtual);
                rx + tx
            }
            FleetSort::Procs => s.procs.len() as f64,
            _ => 0.0,
        })
    };
    let dir = |o: Ordering| if desc { o.reverse() } else { o };
    match key {
        FleetSort::Named => {
            hosts.sort_by_key(|h| h.index);
            if desc {
                hosts.reverse();
            }
        }
        FleetSort::Host => hosts.sort_by(|a, b| dir(a.target.cmp(&b.target))),
        FleetSort::Alerts => {
            // Unreachable first, then most rules firing; the direction only
            // reverses the whole thing.
            let rank = |h: &FleetHost| (!h.is_live() as usize, alerts(h));
            hosts.sort_by(|a, b| dir(rank(a).cmp(&rank(b))));
        }
        _ => hosts.sort_by(|a, b| match (figure(a), figure(b)) {
            (Some(x), Some(y)) => dir(x.partial_cmp(&y).unwrap_or(Ordering::Equal)),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{CpuSample, HostInfo, ProcRow};

    /// A source that hands back a fixed frame, optionally with an error.
    struct Fake {
        host: &'static str,
        cpu: f32,
        error: Option<String>,
        blank: bool,
        calls: std::cell::Cell<usize>,
    }

    impl Fake {
        fn new(host: &'static str, cpu: f32) -> Box<Fake> {
            Box::new(Fake { host, cpu, error: None, blank: false, calls: Default::default() })
        }
        fn broken(host: &'static str) -> Box<Fake> {
            Box::new(Fake {
                host,
                cpu: 0.0,
                error: Some(format!("{host}: No route to host")),
                blank: true,
                calls: Default::default(),
            })
        }
    }

    impl MetricSource for Fake {
        fn snapshot(&mut self, _dt: Duration) -> Snapshot {
            self.calls.set(self.calls.get() + 1);
            if self.blank {
                return Snapshot::default();
            }
            Snapshot {
                host: HostInfo { hostname: self.host.into(), ..Default::default() },
                cpu: CpuSample { per_core: vec![self.cpu], freq_mhz: vec![] },
                procs: vec![ProcRow { pid: 1, name: "init".into(), ..Default::default() }],
                taken_at_unix: 1_700_000_000,
                ..Default::default()
            }
        }
        fn error(&self) -> Option<String> {
            self.error.clone()
        }
    }

    fn fleet_of(hosts: Vec<(&str, Box<Fake>)>) -> FleetSource {
        FleetSource::new(
            hosts.into_iter().map(|(t, s)| (t.to_string(), s as Box<dyn MetricSource>)).collect(),
        )
    }

    #[test]
    fn targets_are_split_on_commas_and_trimmed() {
        assert_eq!(parse_targets("a,b,c"), vec!["a", "b", "c"]);
        assert_eq!(
            parse_targets(" web-1 , deploy@db-2.internal "),
            vec!["web-1", "deploy@db-2.internal"]
        );
        // One host is still one host, which is what keeps `--remote box`
        // behaving exactly as it always has.
        assert_eq!(parse_targets("box"), vec!["box"]);
        assert_eq!(parse_targets(",,"), Vec::<String>::new());
        assert_eq!(parse_targets(""), Vec::<String>::new());
    }

    /// The point of the table is to show the machine that is in trouble before
    /// you have selected it. A row that only updated while highlighted would
    /// show the fleet as it was whenever you last looked at that host.
    #[test]
    fn every_host_is_sampled_every_tick_not_just_the_selected_one() {
        let mut f = fleet_of(vec![("a", Fake::new("a", 10.0)), ("b", Fake::new("b", 90.0))]);
        f.snapshot(Duration::ZERO);
        f.snapshot(Duration::ZERO);

        let rows = f.fleet();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].snapshot.cpu.avg(), 10.0);
        assert_eq!(rows[1].snapshot.cpu.avg(), 90.0, "the unselected host is live too");
        assert!(rows.iter().all(|r| r.is_live()));
    }

    #[test]
    fn the_selected_host_is_the_one_the_rest_of_the_program_sees() {
        let mut f = fleet_of(vec![("a", Fake::new("a", 10.0)), ("b", Fake::new("b", 90.0))]);
        assert_eq!(f.snapshot(Duration::ZERO).host.hostname, "a");
        assert_eq!(f.selected_host(), 0);

        f.select_host(1);
        assert_eq!(f.selected_host(), 1);
        assert_eq!(f.snapshot(Duration::ZERO).host.hostname, "b");
    }

    #[test]
    fn selecting_a_host_that_is_not_there_leaves_the_selection_alone() {
        let mut f = fleet_of(vec![("a", Fake::new("a", 10.0))]);
        f.select_host(99);
        assert_eq!(f.selected_host(), 0, "an out-of-range index must not blank the view");
        assert_eq!(f.snapshot(Duration::ZERO).host.hostname, "a");
    }

    /// One unreachable machine must cost the others nothing — neither their
    /// numbers nor the interface.
    #[test]
    fn an_unreachable_host_is_one_bad_row_rather_than_a_blank_fleet() {
        let mut f = fleet_of(vec![("up", Fake::new("up", 42.0)), ("down", Fake::broken("down"))]);
        f.snapshot(Duration::ZERO);

        let rows = f.fleet();
        assert!(rows[0].is_live());
        assert_eq!(rows[0].status(), "ok");
        assert!(!rows[1].is_live());
        assert_eq!(rows[1].status(), "unreachable");
        assert!(rows[1].error.as_deref().unwrap().contains("No route"));
        // ...and the good host still has its numbers.
        assert_eq!(rows[0].snapshot.cpu.avg(), 42.0);
    }

    /// A host that has not answered yet is not a host idling at zero. Drawing
    /// one as the other is how a fleet view says everything is fine while it
    /// is still connecting.
    #[test]
    fn a_host_that_has_not_answered_yet_says_so_rather_than_reading_as_idle() {
        let mut slow = Fake::new("slow", 0.0);
        slow.blank = true;
        let mut f = fleet_of(vec![("slow", slow)]);

        let rows = f.fleet();
        assert!(rows[0].waiting);
        assert!(!rows[0].has_data());
        assert_eq!(rows[0].status(), "connecting", "no error yet, just no answer");

        f.snapshot(Duration::ZERO);
        // Still nothing real has arrived, so still not "idle at 0%".
        assert!(f.fleet()[0].waiting);
    }

    /// A host that *was* answering and stopped keeps its last known numbers,
    /// which are worth drawing next to the reason they stopped. That is a
    /// different row from one that never answered at all.
    #[test]
    fn a_host_that_goes_away_keeps_the_last_thing_it_said() {
        struct Flaky {
            calls: usize,
        }
        impl MetricSource for Flaky {
            fn snapshot(&mut self, _dt: Duration) -> Snapshot {
                self.calls += 1;
                if self.calls > 1 {
                    return Snapshot::default();
                }
                Snapshot {
                    host: HostInfo { hostname: "db-1".into(), ..Default::default() },
                    cpu: CpuSample { per_core: vec![77.0], freq_mhz: vec![] },
                    taken_at_unix: 1_700_000_000,
                    ..Default::default()
                }
            }
            fn error(&self) -> Option<String> {
                (self.calls > 1).then(|| "db-1: connection reset".to_string())
            }
        }

        let mut f = FleetSource::new(vec![(
            "db-1".to_string(),
            Box::new(Flaky { calls: 0 }) as Box<dyn MetricSource>,
        )]);
        f.snapshot(Duration::ZERO);
        assert!(f.fleet()[0].is_live());

        f.snapshot(Duration::ZERO);
        let row = &f.fleet()[0];
        assert!(!row.is_live());
        assert!(row.has_data(), "it answered once; that frame is still worth showing");
        assert_eq!(row.status(), "unreachable");
        assert_eq!(row.snapshot.cpu.avg(), 77.0, "the last good frame is kept");
    }

    #[test]
    fn the_label_says_where_you_are_and_how_much_of_the_fleet_is_answering() {
        let mut f = fleet_of(vec![
            ("web-1", Fake::new("web-1", 1.0)),
            ("web-2", Fake::new("web-2", 2.0)),
            ("db-1", Fake::broken("db-1")),
        ]);
        f.snapshot(Duration::ZERO);
        let label = f.label().unwrap();
        assert!(label.contains("web-1"), "{label}");
        assert!(label.contains("1/3"), "{label}");
        assert!(label.contains("2 reachable"), "{label}");

        f.select_host(2);
        let label = f.label().unwrap();
        assert!(label.contains("db-1"), "{label}");
        assert!(label.contains("3/3"), "{label}");
    }

    #[test]
    fn the_error_reported_is_the_selected_host_s() {
        let mut f = fleet_of(vec![("up", Fake::new("up", 1.0)), ("down", Fake::broken("down"))]);
        f.snapshot(Duration::ZERO);
        assert_eq!(f.error(), None);
        f.select_host(1);
        assert!(f.error().unwrap().contains("No route"));
    }

    #[test]
    fn a_frame_id_moves_so_the_charts_know_a_tick_happened() {
        let mut f = fleet_of(vec![("a", Fake::new("a", 1.0))]);
        let before = f.frame_id();
        f.snapshot(Duration::ZERO);
        assert_ne!(f.frame_id(), before);
    }

    #[test]
    fn a_fleet_of_nothing_is_empty_rather_than_a_panic() {
        let mut f = FleetSource::new(Vec::new());
        assert!(f.is_empty());
        assert_eq!(f.snapshot(Duration::ZERO).host.hostname, "");
        assert!(f.fleet().is_empty());
        assert_eq!(f.label(), None);
    }

    fn groups() -> Vec<FleetGroup> {
        vec![
            FleetGroup { name: "db".into(), hosts: vec!["db-1".into(), "db-2".into()] },
            FleetGroup { name: "web".into(), hosts: vec!["web-1".into(), "db-1".into()] },
            FleetGroup { name: "empty".into(), hosts: vec![" ".into()] },
        ]
    }

    #[test]
    fn a_group_expands_in_place_and_a_host_is_watched_once() {
        assert_eq!(
            expand_targets("@db,cache,@web", &groups()).unwrap(),
            ["db-1", "db-2", "cache", "web-1"]
        );
        assert_eq!(expand_targets("deploy@db-1", &groups()).unwrap(), ["deploy@db-1"]);
    }

    #[test]
    fn an_unknown_or_empty_group_is_refused_by_name() {
        let err = expand_targets("@nope", &groups()).unwrap_err();
        assert!(err.contains("nope") && err.contains("@db"), "{err}");
        assert!(expand_targets("@empty", &groups()).unwrap_err().contains("names no hosts"));
        assert!(expand_targets("@x", &[]).unwrap_err().contains("no [[fleet]] group"));
    }

    fn row(target: &str, index: usize, cpu: f32, live: bool) -> FleetHost {
        FleetHost {
            target: target.into(),
            index,
            snapshot: Snapshot {
                cpu: CpuSample { per_core: vec![cpu], freq_mhz: vec![] },
                ..Default::default()
            },
            error: (!live).then(|| "down".to_string()),
            waiting: false,
        }
    }

    fn order(hosts: &[FleetHost]) -> Vec<&str> {
        hosts.iter().map(|h| h.target.as_str()).collect()
    }

    #[test]
    fn hosts_sort_by_a_figure_with_blank_rows_last_either_way() {
        let mut never = row("never", 3, 0.0, true);
        never.waiting = true;
        let mut hosts =
            vec![row("a", 0, 10.0, true), row("b", 1, 90.0, true), row("c", 2, 50.0, true), never];
        sort_hosts(&mut hosts, FleetSort::Cpu, true, false, |_| 0);
        assert_eq!(order(&hosts), ["b", "c", "a", "never"]);
        sort_hosts(&mut hosts, FleetSort::Cpu, false, false, |_| 0);
        assert_eq!(order(&hosts), ["a", "c", "b", "never"]);
        sort_hosts(&mut hosts, FleetSort::Named, false, false, |_| 0);
        assert_eq!(order(&hosts), ["a", "b", "c", "never"]);
    }

    #[test]
    fn the_alerts_key_puts_unreachable_hosts_then_the_most_alerts_first() {
        let mut hosts = vec![
            row("quiet", 0, 1.0, true),
            row("two", 1, 1.0, true),
            row("down", 2, 1.0, false),
            row("one", 3, 1.0, true),
        ];
        let firing = |h: &FleetHost| match h.target.as_str() {
            "two" => 2,
            "one" => 1,
            _ => 0,
        };
        sort_hosts(&mut hosts, FleetSort::Alerts, true, false, firing);
        assert_eq!(order(&hosts), ["down", "two", "one", "quiet"]);
    }

    #[test]
    fn sort_keys_cycle_both_ways() {
        let mut k = FleetSort::Named;
        for _ in 0..FleetSort::ALL.len() {
            assert_eq!(k.next().prev(), k);
            k = k.next();
        }
        assert_eq!(k, FleetSort::Named);
    }
}
