# crabmon — Design

How crabmon is put together, and why.

## Goals

- A compact terminal monitor that is pleasant to read at a glance and honest
  about what it measures.
- Pure polling via `sysinfo` plus a little direct `/proc` and `/sys` reading —
  no daemons, no privileged helpers.
- Cross-platform compile (Linux/macOS/Windows), with Unix- and Linux-specific
  features cleanly feature-gated.
- Every non-trivial decision testable without a terminal or a live host.

## Repository layout

```
README.md            what crabmon does and how to drive it
Makefile             a front end to scripts/build.sh, one line per target
docs/
  crabmon.1          the man page
  DESIGN.md          this document
  CHANGELOG.md       the release history
  crabmon.png        the mascot the README leads with
completions/         bash, zsh and fish
scripts/build.sh     the build driver; build-deb.sh is an alias CI still uses
scripts/embed-snapshot.py  puts a captured frame into README.md
scripts/e2e-ssh.py   opt-in checks against a real SSH host
src/                 the library, plus a thin binary
tests/               eight integration suites, including local and container E2E
```

The prose lives under `docs/` with the man page it duplicates, so the three
documents that have to agree with each other sit in one directory; `tests/docs.rs`
is what enforces that they do.

## Crate layout

The application is a **library** with a thin binary on top. That split exists
for one reason: it makes the whole program testable headlessly.

```
src/
  main.rs            terminal setup/teardown, signals, panic hook, event loop
  lib.rs             module tree and the shared refresh bounds
  cli.rs             argument parsing (hand-rolled, ~700 lines)
  config.rs          the on-disk config, with defaults and clamping
  theme.rs           colour themes and threshold→colour mapping
  format.rs          byte/rate/duration formatting
  filter.rs          the process filter language
  sort.rs            sort keys and comparators
  tree.rs            process-tree flattening
  history.rs         bounded time-series buffers
  alerts.rs          threshold rules and their state machine
  export.rs          JSON/CSV snapshot rendering
  app.rs             all state and input handling
  record.rs          JSONL recording, rotation, the replay source and
                     `Recordings`, which every mode records through
  remote.rs          monitoring another host over SSH
  sampler.rs         sampling on a worker thread
  diff.rs            comparing two snapshots
  doctor.rs          what this host lets crabmon measure, for `--doctor`
  events.rs          the `E` timeline, derived from consecutive snapshots
  watch.rs           the headless wait-for-a-condition mode
  supervisor.rs      alert rules and the flight recorder, outside the TUI
  fleet.rs           several hosts behind one metric source, host groups, sorting
  notify.rs          `[[notify]]`: alert transitions posted via curl
  silence.rs         `[[silence]]` windows and `S` silences, in UTC
  serve.rs           the Prometheus exporter
  incident.rs        the readable summary written beside a flight dump
  audit.rs           the local log of actions taken
  clipboard.rs       OSC 52 copy
  metrics/
    mod.rs           the data model and the `MetricSource` trait
    sysinfo_source.rs the live host implementation
    diskstats.rs     /proc/diskstats: bytes, saturation and service times
    netclass.rs      virtual-interface classification
    gpu.rs           /sys/class/drm and nvidia-smi
    cgroup.rs        cgroup v1/v2 limits
    psi.rs           /proc/pressure stall information
    kstat.rs         CPU time by mode from /proc/stat, paging from /proc/vmstat
    netstat.rs       TCP health from /proc/net/snmp, netstat and sockstat
    containers.rs    container names from the runtimes' own files
    storage.rs       md RAID, ZFS pool and btrfs device health
    smart.rs         opt-in SMART disk health via `smartctl --json`
    units.rs         systemd failures and restart loops via `systemctl show`,
                     journal tails via `journalctl`, restart and reset
    power.rs         batteries, mains and RAPL
    smaps.rs         proportional and unique memory per process
    kmsg.rs          OOM kills, crashes and IO errors from /dev/kmsg
    ioprio.rs        IO scheduling class and level
    procdetail.rs    OOM score, cgroup path and environment, per process
    meminfo.rs       the buffers/cache breakdown
    procgroup.rs     per-process cgroup paths, and what they mean
    sockets.rs       /proc/net joined against a process's fd table
    fds.rs           open descriptors, their limit, and what they point at
    identity.rs      what a process is *now*, after an exec renamed it
    control.rs       live identity validation and Linux pidfd signal delivery
    netns.rs         per-network-namespace traffic, i.e. per container
  ui/
    mod.rs           layout selection and the draw entry point
    header.rs        host banner and status line
    cpu.rs           average chart, per-core strip, per-core detail
    mem.rs           memory, swap and cgroup gauges
    procs.rs         the process table and the grouped view
    psi.rs           pressure gauges
    kernel.rs        what the kernel has complained about
    power.rs         battery and package power
    net.rs           interface table and RX/TX chart
    disks.rs         per-mount gauges
    sensors.rs       grouped temperatures
    gpu.rs           GPU rows
    fleet.rs         one row per host, when several are being watched
    popups.rs        signal, confirm, renice, affinity, ionice, seek, unit,
                     silence, detail, help, alerts, audit log, events
```

## The metric boundary

Everything the UI draws comes from a single `Snapshot` value, produced by a
`MetricSource`:

```rust
pub trait MetricSource {
    fn snapshot(&mut self, dt: Duration) -> Snapshot;
    fn timeline(&self) -> Option<(usize, usize)> { None }
    fn seek(&mut self, _position: usize) {}
    fn peek(&self) -> Option<Snapshot> { None }
    fn label(&self) -> Option<String> { None }
    fn error(&self) -> Option<String> { None }
    fn fleet(&self) -> Vec<FleetHost> { Vec::new() }
    fn select_host(&mut self, _index: usize) {}
    fn selected_host(&self) -> usize { 0 }
    fn is_local(&self) -> bool { true }
    fn frame_id(&self) -> Option<u64> { None }
}
```

`SysinfoSource` implements it against the real host, `ReplaySource` against a
recording, `RemoteSource` against another machine over SSH, and the test-suite
against canned frames. That last one is what lets `tests/render.rs` assert on a
real rendered screen and `tests/app_behaviour.rs` drive the full interaction
model with no terminal, no `/proc`, and no timing flakiness.

The defaulted methods exist for sources that are not a plain synchronous poll.
`timeline` and `seek` make the scrub keys work; `peek` reads the current frame
*without* advancing, which matters because calling `snapshot` after a `seek`
would step straight past the frame just sought. `frame_id` belongs to
asynchronous sources: `None` means every call produces a genuinely new sample,
and anything else lets `App` tell a fresh frame from the same one handed out
again. Live sources inherit the defaults and the scrub keys are inert for them.

`error`, `fleet`, `select_host` and `selected_host` are the fleet view's half:
`label` already says why a source cannot sample, but as prose with the answer
embedded in it, and a row has to *know* whether to draw numbers or an
explanation rather than match on a word inside a sentence. `is_local` is the
one with teeth — everything crabmon does beyond drawing acts on a PID, and
under `--remote` or `--replay` a local PID that happens to match is an
unrelated program, so signalling, renicing, affinity, ionice and the detail
pane's `/proc` reads are all gated on it.

`Snapshot` is plain data (`HostInfo`, `CpuSample`, `MemSample`, `ProcRow`,
`NetIface`, `DiskRow`, `Sensor`, `GpuInfo`, `CgroupInfo`, and the newer
`PsiSample`, `PowerSample`, `NetNamespace`, `CgroupStat`, `KernelEvent`,
`CpuModes`, `VmActivity`, `TcpHealth`, `StorageArray` and `UnitHealth`, with
`kernel_unreadable`, `units_unreadable` and `taken_at_unix`) and derives
`Serialize`, so JSON export is the same type the UI reads rather than a parallel
representation that can drift.

## State: `App`

`App` holds the config, the theme, the current `Snapshot`, the derived process
view, the history buffers, the alert engine and the interaction mode. It owns no
terminal. Its IO is limited to what a key asks for: the metric source, the
config file, exports, the audit log, OSC 52 on `y`, the replay timeline scan
thread, the detail pane's `/proc` reads, signals through a pidfd, and
`systemctl`/`journalctl` for the unit popup — the last two injectable
(`App::unit_control`, `App::unit_journal`) so tests run none of them.

Two details matter:

- **Selection is a PID, not a row index.** `rebuild_view` re-finds the selected
  PID after every refresh, sort change or filter change, so re-sorting cannot
  silently move the highlight onto a different process — which also means the
  signal menu cannot target the wrong one.
- **The row list is private.** `rows()` is read-only and can only be replaced by
  `rebuild_view`, so the view can never disagree with the scroll offset.

### Interaction modes

| Mode | Entered by | Popup |
| --- | --- | --- |
| `Normal` | default | — |
| `Filter` | `/` | inline, in the process panel title |
| `SignalMenu` | `t` | signal list |
| `ConfirmKill` | `Enter` in the signal menu | confirmation |
| `Renice` | `r` | numeric prompt |
| `Affinity` | `A` | CPU-list prompt |
| `IoPriority` | `I` | IO class and level prompt |
| `Detail` | `Enter` on a process row (in the fleet it selects a host, when grouped it drills in) | process detail |
| `Help` | `?` | key bindings |
| `Alerts` | `!` | rules, transitions and delivery notes |
| `AuditLog` | `L` | the local record of actions taken |
| `Seek` | `:` in a replay | frame or time prompt |
| `Unit` | `u` | unit state, journal, restart/reset with `y`/`n` |
| `Silence` | `S`, or `S` in the `!` popup | duration and rule prompt |
| `Events` | `E` | the events timeline, newest first |

Sending a signal is two-step by construction: `SignalMenu` → `ConfirmKill`.
Every action that targets a process — signal, renice, affinity and IO priority
alike — re-checks that the PID still has the start time it had when the prompt
opened,
so a recycled PID is refused rather than acted on. Linux samples also carry
the kernel start tick, distinguishing reuse within one Unix second. Controls
validate live identity immediately before the OS call; signals use a pidfd
so delivery remains tied to that incarnation. The other Linux controls still
use PID-based syscalls, with a live recheck immediately before each call.

That check only works because the target set is **frozen when the prompt opens**,
in `pending_targets`. Rebuilding it at confirm time from the current rows would
silently defeat the guard: the recycled process sits at the same PID and looks
like a perfectly valid target. Bulk actions made this a real hazard rather than
a theoretical one, since a tagged set can outlive several refreshes.

Input handling returns an `Action` (`None`, `Quit`, `Export`) instead of acting
directly, which keeps the side effects in the event loop and the decisions in
testable code.

## The refresh cycle

`App::tick()` runs at most once per interval (default 800 ms, adjustable
200–10000 ms). The floor is not arbitrary: `sysinfo::MINIMUM_CPU_UPDATE_INTERVAL`
is 200 ms, and below it CPU percentages are noise. A unit test asserts our floor
never drops below the library's.

Each cycle takes a `Snapshot`, pushes into the history buffers, evaluates the
alert rules, and rebuilds the process view. Rendering is decoupled: the UI
redraws on every loop iteration so input feels instant, while sampling happens
on the interval.

Rates are derived from deltas of monotonic counters. `metrics::rate` returns 0
for a counter that went backwards, so a device or interface that resets its
counters produces a gap rather than an absurd spike.

## Data sources & quirks

- **Memory** — `sysinfo` reports **bytes**, not KiB. Treating those bytes
  as KiB is what made a 16 GB machine display "15292.6 GiB"; `format::human_bytes`
  now does the scaling and a test pins the exact expected string.
- **Device lists** — `Networks::refresh(false)` and friends only update entries
  already in the list. `refresh(true)` is what discovers hot-plugged disks, new
  containers and fresh VPN interfaces, so the source re-lists every 5 seconds.
  (Before sysinfo 0.33 these were `refresh()` and `refresh_list()`.)
- **Task lists** — `Process::tasks()` is the set of *sub*tasks and excludes the
  main thread, so the thread count is `len() + 1`. sysinfo 0.30 included the
  main task; 0.33 onwards does not, and refreshing the list is opt-in
  (`ProcessRefreshKind::with_tasks`). A live cross-check against
  `/proc/<pid>/task` is what caught both.
- **CPU modes** — `sysinfo` reports one busy figure per core, so the mode
  breakdown reads the aggregate `cpu` line of `/proc/stat` itself. `guest` and
  `guest_nice` are left out of the total because the kernel already counts
  them inside `user` and `nice`. A counter that goes backwards yields no
  reading rather than a huge one, and a failed read keeps the previous
  counters, so one bad read costs one sample rather than producing a rate
  since boot. `--remote` hosts report it only once their own crabmon is new
  enough to send it, and `None` is never drawn or exported as zero.
- **Paging** — `pswpin`/`pswpout` in `/proc/vmstat` count *pages*, which are
  64 KiB on some arm64 and ppc64 kernels, so the byte rate multiplies by
  `sysconf(_SC_PAGESIZE)` rather than by 4096. `oom_kill` exists from 4.13 and
  is `None` before then. It is the one OOM signal an unprivileged crabmon
  always has, since `/dev/kmsg` usually needs root.
- **Cgroup throttling and OOM kills** — `cpu.stat` already carried
  `nr_periods` and `nr_throttled` beside the `usage_usec` it was read for, so
  throttling is the change in one over the change in the other; no period
  elapsed (idle, or no quota) is `None`, not 0%. `memory.events` is
  hierarchical, so a unit's `oom_kill` includes its children's. `oomkill`
  rules count with `Series::increase` rather than `growth`, because a kill
  in the first minute is an event, not a slope too short to trust, and a
  recreated cgroup whose counter restarts has risen by nothing. Each group
  costs four more small reads (`memory.events` and three `*.pressure`),
  still bounded by `MAX_CGROUPS`.
- **Container names** — read from runtime files first: Docker's
  `config.v2.json`, containers/storage's `containers.json`, and the kubelet's
  `/var/log/containers/<pod>_<ns>_<container>-<id>.log` links, split on the
  *last* dash because pod names are full of them. `/var/lib/docker` is
  root-only, so a `docker` group member — the usual case — gets names from
  one `GET /containers/json` on the Engine API socket instead: HTTP/1.0 so
  the body is never chunked, a 500 ms timeout because it runs on the
  sampling thread, a 1 MiB cap on the response, and a socket that refuses
  the connection is simply skipped. A directory walk happens
  only when an id turns up that nothing names, at most every ten seconds, and
  known ids are never re-read. The name is display and filter only; the id
  stays the key, so a rename or a name reused by a new container cannot merge
  two histories.
- **TCP health** — `/proc/net/snmp` and `/proc/net/netstat` are read by
  column *name*, since their column sets differ between kernels; `MaxConn`
  is `-1` and is skipped rather than breaking the line. Retransmits are a
  share of segments sent in the interval, `None` when nothing was sent.
  CLOSE_WAIT is counted from `/proc/net/tcp{,6}` only while `sockstat` says
  there are under `MAX_SOCKETS_SCANNED` sockets; above that it is `None`
  rather than megabytes of reading per sample. All of it describes the
  namespace crabmon runs in.
- **Storage health** — `/proc/mdstat` is parsed per array: `[n/m] [U_]` and
  `(F)` members say degraded, `recovery`/`reshape` say rebuilding (still no
  redundancy), and `resync`/`check`/`repair` of a complete array are upkeep,
  shown as progress and never as trouble — otherwise every monthly scrub
  would page someone. ZFS pools are the directories under
  `/proc/spl/kstat/zfs` that carry a `state` file; the rest is module-wide
  statistics. btrfs `error_stats` (5.14+) is persistent across mounts, so a
  non-zero count is history until reset with `btrfs device stats -z`; it is
  reported as degraded because the disk that produced it is still there.
- **Major faults** — the `/proc/<pid>/stat` read that already confirms a
  process's identity also carries `majflt`, so the MAJFLT column costs no
  syscall. Rates are keyed by PID *and* start time, and threads are skipped
  because their counters are the process's.
- **Disk IO** — `/proc/diskstats` is keyed by kernel device name, but a
  filesystem's device may be `/dev/mapper/root_crypt`. `diskstats::resolve_device_key`
  canonicalises the symlink (`→ /dev/dm-0`) and takes the basename, which is why
  IO rates work on LVM and LUKS roots. Sectors are 512 bytes regardless of the
  device's logical block size.
- **Disk saturation** — bytes alone cannot say a device is *busy*: an NVMe
  serving 4K random reads is pinned at a few MB/s, and a throughput gauge draws
  that as idle. `io_ticks` (field 13) is milliseconds with a non-empty queue,
  so `Δio_ticks / Δwall` is a utilisation fraction; the service-time fields
  divided by completed requests give mean latency. Both arrive on the line the
  parser was already reading and discarding. Utilisation is clamped, because
  the kernel samples the queue on its own timer and a busy device routinely
  reports a few milliseconds more than the interval held. `await` is `None`
  rather than 0 when nothing completed: there is no average of nothing, and
  zero would draw an instant disk.
- **Disk aggregates** — summed per *device*, not per mount. A device carrying
  several mounts — the ordinary btrfs subvolume layout, and every bind mount —
  publishes one set of counters that each of its mounts reports, so the old
  per-mount sum multiplied the machine's disk throughput by however many
  subvolumes happened to be mounted. Utilisation aggregates as a maximum rather
  than a sum: two disks at 100% and 0% are not a machine at 50%, and the pinned
  one is still what everything is waiting on.
- **Network aggregate** — loopback, bridges, veth pairs and VPN tunnels are
  classified as virtual and excluded by default. Counting them meant VPN traffic
  was reported roughly twice, once on the tunnel and once on the physical NIC.
- **Sensors** — grouped by driver prefix (`coretemp Core 8` → `coretemp`) with
  the group's hottest reading. A flat hottest-first list let 22 core sensors
  push the NVMe and chassis readings off a six-line panel.
- **Threads** — on Linux the sampler reports every *task*, so a stock process
  list is several times longer than the number of processes (2133 rows for 477
  processes on the development machine). A real process reports a task count and
  its threads do not, so threads are filtered out by default and `H` shows them.
  The filter is only applied when that distinction actually exists, since
  platforms that never report a task count would otherwise have every row
  hidden.
- **Pressure** — `/proc/pressure/*`. `some` is the share of wall time with at
  least one task stalled, `full` the share with none running at all. The
  thresholds are absolute rather than proportional: 10% of wall time stalled is
  already bad, whatever the machine.
- **Power** — `/sys/class/power_supply` for batteries and mains, with charge
  (µAh) converted through the reported voltage for devices that do not report
  energy (µWh) directly. Package watts come from RAPL energy deltas, whose
  counters are root-only on most kernels since CVE-2020-8694, so the figure is
  omitted rather than guessed.
- **Grouping** — `/proc/<pid>/cgroup` per process would be thousands of extra
  file reads a second, so paths are cached per PID *and start time*, read once
  for each process that is new since the last sample, and evicted when it exits.
  The same trick applies to `getpriority` for the nice column, refreshed on the
  five-second relist cadence rather than every tick; together they cost about a
  tenth of a percent of a core.
- **Sockets** — `/proc/net/{tcp,tcp6,udp,udp6}` keyed by inode, joined against
  the `socket:[inode]` symlinks in a process's fd table. Only ever done for the
  one process whose detail pane is open.
- **File descriptors** — a `getdents` sweep of `/proc/<pid>/fd`, counted rather
  than collected so a process holding 200k sockets does not materialise 200k
  `PathBuf`s, and capped at `MAX_FDS_COUNTED`. The *limit* comes from
  `/proc/<pid>/limits` and is cached per PID and start time, like the cgroup
  path: it is set at exec and almost never changes. The count alone says
  nothing — 4000 descriptors is unremarkable against a limit of a million and
  fatal against the default 1024 — so the ratio is what the column, the filter
  and the alert are all written against. Unlike the `ports` column this is
  sampled for every process rather than the visible rows, because a filter over
  only what is already on screen is a filter over the wrong set. An unreadable
  fd table reads as `None`, not zero: another user's process is not a process
  holding no files, and zero would sweep every one of them into `fd<100`.
- **Per-process IO** — differences of `Process::disk_usage()` totals. Reading
  `/proc/<pid>/io` for another user's process is not permitted, so those rates
  legitimately read as zero.
- **GPU** — amdgpu exposes `gpu_busy_percent` and VRAM counters; i915/xe expose
  clocks but no busy counter, so the clock ratio is shown prefixed with `~` to
  make clear it is not utilisation. NVIDIA needs `nvidia-smi`, which is opt-in
  because it costs tens of milliseconds per refresh. It runs on its own
  thread (`gpu::NvidiaSmi`): each sample queues one query and takes the last
  answer, so the figures are a sample behind, and an answer older than
  `NvidiaSmi::STALE` is dropped rather than drawn as current.
- **systemd units** — `systemctl show -p Id,ActiveState,SubState,Result,
  NRestarts,ExecMainStatus '*.service'`, every five seconds on its own thread
  (`units::UnitWatcher`); about 100 ms for a hundred units, which is why it is
  not in front of the sample. `NRestarts` is cumulative, so the interval
  loses no restarts. `RestartTracker` keeps each unit's count over the
  window to turn "restarted 40 times since March" into "restarted 7 times in
  ten minutes"; a count that drops is a manual restart and starts the
  history again. A snapshot carries only units that are not healthy or have
  restarted at all — the rest would be in every frame of every recording —
  and `units_unreadable` says why there are none where systemd could not be
  read, or was not asked. The five worst units in trouble also carry their
  last journal lines (`journalctl --unit=… -n 12 -o short-iso`), read on the
  same thread and cached by the unit's restart count, state, result and exit
  status (`units::LogCache`) — each read cost a second on a busy one-core
  host, and the fresh states waited behind every one — so a pass publishes
  the new states first and the changed journals after. `UnitWatcher::latest`
  waits up to two seconds from start for a complete first reading, settles
  for the states without journals after that, and with nothing at all
  reports `units::STILL_READING` rather than an empty, healthy-looking list. `units::refresh_now`
  wakes the watcher for an immediate reading after `u` restarts or resets a
  unit: the evidence travels with the snapshot, so `--remote`,
  recordings, notifications (`HookCall::detail`) and incident summaries get
  it without a second connection. `u` restarts and resets through
  `units::control` — `systemctl --no-ask-password`, `--no-block` for a
  restart, `--` before the name — swapped in tests via `App::unit_control`
  and `App::unit_journal`, and refused unless `samples_this_host()`.
- **SMART** — `smart::SmartWatcher` runs `smartctl --json -i -H -A -n standby`
  per disk (`--scan --json` lists them, with the `-d` type to read each as)
  every `[smart] interval_secs` on its own thread, through `[smart] command`
  so `sudo -n smartctl` works, each run bounded at 30 s by
  `remote::wait_with_timeout`. smartctl's exit status is a bit mask, not a
  verdict: bits 0 and 1 mean no reading, bit 3 means the disk says it is
  failing, and the rest describe history, so the JSON is judged, not the
  status. A disk in standby keeps its last reading rather than being woken.
  Disks become `StorageArray`s of kind `smart` carrying a `SmartInfo`, so the
  panel, `storage` rules, the exporter, `--remote`, recordings and incident
  summaries need no second path; the panel draws failing disks with the
  arrays and healthy ones only in the room left after the mounts. A pass that
  reads nothing sets `Snapshot::smart_unreadable` instead of an empty, healthy
  list, and `--doctor` fails while `[smart]` is on and no disk can be read.
- **cgroups** — the panel appears only when the process is actually confined,
  since a limitless root cgroup is noise.

## Alert rules

A rule is a measurement, a threshold and a direction. `below` inverts the
comparison, and is *strict* where the default is inclusive: the obvious
`threshold = 1` on a `proc` rule has to mean "none running" rather than firing
while the one process is still up.

`hysteresis` is a finite, nonnegative recovery margin in the measurement's
unit, defaulting to zero. It applies only after firing: above rules recover
strictly below `threshold - hysteresis`, below rules at or above
`threshold + hysteresis`. Hold times still use the original firing boundary.
A rule inside the margin remains active without rerunning its hook, and its
message shows the recovery boundary rather than claiming it is above the
firing threshold.

Rule state is keyed by **position, not name**. Nothing makes names unique, and
two rules sharing one — the pair `below` makes obvious, one over a threshold
and one under — shared a `fired` flag, each clearing what the other had just
set, so the hook command re-spawned on every refresh for as long as the
condition held.

`proc` rules count the processes matching a filter query, parsed once at
construction. A query that fails to parse leaves the rule inert rather than
counting zero, because counting zero would make every `below` rule fire for
ever on a typo, hook and all.

A hook is `sh -c` with the context in its environment rather than in the
command: `CRABMON_RULE`, `CRABMON_VALUE`, `CRABMON_STATE` and the rest, listed
in `alerts::HOOK_VARS` and checked against both documents by `tests/docs.rs`.
Every variable is always set, empty where it does not apply, so a hook written
with `set -u` need not guess which exist. The two that cannot come from the
engine — which machine the rule was about, and which flight recording this
pass opened for it — are filled in by `supervisor::attach_context`, because
the dump does not exist until `drain` has written it.

Hooks and notices follow what was last *announced* rather than the edge just
crossed. Each rule's `Tracked` state keeps `announced` beside `fired`, and a
call is queued whenever the two differ and no silence covers the rule. Without
a silence they move together and nothing changes; across one they come apart,
and the first unsilenced evaluation reconciles them — the held firing page
when the rule is still over, the recovery when it was announced and has since
cleared, nothing when it came and went unseen. `pending_fired`, which the
flight recorder reads, is still pushed on the edge itself: silencing a page
does not discard its evidence. Silences come from `[[silence]]` windows,
loaded by `AlertEngine::configured`, and from `App::muted`, which the
interface hands to every engine it runs — the selected host's and each
parked one — before evaluating it; the fleet's ALERT column checks them at
draw time, so a new silence empties it at once.

Every transition is also queued, hook or no hook, as a *notice* — the same
`HookCall`, with the same context attached — which is what `[[notify]]`
consumes. `notify.rs` turns one into a `curl --config` file and hands it to
`supervisor::spawn_curl`, which runs `curl --config -` under the hook
machinery (own process group, lazy reaping, `HOOK_TIMEOUT`) and, unlike a
hook, reports a non-zero exit. The config travels on stdin because a Slack or
Discord webhook URL is a credential, and argv is readable by every user;
every value in it is quoted, and header values lose control characters, so a
rule name cannot inject a second `url` or header. The engine bounds the
notice queue at `MAX_HISTORY`, because `--watch` and the tests never drain it.

Under a fleet the TUI runs one engine per host. The selected host's is
`App::alerts`; the rest are parked in `App::host_alerts` and evaluated by
`evaluate_parked_hosts` on every tick against each host's latest frame —
once per frame, by its `taken_at_unix`, and interrupted while a host is not
answering. They feed the ALERT column, hooks and notifications, never the
flight recorder, whose ring is the selected host's frames. Moving off a host
parks its engine with the frame it last saw marked seen, so the switch does
not evaluate a frame twice.

A rule with nothing to measure is skipped rather than read as zero. That is
what makes `kind = "fd"` safe to leave in a config on a machine where
`[procs] fds` is off, and `cpu.full` report nothing on the kernels that
publish no `full` line for CPU rather than silently falling back to `some`,
which measures something else entirely.

## Recording and replay

A recording is JSON Lines, one `Snapshot` per line, flushed after every frame so
a killed session still leaves a usable file. Because the UI reads nothing but a
`Snapshot`, replaying one exercises every panel exactly as the live source
would.

Frames are trimmed before they are written, because they are otherwise
unusable: a full process list on a busy machine is ~1.5 MB per frame, and an
hour would be gigabytes. Three limits bring that to ~55 KiB — the busiest 100
processes, argv truncated to 200 characters, and `skip_serializing_if` on empty
strings and `None`s. Trimming ranks real processes above threads: ranking by CPU
alone fills the budget with one busy application's threads, and since the UI
hides threads by default the replay would then look almost empty.

Every snapshot type carries `#[serde(default)]`, so a recording made by an older
crabmon still loads when fields are added.

A replay starts `record::scan_timeline` on a thread of its own: one pass over
the file, a line and a frame at a time whether the replay is indexed or not,
collecting each frame's time (for `:`) and the markers (for `(` and `)`).
Alert markers come from running today's rules over the recording through a
fresh `AlertEngine`, with the clock taken from the frames' own timestamps, so
hold times mean recorded seconds. Kernel markers are events whose sequence
number is newer than any seen so far, skipping the first frame's backlog.
Positions match `ReplaySource`'s, except at the very end: a truncated final
line is held back until a later line proves it was not the end, and an
indexed replay counts it as a frame, so the two can differ by one there.
Seeks are clamped to the replay.

Credential redaction is applied to a copy of the snapshot before command
truncation. `[record] redact_commands` controls plain, rotating and flight
recordings and incident summaries; `[export] redact_commands` controls snapshot
exports and stream output. Both default to true. The shared implementation is
`export::redact_command`: common credential arguments, bearer tokens and URL
userinfo are hidden. Unquoted credential values hide the remainder of the
command because the joined command cannot recover argv boundaries. Positional
secrets and unknown credential names are outside this heuristic. Local live
filters retain raw data; remote filters see what the remote host exported.
Unit journal tails get `export::redact_log`, applied where they are read
(`units::redact_lines`, from `[units] redact_logs`) so nothing leaves the
unit's machine unredacted. It is deliberately narrower than
`redact_command`: a log line is prose, not argv, so it hides a credential's
*value* — after `key=`, `key:` or `"key": `, up to a space, comma,
semicolon or `&` — rather than the rest of the line, which would erase the
reason the unit failed along with the secret.

### Where a recording comes from

`record::Recordings` is the one place that turns `--record`, `--record-rotate`
and `[record]` into open files. The interface, `--stream`, `--serve` and
`--watch` all hold one and offer it every frame they sample.

That is a single type rather than four copies because the resolution is not
trivial: the rotation directory falls back through `--record-rotate`,
`[record] rotate_dir`, `[record] flight_dir` and finally the working
directory, and the flag turns rotation on by itself where the config alone
would leave it off. Four copies of that is four chances to get it wrong, and
the three that are not the interface have nobody watching a status line to
notice.

A fleet's host names are chosen by `record::unique_labels` over the whole
target list rather than one target at a time. `sanitise_label` reduces a
target to something a filesystem will take and is not injective — `user@host`
is the ordinary way to write an SSH target, so `deploy@db-2` and
`deploy-db-2` both reduce to `deploy-db-2`, and `@@@` reduces to nothing.
Either is a quiet disaster: a shared directory means the second host truncates
the first's segments, because a recorder truncates what it opens, and a
nameless host falls back to the unlabelled path and writes among the flight
dumps, where rotation prunes them. A name already free is kept, so an
ordinary fleet is unchanged.

Where a recording goes is checked at start: an explicit `--record` path that
cannot be created, or a rotation directory that is not one or cannot be made,
stops crabmon with the reason rather than running for ever recording nothing.
After that, a write that fails is reported, never fatal: a full disk under `--serve`
would otherwise stop the exporter answering, which is worse than a gap in a
recording. It is reported on the way *into* failure and again on the way out,
not once a frame — a read-only directory does not improve by itself, and an
exporter scraped every fifteen seconds would otherwise write an identical
line four times a minute for as long as it ran. The same shape as an alert,
which fires on the crossing rather than on every sample past the line. The interface reports into its status line and the headless modes
onto stderr, which under `--stream` is the only place they can go — stdout is
a frame per line that something else is parsing.

### The flight recorder

`--record` only helps if it was started before the incident, which is the one
thing nobody does. `FlightRecorder` holds the last `pre` frames in memory at
all times and, when a rule goes active, writes them plus the next `post` frames
as an ordinary recording.

Three details:

- **Frames are trimmed on the way *into* the ring, not on the way out.** A
  ten-minute window of untrimmed frames is well over a gigabyte of live memory,
  in the process whose job is to make memory pressure visible.
- **The ring is written before the frame budget starts.**
  `supervisor::drain_named` pushes the frame *then* handles triggers, so the
  ring already contains the frame that tripped the rule; the other order would
  stop every recording one frame short of the evidence. Both callers — the TUI
  event loop and `Supervisor::observe` — go through it, which is the point:
  this ordering is the kind of thing that ends up right in one copy and wrong
  in the other.
- **A trigger while a dump is running extends it** rather than opening a second
  file. A flapping rule would otherwise leave a file per refresh, each one
  mostly the same frames.

## Sampling off the draw thread

`App::tick()` called the metric source directly from the event loop, so anything
slow in a sample froze the whole program — not only the numbers, but the
keyboard, including the key that quits. Three real cases: an `nvidia-smi` that
takes its time, a `statvfs` on a wedged NFS mount, and an SSH round trip.

`sampler::ThreadedSource` wraps any `MetricSource + Send` and moves it onto a
worker. `snapshot` drains whatever the worker has finished, asks for one more if
nothing is outstanding, and returns immediately — so the loop is never slower
than a redraw. Three details matter:

- **The first sample is taken synchronously, in the constructor.** Otherwise the
  dashboard opens blank for a whole refresh interval.
- **Only one request is outstanding at a time,** and a backlog collapses to the
  newest frame. Queueing work a slow source can never catch up on would turn a
  brief stall into a permanent lag.
- **A repeated frame is not recorded twice.** `frame_id` tells `App` the sample
  has not changed, so the charts keep their last real point instead of growing a
  flat line that looks measured. The status line says `sampling…` while this is
  happening, because a dashboard that has quietly stopped moving is worse than
  one that explains itself.

A replay is deliberately *not* wrapped: it is instant anyway, and the scrub keys
need `seek`/`peek` to be synchronous.

## Remote monitoring

`--remote` used to run `crabmon --once` over SSH for every sample, paying a TCP
handshake, a key exchange and a full crabmon startup per frame. It now starts
`crabmon --stream` once and reads JSON Lines from the session for as long as it
lasts, with the reader on its own thread and stderr drained on another — a full
stderr pipe would otherwise block the far end mid-frame and stall the stream
with nothing to show for it. The frame channel holds at most two snapshots;
a reader that gets ahead blocks rather than growing a queue without limit.

Falling back to the old behaviour is deliberately narrow: only the far end's
argument parser refusing `--stream` triggers it. Falling back on any failure
would paper over a genuinely broken host, and an explicit `--remote-command` is
never second-guessed at all. It is, however, *read* according to what it asks
for: one with `--stream` as a word of its own (`remote::command_streams`) is a
stream with no fallback, anything else a one-shot. Every explicit command used
to be a one-shot, so the only way to reach a crabmon off the far end's `PATH`
gave up streaming, and a streaming one waited out `FETCH_TIMEOUT` on every
sample and never came up.

The one-shot path is bounded by `FETCH_TIMEOUT`. `ConnectTimeout` only covers
setting the connection up; once SSH is connected, a command that never returns
hangs the fetch, and because the first sample is synchronous that meant the
interface never appeared and there was nothing to press `q` in. The wait drains
both pipes on threads, because a child that fills one and blocks never exits —
polling `try_wait` alone cannot tell "slow" from "deadlocked on a full pipe".

## Layout

`ui::draw` renders a header, one of five layouts, and a status line, then any
popup on top.

| Layout | Content |
| --- | --- |
| `Dashboard` | CPU · memory · swap · processes, with a configurable right column |
| `Processes` | the process table full-width |
| `Cpu` | one bar per core, in as many columns as fit, plus the average chart |
| `Io` | network chart and table, disks, GPU |
| `Fleet` | one row per host, when several `--remote` targets are being watched |

The right-hand column's contents and order come from `panel_order`, and every
panel can be switched off in `[panels]`.

The process table's columns are chosen to fit the available width: the marker,
PID, CPU and memory plus the command are always present, and state, user, disk,
time, nice, the CPU sparkline, virtual size and thread count are added in that
order as the terminal gets wider (`procs::OPTIONAL`). PORTS, FD and PSS are not in
that set at any width: each costs syscalls per row or per process, so they join
only when `[procs] ports`, `[procs] fds` or `[procs] pss` asks for them — and
having been asked for, they outrank the columns that are free. GROWTH and
MAJFLT appear only when named in `[procs] columns`. Whichever subset is chosen
is then drawn in one canonical left-to-right order rather than in the order it
was added. A property test checks the chosen set fits at every width from 20 to
199 columns.

## Terminal lifecycle

Three failure modes are handled explicitly, because each one leaves a broken
terminal or a runaway process:

- **Panic** — a hook restores the terminal before the default handler prints.
- **SIGTERM** — sets a flag the event loop checks, so the config is saved and
  the terminal restored on the way out.
- **The terminal going away** — `crossterm::event::poll` never returns once its
  pty is hung up; it spins internally, so a flag-based SIGHUP handler would
  suppress the kernel's default kill and leave the process burning a core
  forever. SIGHUP is therefore deliberately left at its default action. A
  watchdog thread covers the same wedge for SIGTERM: once the stop flag is set
  it gives the loop a short grace period, then calls `_exit`, which is the only
  exit that cannot deadlock on a stdout lock the wedged main thread is holding.

## Testing

Over 900 tests: the in-crate unit tests plus eight integration suites. All are
offline and deterministic except `proc_control` and `e2e_local`, which exercise
the running machine, and `e2e_docker`, which builds and runs a container:

- **unit tests** (`src/**`) — formatting, filter parsing, sorting, tree
  flattening, diskstats/cgroup/PSI/meminfo/`iostat`/`nvidia-smi`/`limits`
  parsing, GPU and battery sysfs against fabricated trees, socket-table
  decoding, alert state, column planning, base64, the Prometheus exposition
  format, recording round-trips and config round-trips; `systemctl show`
  parsing, restart tracking and journal caching; kernel-message patterns;
  mdstat, ZFS and btrfs health; SNMP/netstat counters; container naming and
  the Engine API reply; silence windows; fleet target expansion and sorting;
  the notify `curl` config; and incident summaries. Also the arithmetic
  that is easy to get quietly wrong: disk utilisation against a counter reset,
  an idle interval having no average service time, per-device rather than
  per-mount aggregation, and every alert kind against a snapshot built to make
  it fire.
- **`tests/app_behaviour.rs`** — the interaction model against a fixture source:
  key handling in every mode, selection stability across re-sorts, PID-reuse
  refusal for signal, renice and affinity (IO priority is covered for prompt
  validation), the fleet table's sorting, host selection and per-host alerts,
  the unit popup, `S` silences, tagging and bulk actions, pinning,
  saved filters, grouping and group drill-down, popup scrolling, replay
  scrubbing, audit logging, mouse handling, history bounds, per-process series
  accumulating and being evicted with the process, and that a repeated frame
  from an asynchronous source never reaches the charts.
- **`tests/render.rs`** — the real panels drawn onto a `TestBackend`, asserting
  on screen content, at sizes from 10×3 up to 300×100 and in every theme,
  including the FD column only appearing once descriptors are counted and the
  detail pane's trend rows.
- **`tests/cli.rs`** — the built binary: `--help`, `--version`, exit codes,
  `--once` JSON/CSV output, `--diff` as text and JSON, `--stream` read back
  through the recording parser, `--watch`'s exit codes, and the refusal to
  combine a headless mode with `--remote` or `--replay`. Also that recording
  works without a terminal — `--stream` leaving a parseable recording no wider
  than the frames it printed, `--watch` leaving a rotating one of the run-up,
  and `--once` refusing both flags and writing nothing — and that `--stream`
  keeps to its interval, bounded against a sample cost the test measures on
  the machine it is running on rather than a fixed multiple of the refresh.
  And `--doctor`: that every check is marked and identified, that
  `--format json` parses, that it reports on the config `--config` actually
  named, and that it exits 1 for a rule this host cannot measure and 0
  otherwise.
- **`tests/proc_control.rs`** — renice, CPU affinity and signals against a real
  spawned child, checked by reading the state back from the kernel, plus
  identity: a process that execs is reported as what it became, and a wrong or
  exited incarnation is refused. A unit test
  can only prove the arguments were well-formed, not that they were accepted.
- **`tests/e2e_local.rs`** — the binary on the current Linux host: complete
  streaming with over 100 real child processes, fleet exporter freshness over
  a controlled SSH transport, an explicit `--stream` remote command read as a
  stream, SMART through a stand-in `smartctl` reaching `--once`, the text
  report, `--watch-rule` and `--doctor`, and replay in a pseudo-terminal with
  hook and flight-recording side effects forbidden. Workloads and files are isolated
  and cleaned up by the tests; Docker is not required.
- **`scripts/e2e-ssh.py`** — opt-in checks on a real Linux SSH host using a
  compatible release binary. Creates a private remote `/tmp` directory and
  temporary workloads, then removes them. Covers credential redaction and
  opt-out, hysteresis hooks, streams, plain/rotating/flight recordings, and a
  real SSH collector through stalls, failures and recovery, that a host
  that never answers exports only `up` and `sample_stale`, that an explicit
  streaming `--remote-command` comes up, and SMART through a stand-in
  `smartctl` (healthy, degraded, unreadable, the alert naming the disk once)
  plus, where smartmontools is installed, the real one read-only. `--units` (root on
  the target) adds transient systemd units — one failing, one crash-looping —
  and checks detection, the exit status, `unit` alert messages naming the
  unit, and recovery on `reset-failed`; the units are stopped and reset
  afterwards. Run with
  `python3 scripts/e2e-ssh.py --target user@host --binary target/release/crabmon`.
- **`tests/build_script.rs`** — the build tooling against itself: that every
  `make` target has a command behind it, that every command is reachable through
  make and described in the script's own help, and that the script runs, reports
  the crate's version and fails with a status on a mistyped command. `make` is a
  front end to `scripts/build.sh`, and a front end that has drifted from what it
  fronts is found when you need the command, not before.
- **`tests/e2e_docker.rs`** — the shipped binary, built from this checkout by
  `tests/e2e/Dockerfile` and run inside a container with an empty network
  namespace: `--once` in both formats, a filter that must be refused, `--watch`
  in both outcomes, `--stream` read back frame by frame, `--diff` across a
  recording, and the exporter scraped over the container's own loopback.

  What this buys over `tests/cli.rs`, which also runs the real binary, is a
  machine whose contents are known. On a developer's laptop "is pid 1 reported"
  is a question about that laptop; in the container pid 1 is the command the
  test started and the only other processes are the ones it spawned, so a
  process appearing between two samples, or changing its name under `exec`,
  is an assertion rather than a hope. It is also where the binary is tested
  against the libc it was built against rather than the one that happens to be
  installed.

  It is also where the features that only exist across machine boundaries are
  tested: per-container network through a second container's namespace, and
  alert rules firing their hooks and writing flight recordings with no
  terminal anywhere.

  **Hermetic**, specifically: every container gets an empty network namespace,
  a read-only image, a tmpfs for scratch and a fixed hostname and environment.
  A test cannot reach the network, cannot see the machine running the suite,
  and cannot find anything an earlier test left behind — which is what lets
  these run in parallel and in any order. Two of the tests assert exactly
  those properties, because a sandbox nothing checks is a sandbox that quietly
  stops being one: loosen a flag and they fail rather than the suite silently
  becoming less isolated. Nothing is installed into the image either; the
  exporter is scraped over the container's own loopback by a twenty-line bash
  client, where an `apt-get install curl` would have made every run depend on
  whatever a Debian mirror held that day.

  `make e2e` needs **Docker and nothing else, not even Rust**. The harness is
  a `cargo test` suite, so a toolchain has to exist somewhere;
  `tests/e2e/Dockerfile.runner` supplies one, together with a Docker CLI
  copied from the official image rather than installed. The checkout goes in
  read-only and cargo's output goes to named volumes, so a run cannot leave a
  root-owned `target/` in somebody's tree. The daemon is the host's, so the
  containers the tests create are siblings rather than nested — no privileged
  mode, no daemon inside a daemon — at the cost of handing that container the
  host's socket, which is root-equivalent access to the machine.

  `make e2e-local` is the same tests driven by whatever toolchain is already
  installed: no runner container, nothing given the socket, output in the
  ordinary `target/`. That is the one to iterate in, and the one for anyone
  who would rather not hand a container their Docker socket. Both are CI jobs,
  the first with no toolchain installed at all — a path nothing exercises is
  broken by the time somebody needs it.

  The image is pinned — both bases by digest, the dependencies by `Cargo.lock`
  with `--locked` — so the same checkout builds the same container. Building it
  does fetch those dependencies from crates.io, which is the one input that
  still needs the network. `cargo vendor` would close it and costs 446 MB in
  the repository (44 MB without the Windows crates this image never compiles),
  which is not a trade worth making for a test suite: pinning already fixes
  *what* is fetched.

  It is **opt-in**: `make e2e`, or `CRABMON_E2E=1 cargo test --test e2e_docker`.
  Without that every test in it reports that it was skipped and passes, because
  a suite that fails on a machine with no Docker is a suite people delete.

- **`tests/docs.rs`** — that the man page, completions and README still describe
  the flags, sort keys, themes, layouts, groupings, panels, config fields and key
  bindings the code actually has; that the aliases documented in one place are
  documented in the other; that the packaged description covers the current
  feature set; and that the man page's version matches the crate's.
  Documentation drift is a build failure, not a review comment.

  These checks derive their expectations from the code — a serialised default
  `Config`, `cli::LONG_FLAGS`, `ALL_SORTS`, `PRESETS`, `ALL_GROUPINGS`, the
  `KEYS` table, `ALL_TREND_SERIES`, and the key literals parsed out of
  `on_normal_key` — rather than from a hand-maintained list that could go stale
  in the same way the docs do. `--help`'s own enumerations are checked too: its
  `--layout` row sat at four layouts for the whole life of the fifth.
  They are also scoped to the relevant section of each document, so an
  incidental mention of a word elsewhere cannot satisfy them.

  The lists that are enumerated in prose — `[procs] columns`, the alert kinds,
  the filter fields — are compared as *parsed lists* rather than by substring.
  `contains("fd")` is satisfied by the word "fds" three lines further down,
  which is how the man page lost the `fd` column with the suite green.

## Configuration & persistence

`Config` has a default for every field, `#[serde(default)]` on the struct, and a
`sanitize()` pass that clamps anything a hand-edited file could set to a value
that would break rendering or make the metrics meaningless. A corrupt file falls
back to defaults rather than refusing to start.

The config path is explicit state on `App`: `config_path: Option<PathBuf>`, and
`None` disables saving entirely. An `App` constructed by a test or an embedder
must never write to the user's real config file as a side effect of being used.

## Packaging

`scripts/build.sh` is the build driver: building, testing, linting, packaging,
installing under a `PREFIX` and cleaning up after itself. The `Makefile` is a
front end to it — one line per target — rather than a second implementation, so
`make test` and `./scripts/build.sh test` cannot come to mean different things.
A pair of tests in `tests/build_script.rs` assert that the two lists agree and that the
script's own help covers every command it accepts.

`make snapshot` redraws the terminal capture the README carries. It drives a
real pty through tmux and takes `capture-pane`'s output, so the braille
sparklines and the box drawing are what the interface actually drew rather
than an approximation of them; it builds release first, because a debug binary
spends enough CPU to put itself at the top of its own process table. The
capture is of whatever machine runs it, which is why the README's copy is
replaced between markers rather than added to. `tests/docs.rs` compares its
process-table header against `ui::procs::ALL_COLUMNS` as a token list, so a
renamed column makes the snapshot a build failure rather than a quietly wrong
picture.

`make deb` (still reachable as `scripts/build-deb.sh`, the name CI uses) builds
a release binary and runs `cargo-deb`, which installs the binary, the man page,
bash/zsh/fish completions and the prose under `/usr/share/doc/crabmon` — the
README, this document and the changelog, the last of which Debian expects at
that exact path. `make dist` produces the same file set as a tarball for the
platforms the deb does not cover; `make install` is the binary, the man page
and the completions only, since the prose has nowhere agreed to go under an
arbitrary `PREFIX`. CI runs the test-suite on
Linux, macOS and Windows, checks formatting and clippy, and cross-checks the
cfg-gated code paths against aarch64 Linux and FreeBSD.

## The exporter

`--serve` is a hand-rolled `TcpListener` loop rather than a framework, because
the whole contract is one method and four routes (`/`, `/metrics`, `/healthz`,
and 404 for everything else); any other method is a 405. With `[serve] token`
or `CRABMON_SERVE_TOKEN` (which wins) set, every route but `/healthz` needs
`Authorization: Bearer`, compared in constant time by `secret_eq`; a missing
or wrong token is a 401 with `WWW-Authenticate`. It samples once per scrape, on the connection
thread serving it, behind a mutex. Per-process series are capped (default 20) — one series per process on
a 2000-task machine would dwarf every other metric in the scrape and make the
exporter the most expensive thing on the box.

Without `--remote`, it samples the local host. `--serve --remote` exports a
fleet with a `host` label on every series and honors `--remote-command` for
nonstandard binary locations. `--serve --replay` is refused. The local-only
`--once`, `--stream` and `--watch` modes refuse both `--remote` and `--replay`: `crabmon --once --remote prod-db` used to print the *local* machine's
snapshot with exit 0, which is a worse failure than an error, because a script
gets plausible data about the wrong host and never finds out.

`serve` measures the gap between scrapes and passes it to the source, because
rates are deltas of monotonic counters divided by the interval they are handed.
Passing the *configured* refresh instead — which is what it used to do —
multiplied every rate series by `scrape_interval / refresh_ms`: 200 MiB pushed
over loopback across an 18 s gap at the default 800 ms interval reported
250 MiB/s rather than 11.6 MiB/s. Gauges carry no delta and were unaffected,
which is what made it easy to miss.

The whole request is read before anything is answered, even when nothing is
going to look at a header. Reading only the request line looks like a saving
and is a bug: closing a socket with unread bytes still in its receive buffer
makes Linux send RST rather than FIN, and the RST discards the response that
was just written. It is a race, so it hides on a fast machine — on a
single-core one it was losing more than half the scrapes, which Prometheus
records as failures rather than as anything crabmon said.

Each connection is handled on its own thread, capped at `MAX_CONNECTIONS` with
a 503 past it, and both socket directions carry a timeout. Request reads also
share one five-second deadline, so a trickling client cannot renew its slot
indefinitely. Incomplete headers are rejected before a scrape is generated. Doing the socket IO
on the accept loop meant a client that connected and never sent a request — a
stalled load balancer health check will do — stopped the exporter answering
anybody, with no error anywhere; a timeout alone only bounds how long that
lasts. Sampling stays behind a mutex, so concurrent scrapes see consistent
counter deltas instead of racing each other's `prev` state.

A bare `--serve 9100` binds to loopback; `:9100` keeps its conventional "every
interface" meaning and says so at startup. Per-process series expose the
busiest processes' names and PIDs, not their command arguments.

For remote hosts, `supervisor::RemoteFreshness` distinguishes new measurements
from repeated frames. `ThreadedSource` carries the worker's monotonic arrival
time, unchanged when the underlying frame ID repeats. Delayed scrapes cannot
make old data young, and remote wall-clock skew does not affect age.
`crabmon_sample_age_seconds` is omitted before first contact;
`crabmon_sample_stale` is 1 while connecting, failed, or without a new frame
for at least the larger of five seconds and three refresh intervals.
`crabmon_up` is 0 in those cases. Stale samples neither complete pending
alert hold timers nor get recorded again; an already firing rule awaits a
real recovery measurement.

## Answering for the machine

`--doctor` exists because of the rule above. crabmon never reports zero for
something it could not read — an unreadable fd table is unknown, a root-only
`/dev/kmsg` says so, a rule with nothing to measure is skipped — which is
right, and which leaves "is that nothing, or something I am not allowed to
see?" as a question the interface cannot answer. `doctor::run` asks all of
them once and says so.

It is a *check*, not only a report: it exits 1 when something the config
actually asks for cannot work here, and 0 otherwise. That distinction is the
whole design. Missing optional capabilities such as a GPU, pressure stall
data or kernel events cost nothing, even when the report offers a remedy.
Checks for the config file, enabled fd sampling and alert rules are marked
as required. `Report::has_problems` means an unavailable required check;
diagnostic advice alone does not cause a failure. An alert rule that cannot
measure anything still fails the check, because a rule that never fires
reads as a quiet machine.

## Supporting subsystems

- **The events timeline (`events.rs`)** — `Timeline::observe` compares each
  host's snapshot with the last one from the same host: kernel events newer
  than the last sequence number seen (the ring read at start included, timed
  by boot time plus their monotonic stamp), units changing state, restarting
  or dropping out of the snapshot (which means healthy again), and storage
  rows changing state. Rule transitions come from the engine's history by
  `AlertEvent::seq`, a process-wide counter assigned when a transition is
  recorded, so an engine rebuilt by a reload, a replay seek or a fleet host
  switch is neither re-read nor missed. Fleet hosts are observed as their
  frames arrive in `evaluate_parked_hosts`, which also reports a host that
  stops or starts answering; actions and silences are added where the action
  log is written. What a host is already in at its first snapshot is reported
  "(at start)" rather than as a change. Nothing is sampled for it, so a replay
  rebuilds the same timeline. A jump in a replay resets it, because the frames
  skipped are time nobody watched, and a single step forward (`]`) continues
  it; either way the frame landed on is observed at once. Bounded at `events::CAPACITY` (500), kept in time order.

- **Silences (`silence.rs`)** — a `[[silence]]` window is either one-off
  (`start`/`end`, like `2026-11-20T02:00Z`) or `daily = "HH:MM-HH:MM"`, which
  may cross midnight. Everything is UTC; offsets are refused rather than
  guessed. A window can be narrowed to rule names and to fleet targets or
  hostnames, and `validate` refuses malformed or empty windows at start. The
  `S` prompt parses `1h [rule]` or `off [rule]` into a `Muted { rule,
  until_unix }`; `covering` checks interface silences before configured ones
  and reports the one that reaches furthest.
- **Fleets (`fleet.rs`)** — `FleetSource` holds one `ThreadedSource` per
  target, so hosts are sampled concurrently and a dead one costs the rest
  nothing. Its `frame_id` advances when any host produces a new frame, and
  `is_local` is always false. `--remote a,b` lists targets, `@name` expands a
  `[[fleet]]` group, and duplicates keep their first position. `FleetSort`
  has ten keys (as named, host, cpu, mem, swap, load, disk, net, procs,
  alerts), cycled with `>`/`<` and reversed with `s`; figures sort descending,
  hosts without data sort last except under `alerts`, where an unreachable
  host ranks worst, and the sort is stable so equal hosts do not jitter.
- **Themes (`theme.rs`)** — five presets (default, mono, nord, solarized,
  gruvbox; `--no-color` is mono). `[colors]` overrides individual roles with
  names, `#rrggbb`, a 256-colour index or `reset`; a bad entry is ignored
  rather than fatal. Temperatures colour against the sensor's own critical
  point, or 70/85 °C when it reports none. Selection falls back to REVERSED
  when both selection colours are `reset`, so it never disappears.
- **Clipboard (`clipboard.rs`)** — `y` writes OSC 52 straight to stdout, so a
  copy works through SSH and tmux with no clipboard daemon. The payload is cut
  to 8 KiB, past which many terminals drop it silently.
- **Incident summaries (`incident.rs`)** — a Markdown file beside each flight
  dump (same name, `.md`): the rule, measurement and threshold (and "Paged:
  no" when silenced), machine state, kernel events, units in trouble, the top
  eight processes, a `diff::compare` of the dump's first frame against the
  tripping one, and the command that replays it.
- **Diffs (`diff.rs`)** — processes are matched by PID *and* start time, so a
  recycled PID is a start and an exit rather than growth. A process has moved
  if its memory changed or its CPU moved by more than half a point; lists sort
  by absolute memory change, and disks compare per mount. `load_ends` takes a
  single JSON document or a JSONL recording, whose first and last frames are
  compared.

## Known limitations

- Process CPU percentages are relative to a single core, as `sysinfo` reports
  them, so a busy multi-threaded process can exceed 100%.
- `nvidia-smi` figures are one sample behind, and the first sample waits up
  to two seconds for an answer so `--once` has one.
- The FreeBSD disk-IO path shells out to `iostat -x` per tick and reports
  since-boot averages rather than instantaneous rates — which now applies to
  the utilisation and latency figures it reads out of `%b` and `ms/t` as well,
  so a FreeBSD host's "% busy" is its average since boot and not what the disk
  is doing now. It is compile-checked in CI and unit-tested against captured
  output, but has never run on real FreeBSD hardware.
- `--remote` streams by default (`[remote] stream`, on): one SSH session and
  one `crabmon --stream` for the whole session. The one-shot fallback, taken
  when the far end is too old to understand the flag, does re-run
  `crabmon --once` per sample, and there the refresh rate is bounded by the
  round trip and the far end pays a full startup each time.
- The process list is rebuilt and re-sorted on every refresh rather than
  incrementally maintained. At ~2000 processes and an 800 ms interval this costs
  a low double-digit percentage of one core, which is in line with comparable
  monitors but is the obvious next thing to optimise.
- A replay over 8 MiB is indexed by line offset and parsed one frame at a
  time; smaller ones are still read whole, where holding them costs less than
  indexing them. The index itself is one `(u64, u32)` per frame.
- The flight recorder's ring is bounded in *frames*, not bytes. A machine with
  an unusually long process list and a large `[record] top_n` costs
  proportionally more per frame than the figures quoted in the docs.
- `[procs] pss` adds one `smaps_rollup` read per *process* per sample
  (threads are skipped, which on a 3300-task machine with 530 processes is
  most of the saving). It is the same trade as `fds` and off for the same
  reason.
- Per-container cgroup accounting reads three or four small files per
  container or unit per sample, capped at `cgroup::MAX_CGROUPS`. The paths
  come from the process list that was just built, so nothing is enumerated
  that is not running.
- The GROWTH column and `mem_growth` read the history already kept for the
  TREND column, so they cost a subtraction. The window they measure over is
  `PROC_HIST_LEN` samples of whatever the refresh interval is, not a fixed
  wall-clock span, so changing `refresh_ms` changes what "growth" covers.
- `[procs] fds` adds a `getdents` sweep per process per sample. Measured on a
  2800-process machine at the default 800 ms refresh it roughly doubles
  crabmon's sampling cost, from about 10% of one core to about 20%. That is why
  it is off by default, and it is the second thing — after the process-list
  rebuild above — that a busy host pays for.
- Per-process history is kept for every process in the snapshot, not only the
  ones being looked at, so that the detail pane has something to draw the
  moment it opens. At `PROC_HIST_LEN` samples of three series that is a few
  megabytes on a 2000-process machine.
- The `ports` column walks one fd table per visible row, which is why it is off
  by default — `[procs] ports`, or naming it in `[procs] columns`, is what asks
  for it — and resolved only for rows actually on screen. A process holding tens
  of thousands of descriptors is cut off at `MAX_FDS_SCANNED`, which bounds the
  walk itself rather than the list it returns.
- `--once` and `--diff` never fire a rule: they take one sample, and a hold
  time needs a sequence. `--serve --alerts`, `--stream --alerts` and
  `--watch` all evaluate rules, through the same `Supervisor` the TUI uses.
- `--record` and `--record-rotate` work in every mode that samples on a loop:
  the interface, `--stream`, `--serve` and `--watch`, all through one
  `record::Recordings`. `--once` and `--diff` refuse them — a recording is a
  sequence and they take one sample. A fleet records one timeline per host,
  with distinct file and directory labels. Under local `--serve`, cadence
  follows scrapes; a remote host is recorded only when a new live frame is
  observed on a scrape.
- `--diff` renders JSON, long-form CSV or the text report. Only the text
  report is cut to `LIST_LIMIT` rows per list.
- `supervisor::spawn_hooks` reaps finished hooks lazily, on the next tick,
  and kills a hook's whole process group once it has run for
  `HOOK_TIMEOUT` (60 s), reporting that it did. A hook that needs longer has
  to detach itself.
- `app::ProcSeries` (through `push_bounded`) drops the oldest sample with `Vec::remove(0)`, which
  is a memmove of `PROC_HIST_LEN` elements per series per process per sample.
  At 120 elements that is under half a kilobyte and does not show up against
  the cost of taking the sample; a `VecDeque` was considered and not taken,
  because the sparkline and detail pane draw from contiguous slices.
- Crash-loop detection counts only restarts crabmon has watched happen: a
  loop already under way at start is reported as restarting at once
  (`SubState=auto-restart`) and as a crash loop after `[units] crash_loop`
  more restarts. `--once` therefore never reports a crash loop, only failed
  and restarting units. User units are not read.
- `[smart]` needs smartmontools 7.0+ (for `--json`) and the privilege to
  open the disks — root, or a sudoers rule behind `command`. Disks behind a
  RAID controller are read only where `smartctl --scan` lists them with a
  type (`megaraid,N`); USB bridges that do not pass SMART through report
  nothing. It has been unit-tested against smartctl's JSON shapes, not yet
  run against a failing disk.
- Kernel events in the timeline are timed by the sample that first saw them,
  one refresh late at worst; only the ring read at start is timed by boot time
  plus the kernel's printk stamp, which drifts from wall time (17 s behind on
  a two-week-old VM), so backlog times are approximate. The Kernel panel's
  "ago" uses the same stamp and carries the same drift.
- The events timeline lives in memory for the session: it is not written
  anywhere, and a restart begins it again (with what is already wrong marked
  "at start"). Recordings carry everything it is derived from, so a replay
  rebuilds it.
- `[[notify]]` needs `curl` on the machine running the rules; crabmon links
  no HTTP or TLS client. A missing `curl` is reported per post, not at start.
- The replay timeline scan parses every frame once. On a 240 MiB hour that is
  seconds of one core, on its own thread; markers and time seeks are
  unavailable until it finishes.
- Under a fleet, every host's rules run each tick in the TUI. That is one
  rule evaluation per host per tick, against frames the workers have already
  fetched; nothing extra crosses the network.
- Pins do not reorder the tree view. The ordering there is structural, and
  hoisting a child out of its parent would draw a tree that is not one.
- `--diff` holds both snapshots in memory, so diffing two ends of a large
  recording costs what loading it costs.
- The audit log keeps one rotated generation (`[audit] max_bytes`, 1 MiB),
  so at most about twice that is on disk. `audit::tail` reads from the end of
  the file in 16 KiB chunks, continuing into the rotated file when the
  current one is short.
- `config::save_to` writes a sibling temp file and renames it over the target,
  so a crash mid-save leaves either the old config or the new one. A config
  that cannot be parsed is reported and never written back over.
- The streaming `--remote` path is unit-tested against a controlled child and
  driven end to end through a stand-in `ssh`. `scripts/e2e-ssh.py` also
  exercises real SSH: exports, redaction, hooks, recordings, stale/failure
  handling and recovery, and with `--units` systemd failure and crash-loop
  detection. This passed against the Linux test server (Ubuntu 26.04,
  systemd 259) for 0.18.0, `--units` included;
  it does not establish coverage for every SSH or remote-platform setup.
- Kernel events are matched on message text, because the kernel has no
  structured form of "this is an OOM kill". The patterns are ones that have
  been spelled the same way for many years, and anything unrecognised is
  simply not an event rather than a guess. `/dev/kmsg` is root-only wherever
  `kernel.dmesg_restrict=1`, which is most distributions.
- A `trend` rule keeps about eight points per series across its window. That
  is enough to measure a slope and to survive a missed sample; it is not a
  regression, and a sawtooth that happens to start and end high reads as
  growth. Per-process series are capped at 4096 processes, and only the axes
  (`rss`, `cpu`, `io`) some rule actually names are kept — `TrendTracker`
  clears what it holds when that set changes, because a series whose points
  stopped arriving would otherwise be read as a slope frozen at whatever it
  last was.
- A `trend` rule's CPU series is stored in hundredths of a percent, because a
  process creeping up by a third of a percent a minute is exactly what it is
  for and whole percent would round that series flat. The scaling is undone in
  `measure`, so a threshold is written in whole percentage points.
- `--serve --remote` polls each host's worker on a scrape without waiting
  for outstanding samples. A slow host affects its sample age and stale
  state, not the other hosts' scrape latency.

Open defects, found by review and not yet fixed:

- None outstanding. The four listed here previously — `frame_id` counting
  deliveries rather than measurements, an explicit `[procs] columns` list
  blanking process names on a narrow terminal, the detail pane reading the
  local host under `--remote`/`--replay`, and `MATCHED` colliding with the
  failure exit code — were fixed together with the process-control guard that
  the third of them turned up. The most recent review's four are fixed too:
  the recording flags that did nothing in a headless mode, `secret_eq`
  narrowing its length fold to a `u8`, the environment mask missing `_KEY`-
  suffixed credentials, and `a_process_that_execs_is_reported_as_what_it_became`
  racing its own first sample.
