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

## Crate layout

The application is a **library** with a thin binary on top. That split exists
for one reason: it makes the whole program testable headlessly.

```
src/
  main.rs            terminal setup/teardown, signals, panic hook, event loop
  lib.rs             module tree and the shared refresh bounds
  cli.rs             argument parsing (hand-rolled, ~200 lines)
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
  record.rs          JSONL recording and the replay source
  remote.rs          monitoring another host over SSH
  sampler.rs         sampling on a worker thread
  diff.rs            comparing two snapshots
  watch.rs           the headless wait-for-a-condition mode
  serve.rs           the Prometheus exporter
  audit.rs           the local log of actions taken
  clipboard.rs       OSC 52 copy
  metrics/
    mod.rs           the data model and the `MetricSource` trait
    sysinfo_source.rs the live host implementation
    diskstats.rs     /proc/diskstats and device-name resolution
    netclass.rs      virtual-interface classification
    gpu.rs           /sys/class/drm and nvidia-smi
    cgroup.rs        cgroup v1/v2 limits
    psi.rs           /proc/pressure stall information
    power.rs         batteries, mains and RAPL
    meminfo.rs       the buffers/cache breakdown
    procgroup.rs     per-process cgroup paths, and what they mean
    sockets.rs       /proc/net joined against a process's fd table
  ui/
    mod.rs           layout selection and the draw entry point
    header.rs        host banner and status line
    cpu.rs           average chart, per-core strip, per-core detail
    mem.rs           memory, swap and cgroup gauges
    procs.rs         the process table and the grouped view
    psi.rs           pressure gauges
    power.rs         battery and package power
    net.rs           interface table and RX/TX chart
    disks.rs         per-mount gauges
    sensors.rs       grouped temperatures
    gpu.rs           GPU rows
    popups.rs        signal, confirm, renice, affinity, detail, help, alerts
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

`Snapshot` is plain data (`HostInfo`, `CpuSample`, `MemSample`, `ProcRow`,
`NetIface`, `DiskRow`, `Sensor`, `GpuInfo`, `CgroupInfo`) and derives
`Serialize`, so JSON export is the same type the UI reads rather than a parallel
representation that can drift.

## State: `App`

`App` holds the config, the theme, the current `Snapshot`, the derived process
view, the history buffers, the alert engine and the interaction mode. It owns no
terminal; its only IO is the metric source, the config file and writing an
export when the user asks for one.

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
| `Detail` | `Enter` | process detail |
| `Help` | `?` | key bindings |
| `Alerts` | `!` | rule list and state |
| `AuditLog` | `L` | the local record of actions taken |

Sending a signal is two-step by construction: `SignalMenu` → `ConfirmKill`.
Every action that targets a process — signal, renice and affinity alike —
re-checks that the PID still has the start time it had when the prompt opened,
so a recycled PID is refused rather than acted on.

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

- **Memory** — `sysinfo` 0.30 reports **bytes**, not KiB. Treating those bytes
  as KiB is what made a 16 GB machine display "15292.6 GiB"; `format::human_bytes`
  now does the scaling and a test pins the exact expected string.
- **Device lists** — `Networks::refresh()` and friends only update entries
  already in the list. `refresh_list()` is what discovers hot-plugged disks, new
  containers and fresh VPN interfaces, so the source re-lists every 5 seconds.
- **Disk IO** — `/proc/diskstats` is keyed by kernel device name, but a
  filesystem's device may be `/dev/mapper/root_crypt`. `diskstats::resolve_device_key`
  canonicalises the symlink (`→ /dev/dm-0`) and takes the basename, which is why
  IO rates work on LVM and LUKS roots. Sectors are 512 bytes regardless of the
  device's logical block size.
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
- **Per-process IO** — differences of `Process::disk_usage()` totals. Reading
  `/proc/<pid>/io` for another user's process is not permitted, so those rates
  legitimately read as zero.
- **GPU** — amdgpu exposes `gpu_busy_percent` and VRAM counters; i915/xe expose
  clocks but no busy counter, so the clock ratio is shown prefixed with `~` to
  make clear it is not utilisation. NVIDIA needs `nvidia-smi`, which is opt-in
  because it costs tens of milliseconds per refresh.
- **cgroups** — the panel appears only when the process is actually confined,
  since a limitless root cgroup is noise.

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
with nothing to show for it.

Falling back to the old behaviour is deliberately narrow: only the far end's
argument parser refusing `--stream` triggers it. Falling back on any failure
would paper over a genuinely broken host, and an explicit `--remote-command` is
never second-guessed at all.

The one-shot path is bounded by `FETCH_TIMEOUT`. `ConnectTimeout` only covers
setting the connection up; once SSH is connected, a command that never returns
hangs the fetch, and because the first sample is synchronous that meant the
interface never appeared and there was nothing to press `q` in. The wait drains
both pipes on threads, because a child that fills one and blocks never exits —
polling `try_wait` alone cannot tell "slow" from "deadlocked on a full pipe".

## Layout

`ui::draw` renders a header, one of four layouts, and a status line, then any
popup on top.

| Layout | Content |
| --- | --- |
| `Dashboard` | CPU · memory · swap · processes, with a configurable right column |
| `Processes` | the process table full-width |
| `Cpu` | one bar per core, in as many columns as fit, plus the average chart |
| `Io` | network chart and table, disks, GPU |

The right-hand column's contents and order come from `panel_order`, and every
panel can be switched off in `[panels]`.

The process table's columns are chosen to fit the available width: PID, CPU and
memory plus the command are always present, and state, user, disk, time, virtual
size and thread count are added in that order as the terminal gets wider. A
property test checks the chosen set fits at every width from 20 to 200 columns.

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

Over 450 tests: the in-crate unit tests plus six integration suites. All
are offline and deterministic except `proc_control`, which deliberately touches
the kernel:

- **unit tests** (`src/**`) — formatting, filter parsing, sorting, tree
  flattening, diskstats/cgroup/PSI/meminfo/`iostat`/`nvidia-smi` parsing, GPU and
  battery sysfs against fabricated trees, socket-table decoding, alert state,
  column planning, base64, the Prometheus exposition format, recording
  round-trips and config round-trips.
- **`tests/app_behaviour.rs`** — the interaction model against a fixture source:
  key handling in every mode, selection stability across re-sorts, PID-reuse
  refusal for all three process actions, tagging and bulk actions, pinning,
  saved filters, grouping and group drill-down, popup scrolling, replay
  scrubbing, audit logging, mouse handling, history bounds, and that a repeated
  frame from an asynchronous source never reaches the charts.
- **`tests/render.rs`** — the real panels drawn onto a `TestBackend`, asserting
  on screen content, at sizes from 10×3 up to 300×100 and in every theme.
- **`tests/cli.rs`** — the built binary: `--help`, `--version`, exit codes,
  `--once` JSON/CSV output, `--diff` in both formats, `--stream` read back
  through the recording parser, `--watch`'s exit codes, and the refusal to
  combine a headless mode with `--remote` or `--replay`.
- **`tests/proc_control.rs`** — renice, CPU affinity and signals against a real
  spawned child, checked by reading the state back from the kernel. A unit test
  can only prove the arguments were well-formed, not that they were accepted.
- **`tests/build_script.rs`** — the build tooling against itself: that every
  `make` target has a command behind it, that every command is reachable through
  make and described in the script's own help, and that the script runs, reports
  the crate's version and fails with a status on a mistyped command. `make` is a
  front end to `scripts/build.sh`, and a front end that has drifted from what it
  fronts is found when you need the command, not before.
- **`tests/docs.rs`** — that the man page, completions and README still describe
  the flags, sort keys, themes, layouts, groupings, panels, config fields and key
  bindings the code actually has; that the aliases documented in one place are
  documented in the other; that the packaged description covers the current
  feature set; and that the man page's version matches the crate's.
  Documentation drift is a build failure, not a review comment.

  These checks derive their expectations from the code — a serialised default
  `Config`, `cli::LONG_FLAGS`, `ALL_SORTS`, `PRESETS`, `ALL_GROUPINGS`, the
  `KEYS` table, and the key literals parsed out of `on_normal_key` — rather than
  from a hand-maintained list that could go stale in the same way the docs do.
  They are also scoped to the relevant section of each document, so an
  incidental mention of a word elsewhere cannot satisfy them.

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
A pair of tests in `tests/docs.rs` assert that the two lists agree and that the
script's own help covers every command it accepts.

`make deb` (still reachable as `scripts/build-deb.sh`, the name CI uses) builds
a release binary and runs `cargo-deb`, which installs the binary, the man page
and bash/zsh/fish completions. `make dist` produces the same file set as a
tarball for the platforms the deb does not cover. CI runs the test-suite on
Linux, macOS and Windows, checks formatting and clippy, and cross-checks the
cfg-gated code paths against aarch64 Linux and FreeBSD.

## The exporter

`--serve` is a hand-rolled `TcpListener` loop rather than a framework, because
the whole contract is one method and four routes (`/`, `/metrics`, `/healthz`,
and 404 for everything else). It samples once per scrape, on the accepting
thread. Per-process series are capped (default 20) — one series per process on
a 2000-task machine would dwarf every other metric in the scrape and make the
exporter the most expensive thing on the box.

It always samples the host it runs on, as do `--once`, `--stream` and `--watch`.
Combining any of them with `--replay` or `--remote` is refused rather than
ignored: `crabmon --once --remote prod-db` used to print the *local* machine's
snapshot with exit 0, which is a worse failure than an error, because a script
gets plausible data about the wrong host and never finds out.

`serve` measures the gap between scrapes and passes it to the source, because
rates are deltas of monotonic counters divided by the interval they are handed.
Passing the *configured* refresh instead — which is what it used to do —
multiplied every rate series by `scrape_interval / refresh_ms`: 200 MiB pushed
over loopback across an 18 s gap at the default 800 ms interval reported
250 MiB/s rather than 11.6 MiB/s. Gauges carry no delta and were unaffected,
which is what made it easy to miss.

Each connection is handled on its own thread, capped at `MAX_CONNECTIONS` with
a 503 past it, and both socket directions carry a timeout. Doing the socket IO
on the accept loop meant a client that connected and never sent a request — a
stalled load balancer health check will do — stopped the exporter answering
anybody, with no error anywhere; a timeout alone only bounds how long that
lasts. Sampling stays behind a mutex, so concurrent scrapes see consistent
counter deltas instead of racing each other's `prev` state.

A bare `--serve 9100` binds to loopback; `:9100` keeps its conventional "every
interface" meaning and says so at startup. The per-process series name every
running command and its arguments, and the shortest spelling should not be the
one that publishes them.

## Known limitations

- Process CPU percentages are relative to a single core, as `sysinfo` reports
  them, so a busy multi-threaded process can exceed 100%.
- `nvidia-smi` polling is opt-in and synchronous; it will stall a refresh on a
  machine where the driver is slow to answer.
- The FreeBSD disk-IO path shells out to `iostat -x` per tick and reports
  since-boot averages rather than instantaneous rates. It is compile-checked in
  CI and unit-tested against captured output, but has never run on real FreeBSD
  hardware.
- `--remote` streams by default (`[remote] stream`, on): one SSH session and
  one `crabmon --stream` for the whole session. The one-shot fallback, taken
  when the far end is too old to understand the flag, does re-run
  `crabmon --once` per sample, and there the refresh rate is bounded by the
  round trip and the far end pays a full startup each time.
- The process list is rebuilt and re-sorted on every refresh rather than
  incrementally maintained. At ~2000 processes and an 800 ms interval this costs
  a low double-digit percentage of one core, which is in line with comparable
  monitors but is the obvious next thing to optimise.
- A replay is parsed in full into memory before the first frame is drawn, so a
  long capture costs roughly its JSON size several times over. Nothing indexes
  the file by line offset.
- The `ports` column walks one fd table per visible row, which is why it is off
  by default — `[procs] ports`, or naming it in `[procs] columns`, is what asks
  for it — and resolved only for rows actually on screen. A process holding tens
  of thousands of descriptors is cut off at `MAX_FDS_SCANNED`, which bounds the
  walk itself rather than the list it returns.
- Pins do not reorder the tree view. The ordering there is structural, and
  hoisting a child out of its parent would draw a tree that is not one.
- `--diff` holds both snapshots in memory, so diffing two ends of a large
  recording costs what loading it costs.
- The audit log grows without bound and `audit::tail` reads the whole file to
  show its last 200 lines.
- `config::save_to` is a plain write rather than write-and-rename, so a crash
  during the save that `+`/`-` triggers can truncate the config. It falls back
  to defaults on the next start, which limits the damage to the settings.
- The streaming `--remote` path is unit-tested against a controlled child and
  driven end to end through a stand-in `ssh`, but has not been run over a real
  SSH connection; the same caveat the FreeBSD `iostat` path carries.

Open defects, found by review and not yet fixed:

- `ThreadedSource::frame_id` counts frames *delivered*, not distinct samples,
  and does not consult the source it wraps. Since `RemoteSource` re-hands its
  last good frame when the far end stalls, an outage is charted as flat,
  real-looking measurement rather than a gap, and `is_stale()` never raises the
  `sampling…` notice. `ReplaySource` has the same shape from the other end: it
  holds the final frame forever while reporting `frame_id() == None`, so a
  finished replay keeps feeding copies into the history until the recorded
  incident scrolls out of it.
- An explicit `[procs] columns` list is never width-checked, so on a narrow
  terminal `name_width` can reach zero and every process name renders blank.
  The automatic planner reserves `MIN_NAME`; the configured path does not.
- Alert rule state is keyed by `rule.name` and nothing enforces that names are
  unique. Two rules sharing one — one over threshold, one under — clear each
  other's `fired` flag, so the hook command re-spawns every refresh.
- The detail popup reads the *local* host for the nice value and socket list
  rather than the snapshot, so under `--replay` they are missing and under
  `--remote` they belong to whatever local process holds that PID. The same
  applies to the `ports` column.
- Aggregate disk throughput is summed per mount, so a device carrying several
  mounts — the normal btrfs subvolume layout — is counted once per mount.
- `MATCHED = 1` is also the exit code a failed run returns, so a `--watch`
  caller cannot distinguish a match from a typo in the query.
