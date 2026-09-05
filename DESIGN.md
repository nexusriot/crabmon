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
}
```

`SysinfoSource` implements it against the real host, `ReplaySource` against a
recording, `RemoteSource` against another machine over SSH, and the test-suite
against canned frames. That last one is what lets `tests/render.rs` assert on a
real rendered screen and `tests/app_behaviour.rs` drive the full interaction
model with no terminal, no `/proc`, and no timing flakiness.

The four defaulted methods exist for finite sources. `timeline` and `seek` make
the scrub keys work; `peek` reads the current frame *without* advancing, which
matters because calling `snapshot` after a `seek` would step straight past the
frame just sought. Live sources inherit the defaults and the scrub keys are
inert for them.

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

Over 320 tests: the in-crate unit tests plus five integration suites. All
are offline and deterministic except `proc_control`, which deliberately touches
the kernel:

- **unit tests** (`src/**`) — formatting, filter parsing, sorting, tree
  flattening, diskstats/cgroup/PSI/meminfo/`iostat`/`nvidia-smi` parsing, GPU and
  battery sysfs against fabricated trees, socket-table decoding, alert state,
  column planning, base64, the Prometheus exposition format, recording
  round-trips and config round-trips.
- **`tests/app_behaviour.rs`** — the interaction model against a fixture source:
  key handling in every mode, selection stability across re-sorts, PID-reuse
  refusal for all three process actions, tagging and bulk actions, saved
  filters, grouping, replay scrubbing, audit logging, mouse handling and
  history bounds.
- **`tests/render.rs`** — the real panels drawn onto a `TestBackend`, asserting
  on screen content, at sizes from 10×3 up to 300×100 and in every theme.
- **`tests/cli.rs`** — the built binary: `--help`, `--version`, exit codes, and
  `--once` JSON/CSV output.
- **`tests/proc_control.rs`** — renice, CPU affinity and signals against a real
  spawned child, checked by reading the state back from the kernel. A unit test
  can only prove the arguments were well-formed, not that they were accepted.
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

`scripts/build-deb.sh` builds a release binary and runs `cargo-deb`, which
installs the binary, the man page and bash/zsh/fish completions. CI runs the
test-suite on Linux, macOS and Windows, checks formatting and clippy, and
cross-checks the cfg-gated code paths against aarch64 Linux and FreeBSD.

## The exporter

`--serve` is a hand-rolled `TcpListener` loop rather than a framework, because
the whole contract is one method and three routes. It samples per scrape, so the
scrape interval is the sample interval. Per-process series are capped (default
20) — one series per process on a 2000-task machine would dwarf every other
metric in the scrape and make the exporter the most expensive thing on the box.

## Known limitations

- Process CPU percentages are relative to a single core, as `sysinfo` reports
  them, so a busy multi-threaded process can exceed 100%.
- `nvidia-smi` polling is opt-in and synchronous; it will stall a refresh on a
  machine where the driver is slow to answer.
- The FreeBSD disk-IO path shells out to `iostat -x` per tick and reports
  since-boot averages rather than instantaneous rates. It is compile-checked in
  CI and unit-tested against captured output, but has never run on real FreeBSD
  hardware.
- `--remote` re-runs `crabmon --once` over SSH for every sample, so the refresh
  rate is bounded by the round trip and the far end pays a full startup each
  time. A streaming mode would be better and is not implemented.
- The process list is rebuilt and re-sorted on every refresh rather than
  incrementally maintained. At ~2000 processes and an 800 ms interval this costs
  a low double-digit percentage of one core, which is in line with comparable
  monitors but is the obvious next thing to optimise.
