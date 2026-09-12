# Changelog

Notable changes to crabmon. Versions follow [semantic versioning](https://semver.org),
with the usual 0.x caveat that minor releases may change behaviour.

## 0.6.0 — unreleased

Nothing in the interface blocks any more, the exporter's rates are correct, and
the aggregate view is no longer a dead end.

### Added

- **Sampling on its own thread.** `App::tick()` used to call the metric source
  from the event loop, so anything slow in a sample froze the whole program —
  including the key that quits. A `nvidia-smi` that takes its time, a `statvfs`
  on a wedged NFS mount, or an SSH round trip were each enough. The interface is
  now never slower than a redraw, and says "sampling…" rather than appearing to
  have stopped.
- **Streaming `--remote`.** One SSH session and one process start for the whole
  session instead of one per sample, via a new `--stream` mode on the far end.
  Falls back automatically when the remote crabmon predates it. `[remote]
  stream` switches it off.
- **`--diff A [B]`.** What changed between two snapshots, or between the ends of
  one recording: CPU, load, memory, swap, per-filesystem usage, and the
  processes that started, exited and moved. Processes are matched on PID *and*
  start time, so a recycled PID is not reported as enormous growth.
- **`--watch QUERY`.** Blocks until processes match, prints them and exits 1;
  exits 0 on `--watch-timeout`. `--watch-for` requires the match to hold that
  many consecutive seconds.
- **Pinned processes** (`f`, `F`) — held at the top of the table whatever the
  sort, and visible through a filter that would otherwise hide them.
- **The grouped view is a real view.** Its own cursor and viewport, and `Enter`
  opens a group into the processes inside it, via new `service:` and
  `container:` filter terms.
- **Scrolling popups.** The key list, alert log and action log all held more
  than a short terminal could show and silently dropped the rest; the title now
  says which part is on screen.
- **Alert recovery.** `command_clear` runs when a rule recovers, and `!` logs
  every transition — an alert that fired at 03:12 and cleared at 03:14 says so.
- **Configurable columns** (`[procs] columns`), a **listening-ports column**
  resolved only for the rows on screen, and a marker column so pinned and tagged
  rows are recognisable without colour.
- Exporter series for cgroup limits, GPU memory and temperature, interface
  errors and byte totals, and inode counts.
- `-d`/`--descending`, to override a config file that asks for ascending.
- Listening ports and an open-socket count in the process detail pane.

### Fixed

- **Every rate in `--serve` was wrong** by `scrape_interval / refresh_ms`. The
  exporter divided counter deltas by the *configured* refresh rather than the
  time since the previous scrape: 200 MiB pushed across an 18 s gap at the
  default 800 ms interval reported 250 MiB/s instead of 11.6 MiB/s. Gauges were
  unaffected.
- **One idle client wedged the exporter.** A connection that opened and never
  sent a request blocked the single-threaded accept loop forever. Connections
  are now handled off the accept loop, capped, and timed out in both
  directions.
- **The affinity prompt could abort the process.** `0-4000000000` expanded the
  range before checking it against the CPU count, asking for a 32 GB allocation;
  the resulting abort skips the panic hook and leaves the terminal in raw mode.
  The bound is checked first.
- **`--once` and `--serve` silently ignored `--remote` and `--replay`**, so
  `crabmon --once --remote prod-db` printed the *local* machine's snapshot with
  exit 0. The combination is now refused.
- **A `--remote` host that answered but never finished hung startup.**
  `ConnectTimeout` only bounds connecting; the command itself could run
  forever, and since the first sample is synchronous the interface never
  appeared and there was no `q` to press. One-shot fetches are now bounded and
  killed.
- Process actions could fire from the grouped view, which shows no process rows,
  targeting whichever process the invisible cursor was on.
- The renice and affinity prompts named only the first of several tagged
  processes, despite applying to all of them on one keystroke.
- Tags and pins outlived their processes: the "[N tagged]" counter kept counting
  the dead, and a recycled PID could inherit a pin.
- The selected row was bold-only under `mono` and `--no-color`, which is no
  highlight at all on a terminal that renders bold as plain.

### Changed

- A bare `--serve 9100` now binds to loopback; `:9100` still means every
  interface, and a public bind says so at startup. Per-process series carry
  command lines.
- CSV export carries every field the JSON does — `uid`, `nice`, `service`,
  `container`, `exe`, `cwd` and the start time were all being dropped.
- IPv6 socket addresses are written RFC 5952 style (`::1`, not
  `0:0:0:0:0:0:0:1`).
- `--record` documents that it replaces an existing file rather than appending.

## 0.4.0 — unreleased

Recording, remote monitoring, and the metrics a laptop actually needs.

### Added

- **Record and replay.** `--record run.jsonl` appends every sample; `--replay
  run.jsonl` drives the whole interface from one. `[` `]` step a frame, `{` `}`
  step ten, `z` plays on. Frames are trimmed to keep recordings a usable size.
- **Pressure stall information.** A panel for `/proc/pressure`, showing `some`
  and `full` stall time for CPU, memory and IO — the figure that explains a
  machine that feels frozen while the CPU graph looks idle.
- **Battery and power.** Charge, charge/discharge watts, time remaining, cell
  health and mains state, plus CPU package draw from RAPL where the counters
  are readable.
- **Grouping** (`G`, `--group`) by systemd unit, container or user, totalling
  CPU, memory and disk IO per group.
- **Prometheus exporter.** `--serve :9100` runs headless and serves `/metrics`.
- **Remote monitoring.** `--remote host` drives the interface from another
  machine over SSH, with no daemon and no open port; `--remote-command`
  overrides what runs on the far end.
- **Bulk actions.** `Space` tags processes; signals, renice and affinity then
  apply to every tagged process.
- **Saved filters** on the number keys, configured under `[[filter_preset]]`.
- **Action log.** Every signal, renice and affinity change is recorded locally
  and viewable with `L`.
- **Pause** (`z`) to freeze sampling and read a spike.
- Per-process CPU sparkline and nice columns, open sockets in the detail pane,
  the reclaimable buffers/cache share in the memory gauge, and an inode warning
  for filesystems running out of inodes rather than bytes.
- Clipboard copy (`y`) over OSC 52, which works through SSH and tmux.
- Per-disk IO on FreeBSD, via `iostat -x`. Compile-checked in CI and unit-tested
  against captured output, but not yet exercised on real FreeBSD hardware.

### Fixed

- The PID-reuse guard no longer rebuilds its target set at confirmation time,
  which silently defeated it once bulk actions could outlive a refresh. Targets
  are frozen when the prompt opens.
- Scrubbing a recording advanced two frames per keypress.
- An unreachable `--remote` host showed an empty dashboard with no explanation.

### Changed

- Recordings are trimmed before being written — the busiest 100 processes, argv
  truncated to 200 characters, empty fields omitted — taking a frame from
  ~1.5 MB to ~55 KiB. Trimming prefers real processes over threads, so a replay
  is not filled with one application's threads. All three limits are
  configurable under `[record]`.
- The new metric sources cost about a tenth of a percent of a core: cgroup paths
  and nice values are cached per process rather than read every tick.

## 0.3.0 — unreleased

A rewrite from a single 1071-line file into a library with a thin binary, so
the program can be tested without a terminal.

### Added

- Per-core CPU with a dedicated layout, clock speeds and history.
- A process detail pane, tree view, per-process disk IO, and thread hiding.
- Signal, renice and CPU affinity control, each guarded against PID reuse.
- A process filter language (`user:`, `pid:`, `cpu>`, `mem>`, `re:`, `!`).
- Sorting on every column, by key or by clicking the header; mouse support.
- Network, disk and sensor panels reworked; GPU and cgroup panels added.
- Threshold alerts with a hold time and a command hook.
- Snapshot export as JSON or CSV, and headless `--once` with `--format` and
  `--top`.
- A full command line — `--refresh`, `--sort`, `--ascending`, `--filter`,
  `--tree`, `--layout`, `--theme`, `--no-color`, `--no-mouse` and `--config` —
  five themes, four layouts, and a config file covering every panel, threshold
  and colour.
- A man page, shell completions, and CI across Linux, macOS and Windows.

### Fixed

- **Memory was reported 1024× too large.** `sysinfo` 0.30 returns bytes, not
  KiB, so a 16 GB machine displayed "15292.6 GiB".
- **Disk IO read as zero on LVM and LUKS volumes.** `/dev/mapper/…` is a
  symlink; only the resolved `dm-N` device appears in `/proc/diskstats`.
- The aggregate network rate counted loopback, bridges and VPN tunnels, so
  tunnelled traffic was counted twice.
- Hot-plugged disks, interfaces and sensors never appeared, because only known
  entries were being refreshed.
- The refresh interval could be set below the sampler's 200 ms minimum, making
  CPU percentages meaningless.
- Ctrl-C did nothing, and Ctrl-Q quit.
- A panic left the terminal in raw mode on the alternate screen.
- Closing the terminal left crabmon spinning at 100% CPU, ignoring SIGTERM.
- Per-core temperature sensors crowded every other sensor off the panel.
- Scrolling up moved the viewport instead of the cursor.
- Configuration was lost unless the session ended with `q`.

## 0.2.2

The proof-of-concept: CPU, memory and swap, a sortable process list, network
rates, per-disk usage, temperature sensors, and a Debian package.
