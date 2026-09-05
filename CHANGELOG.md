# Changelog

Notable changes to crabmon. Versions follow [semantic versioning](https://semver.org),
with the usual 0.x caveat that minor releases may change behaviour.

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
- **Grouping** (`G`) by systemd unit, container or user, totalling CPU, memory
  and disk IO per group.
- **Prometheus exporter.** `--serve :9100` runs headless and serves `/metrics`.
- **Remote monitoring.** `--remote host` drives the interface from another
  machine over SSH, with no daemon and no open port.
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
- Snapshot export as JSON or CSV, and headless `--once`.
- A full command line, five themes, four layouts, and a config file covering
  every panel, threshold and colour.
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
