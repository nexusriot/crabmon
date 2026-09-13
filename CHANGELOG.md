# Changelog

Notable changes to crabmon. Versions follow [semantic versioning](https://semver.org),
with the usual 0.x caveat that minor releases may change behaviour.

## 0.7.0 — unreleased

Building, testing and installing crabmon is one command each, and the tooling
that does it is checked by the suite like everything else.

### Added

- **A `Makefile` and `scripts/build.sh`** covering `build`, `release`, `bin`,
  `run`, `test`, `test-unit`, `check`, `fmt`, `lint`, `ci`, `deb`, `dist`,
  `install`/`uninstall` under a `PREFIX` (honouring `DESTDIR`) and `clean`. The
  Makefile is a front end to the script — one line per target — rather than a
  second implementation of it, so `make test` and `./scripts/build.sh test`
  cannot come to mean different things, and every command works on a machine
  with no make.
- **`tests/build_script.rs`**, which fails the build if the two drift apart: a
  `make` target with nothing behind it, a command make cannot reach, or one
  missing from the script's own help. It also runs the script, which is the only
  way to catch a syntax error or a dispatcher that resolves nothing before
  someone needs a build.
- **Thirty unit tests** over the gaps a `pub fn`-by-`pub fn` audit turned up:
  `group_rows` and the selection and popup helpers in `app.rs`, which had no
  in-crate tests at all; the recorder's configured limits actually reaching the
  frames on disk; `$USER` expansion in saved filters; inode starvation, absent
  clock readings and swapless machines in the metric model; battery charge
  direction; popup scroll titles and audit timestamps.

### Fixed

- **The Prometheus exporter was rejected outright by Prometheus.** The PSI loop
  emitted a fresh `# HELP`/`# TYPE` pair per resource, so
  `crabmon_pressure_some_ratio` was declared three times — and a repeated HELP
  makes Prometheus abort the *whole* scrape, so none of the other ~40 series
  were ingested either. On any host with `/proc/pressure`, i.e. every modern
  Linux. The resources are now labelled samples under one declaration.
- **`--serve` exported an arbitrary 20 processes, never the busiest.**
  `snap.procs` arrives in the sampler's hash order and nothing sorted it, so on
  a 2400-process machine every exported series was an idle kernel thread.
  `--once` and the recorder both sort before truncating; now this does too.
- **Pausing, or scrubbing a replay, pegged a core.** `last_tick` only advances
  when a tick actually runs, so while paused the countdown to the next refresh
  stayed expired and the poll timeout collapsed to zero — a redraw spin
  measured at 100% CPU and 4.4 MB/s of escape sequences, in a system monitor.
  `App::scrub` pauses too, so this was replay mode's primary interaction.
- **A malformed `--filter` matched everything instead of failing.**
  `--once --filter 'cpu>'` printed all 2515 processes with exit 0, while
  `--watch 'cpu>'` rejected the identical string. The parser now validates it,
  and `--once` propagates a bad filter from the config file rather than
  degrading to the empty filter.
- **`--remote host --watch …` watched the local machine.** `--watch` was the
  one host-sampling mode missing from the `--remote`/`--replay` refusal — and
  the one whose exit code drives `&&` in scripts. README and `main.rs` both
  already documented the guarantee.
- **A value-taking flag swallowed the next flag.** `--serve --stream` read
  `--stream` as the bind address, so the exclusivity check never saw it and the
  run died later at bind time; `--filter --once` started the TUI. Only
  crabmon's own long flags are refused, so dash-leading values still work.
- **`service:-` matched most of the machine.** The "no unit" sentinel fell
  through to a substring match, so drilling into the grouped view's `-` row
  listed `systemd-journald.service`, `user-1000.slice` and every other
  hyphenated unit.
- **Package power was roughly doubled on machines exposing `psys`.** RAPL
  summed every top-level domain, but `psys` meters the whole platform and
  already contains `package-0`. The domain's `name` file now decides. The test
  that covered this had labelled `intel-rapl:1` "second package", certifying
  the double-count.
- **A wireless mouse could appear as the machine's battery.** Peripherals
  publish `type=Battery` too; `scope = Device` now excludes them, so a desktop
  with a Logitech receiver no longer grows a Power panel.
- **`MAX_FDS_SCANNED` bounded nothing.** The cap was applied to the finished
  Vec, after every fd had already been `read_link`ed, so a process holding
  200k sockets cost 200k syscalls per sample — in the render path.
- **The interactive filter is no longer persisted.** A `/` query, or a one-off
  `--filter`, was written into the config on exit, after which every later
  `crabmon --once` printed an empty process list with exit 0 and the TUI opened
  on an empty table with no visible cause.
- **`[procs] ports` did nothing.** It is documented in the README and the man
  page as the switch that turns the listening-ports lookup on, but nothing in
  the crate ever read it — the column was gated solely on `[procs] columns`.
  It now adds the PORTS column to the automatic set, and having been asked for,
  the column outranks the other optional ones for the width available.
- **The `--watch` shell example was inverted.** Both the README and the man
  page said "`&&` runs on a match and `||` on a clean run". A match exits 1, so
  it is the other way round, and the shipped `… && notify-send 'stuck IO'`
  fired exactly when nothing was stuck.
- The man page's EXIT STATUS listed only 0 and 2, never mentioning 1 — which is
  the code `--watch` exists to return.
- DESIGN.md's limitations contradicted themselves about `--remote`: one entry
  said a streaming mode "is not implemented" while another discussed testing
  the streaming path. Streaming has been the default since 0.6.0.
- Three panics reachable from ordinary input: a `nan` threshold in the config
  file tripped `clamp`'s `min <= max` assertion and aborted every mode at
  startup; a non-ASCII `[colors]` value passed a byte-length guard and then
  sliced across a char boundary; and `centered_rect` overflowed `u16` from
  about 860 columns up, collapsing popups to a sliver on wide terminals. An
  empty `ReplaySource` also underflowed on first use.

### Changed

- `scripts/build-deb.sh` still works — CI and the README have always used that
  name — but it now delegates to `build.sh deb` rather than being a second copy
  of the packaging steps.
- `make install` installs with `mkdir -p` and `install -m` rather than
  `install -D`, which is a GNU extension the BSDs do not have; and it always
  rebuilds first, so it cannot ship a stale binary from an earlier checkout.

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
