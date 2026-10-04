# Changelog

Notable changes to crabmon. Versions follow [semantic versioning](https://semver.org),
with the usual 0.x caveat that minor releases may change behaviour.

## 0.9.0 — unreleased

Everything crabmon measures, reachable from outside the full-screen interface:
alert rules that run headlessly, a watch that waits on any of them, several
hosts in one table, and the per-container network figure the kernel will
actually account. Plus the bugs an end-to-end suite found once there was one.

### Added

- **Alerts without a terminal: `--serve --alerts` and `--stream --alerts`.**
  The rules, their hold times, their hooks and the flight recorder all ran
  only inside the TUI, which made the flight recorder — whose whole point is
  to have a recording of an incident nobody predicted — depend on someone
  sitting in front of a full-screen interface at the moment the incident
  happened. Under `--serve` each rule is also exported as
  `crabmon_alert_active`, `crabmon_alert_value` and `crabmon_alert_threshold`,
  firing or not, which is what finally makes crabmon's own measurements —
  pressure stall, device utilisation, a process against *its own* descriptor
  limit — reachable from a pager. The TUI and the headless modes run the same
  code: an incident is handled identically whether or not anyone is watching.
- **`--watch-rule`: wait for a measurement, not just a process list.**
  `cpu>=90`, `psi:io>20`, `disk:/var>=95`, `net>=1M`, `proc:nginx<1` — any
  `[[alert]]` kind, spelled compactly, with `--watch-for` as its hold time.
  Both forms of `--watch` are now alert rules underneath (a process watch is
  `proc:<query>>=1`), which deletes the second hold-time state machine this
  mode used to carry and makes a condition behave the same in a shell script,
  in the config file and under `--serve --alerts`.
- **Several hosts at once: `--remote web-1,web-2,db-1`.** A fleet table with
  one row per host — CPU, memory, swap, load, disk utilisation, network,
  processes — so the machine in trouble is visible before you have picked one.
  `Enter` points the rest of the interface at the host under the cursor, `W`
  goes back. Hosts are sampled concurrently, one worker each, so an
  unreachable machine is one row saying why next to the last numbers it
  reported, rather than a blank dashboard. A single target behaves exactly as
  before, with no fleet table and no extra stop in the `Tab` cycle.
- **Per-container network throughput.** Per-*process* network bytes is the
  thing a process monitor is asked for most and cannot honestly provide: the
  kernel does not account bytes to a PID, and getting there means eBPF or
  packet capture, which need privileges crabmon has promised not to ask for.
  What the kernel does account is bytes per network *namespace*, which is
  per container — so the grouped view gains a NET column, and `--serve`
  exports `crabmon_container_receive_bytes_per_second` and friends. The host's
  own namespace is excluded, because the per-interface series already carry it.
- **Open files in the process detail pane.** The fd walk was already there to
  count descriptors; resolving the links turns it into `lsof` for one process,
  next to the sockets the pane already lists. "Which log is filling the disk"
  and "what is this wedged process still holding" used to mean leaving
  crabmon, which loses the process you were looking at.
- **`--columns`**, to choose the process table's columns for one run without
  editing `[procs] columns`. An unknown name is an error that lists the real
  ones, rather than a column quietly missing from the table.
- **A hermetic end-to-end suite that runs in a container**
  (`tests/e2e_docker.rs`, `make e2e`). The binary is built from the checkout by
  a Dockerfile whose bases are pinned by digest, and exercised inside a
  container with an empty network namespace, a read-only image, a tmpfs for
  scratch and a fixed hostname — so assertions are about a machine whose
  contents are known, rather than about whatever the developer's own `/proc`,
  network or leftover files happen to hold. Two of the tests assert that
  sandbox itself, because a sandbox nothing checks is one that quietly stops
  being a sandbox. Nothing is installed into the image: the exporter is scraped
  over the container's own loopback by a small bash client, where an
  `apt-get install curl` would have made every run depend on whatever a Debian
  mirror held that day. It is what found the `exec` bug above. Opt-in: without
  `CRABMON_E2E=1` every test in it reports a skip and passes, so a machine with
  no Docker still runs a green suite.

  `make e2e` needs Docker and nothing else, not even Rust. The harness is
  itself a `cargo test` suite, so a toolchain has to exist somewhere; it comes
  from a container too, with the checkout mounted read-only so the run leaves
  nothing behind. `make e2e-local` is the same tests driven by the toolchain
  already installed — faster to iterate in, and it hands nothing the Docker
  socket. Both are CI jobs, the first with no toolchain installed at all.
- **`--layout fleet`**, and `ui::ALL_LAYOUTS` so the docs tests check the real
  set rather than a copy of it. The same gap had let `W` be bound in the TUI
  without ever appearing in the `?` key list; there is now a test for that
  direction too.

### Fixed

- **A process that `exec`ed kept the name it had beforehand, for ever.** `exec`
  replaces the program without changing the PID, the parent or the start time,
  and the sampler reads a process's name only when it first meets a PID and
  treats an unchanged start time as "nothing to re-read". So a wrapper script
  that hands off to the real program — every container entrypoint, every unit
  with a shell in front of it, `sudo`, `nohup` — was reported as `sh` for as
  long as it ran, with the wrapper's argv, executable and working directory
  alongside it. The process table, `service:` grouping, `cmd:` filters and
  `proc` alerts were all looking at a program that was no longer running, and
  `--watch nginx` could never match an nginx that had been exec'd into.
  `/proc/<pid>/comm` is now checked every sample — one small read per process,
  around 10 ms on a 2000-process host — and the heavier re-reads happen only
  for the processes it shows have actually changed.
- **`--once`, `--stream` and `--watch` divided every rate by the interval they
  asked for rather than the one that passed.** Each slept the whole refresh
  interval and *then* sampled, so a frame took `interval + sample_cost` while
  the counter deltas in it were still divided by `interval` — every
  bytes-per-second series overstated by that ratio, which is a few percent on
  an idle laptop and more than double on a host where a sample costs more than
  a refresh. `--serve` already measured the real gap between scrapes; the other
  three now do too, and sleep only the remainder of the interval, so the
  cadence is the configured one.
- **Two identical GPUs made the Prometheus exporter return nothing at all.** A
  card's name is `"<vendor> <driver>"` from sysfs, or whatever `nvidia-smi`
  calls the model, so a matched pair produced the same `gpu="..."` label twice
  in one metric family — and Prometheus rejects the *entire* scrape over a
  duplicate series, exactly as it does over a repeated `HELP` line. Repeated
  label sets are now dropped rather than taking every other metric with them.
  Two NVMe drives presenting the same sensor label did the same thing.
- **A dropped `--remote` stream ran the streaming command as a one-shot.** When
  a streaming SSH session died for any reason other than an old crabmon at the
  far end, the next sample fell through to the single-frame path — which still
  held `crabmon --stream`, a command that by definition never exits. Every tick
  after the drop paid the full 15-second fetch timeout and left another crabmon
  running on the remote host, and the real failure was overwritten by whatever
  that second `ssh` said. The session is simply reopened on the next tick now.
- **The Prometheus exporter read a request line without a bound.** The read
  timeout limits how long a single read may block, which a client that keeps
  sending never trips, so one connection could make the exporter buffer for as
  long as it cared to type. Request lines are capped at 8 KiB.
- **The FreeBSD cross-check did not compile.** `setpriority` and
  `getpriority` take `id_t` for the process on Linux and `int` on the BSDs, so
  passing a `u32` built on one and not the other. `pid as _` lets each target
  pick its own.
- **An unknown name in `panel_order` silently dropped that panel.** The list
  went straight into the draw loop, where a name no arm matched fell through
  and did nothing — so a typo, or a panel renamed between releases, cost you
  the panel with nothing anywhere to say why. `[procs] columns` has always
  been validated; `panel_order` now is too, including the duplicate that would
  otherwise split the column between two copies of the same panel.
- **A negative size in a filter query silently became zero.** `mem>-1M` parsed
  as `mem>0`, a filter that matches the whole process table; `parse_duration`
  had always refused the same mistake. Sizes are now rejected the same way.

## 0.8.0 — unreleased

Six things the existing machinery was one step short of: disks that say when
they are saturated rather than only when they are full, alerts on the numbers
crabmon already measured, a filter language that can name every column the
table sorts on, recordings of incidents nobody predicted, per-process history,
and the descriptor count that explains a daemon failing every `accept` while
all its other figures look ordinary.

### Added

- **Disk saturation and latency.** `/proc/diskstats` fields 12–14 — in-flight
  requests, `io_ticks` and service times — were parsed away and discarded;
  they now drive a `% busy` and mean request latency on each disk gauge, in the
  panel title for the busiest device, in `--once`/`--stream`/recordings, and as
  `crabmon_disk_utilisation_ratio` and `crabmon_disk_request_seconds`. This is
  the figure a throughput gauge cannot carry: an NVMe serving 4K random reads
  is pinned at a few MB/s, and the rates draw that as very nearly idle. FreeBSD
  gets the same numbers from `iostat -x`'s `%b` and `ms/t` columns.
- **Five more alert kinds.** `psi` (targeting `cpu`, `mem` or `io`, optionally
  `.full`), `io` for device saturation, `net` for aggregate throughput, `fd`
  for descriptor exhaustion, and `proc`, which counts the processes matching a
  filter query. Pressure-stall was the one crabmon called "the number that
  explains a machine that feels frozen while the CPU graph looks idle" and then
  offered no way to alert on.
- **`below` on an alert rule**, firing when the measurement is strictly under
  the threshold. With `kind = "proc"` this is process-*absence* alerting —
  `query = "nginx"`, `threshold = 1`, `below = true` pages you when nginx dies
  and clears when it comes back — which is the condition anyone actually wants
  to be woken for, and which needed all three pieces to already exist.
- **`OR` in the filter language.** A standalone `|` between terms, binding
  looser than the implicit AND, so `nginx cpu>10 | apache` is
  `(nginx AND cpu>10) OR apache`. Only a `|` with whitespace on both sides is
  an operator, so `re:^chrom(e|ium)$` is still one regular expression.
- **The columns the table sorts on can now all be filtered on**: `virt>`,
  `thr>`, `nice<`, `time>` (in `s`/`m`/`h`/`d`) and `fd>`/`fd%>`. `--sort nice`
  worked and `nice<0` did not, in a query language whose whole job is to narrow
  the same table.
- **A flight recorder.** `[record] flight` keeps that many trimmed frames in
  memory and writes them out, plus `flight_after` more, whenever an alert
  fires — so the recording of an incident exists because the incident happened
  rather than because someone predicted it. The dump is an ordinary recording
  that `--replay` and `--diff` read. A flapping rule extends the dump in
  progress instead of leaving a file per refresh.
- **Per-process history.** The 24-sample CPU buffer behind the TREND column is
  now 120 samples of CPU, resident memory and disk IO, and `Enter` charts all
  three with the peak each is scaled against. One sample cannot distinguish a
  leak from a process that was always big, and leaving the program to find out
  loses the process.
- **File descriptors.** `[procs] fds` counts each process's open descriptors
  and reads its `RLIMIT_NOFILE`, giving an FD column that switches to a
  percentage once the process is close to its limit, a `fds` sort key, `fd>`
  and `fd%>` filters, a `kind = "fd"` alert, `fds`/`fd_limit` CSV columns and
  `crabmon_process_open_fds`/`crabmon_process_max_fds`. 4000 descriptors is
  unremarkable against a limit of a million and fatal against the default 1024,
  so the limit is what is actually reported against. Off by default: unlike
  PORTS it is sampled for every process, because a filter that saw only the
  rows on screen would be a filter over the wrong set.

### Fixed

- **Aggregate disk throughput counted a device once per mount.** The ordinary
  btrfs subvolume layout has four or five mounts on one block device, all
  reporting the same kernel counters, so the dashboard multiplied the machine's
  whole disk throughput by however many subvolumes happened to be mounted.
  Totals are now taken per device.
- **Two alert rules sharing a name cleared each other's state.** Rule state was
  keyed by `rule.name` and nothing made names unique, so a pair — the obvious
  one being a rule over a threshold and one under it — shared a `fired` flag,
  each clearing what the other had just set, and the hook command re-spawned on
  every single refresh for as long as the condition held. State is keyed by
  position now.

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
