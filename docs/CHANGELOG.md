# Changelog

Notable changes to crabmon. Versions follow [semantic versioning](https://semver.org),
with the usual 0.x caveat that minor releases may change behaviour.

## 0.14.0 — unreleased

What a round of deliberately awkward inputs turned up. Two of a fleet's
targets can reduce to the same filesystem name, and one can reduce to no name
at all — both of which quietly cost a host its recording.

### Fixed

- **Two fleet targets could share a recording and truncate each other's.**
  A target is reduced to something a filesystem will take, and that reduction
  is not injective: `deploy@db-2` and `deploy-db-2` both become
  `deploy-db-2`, and a target with no usable character left — `@@@` — becomes
  nothing at all. The first pair shared a directory, and since a recorder
  truncates what it opens, the second host destroyed the first one's
  segments; the nameless one fell back to the unlabelled path and wrote among
  the flight dumps, where rotation would prune them. `user@host` is the
  ordinary way to write an SSH target, which is exactly how two of them
  collide. Names are now chosen over the whole target list by
  `record::unique_labels`, which keeps a name that is already free — so an
  ordinary fleet is unchanged — and disambiguates the rest.

- **A recording that could not be written said so once a frame.** A read-only
  directory or a full disk does not get better by itself, so `--serve` wrote
  an identical line per scrape for as long as it ran. Reported on the way
  into failure and on the way out of it now, the same shape as an alert.

### Changed

- **The documentation and the ignore rules cover how a fleet's recordings are
  named.** `record::unique_labels` is what names them now, including the
  disambiguated forms two colliding targets and a nameless one get, so
  `tests/docs.rs` checks those shapes against `.gitignore` rather than the
  hand-written examples it used to. The README and the man page say that the
  names are chosen over the whole fleet and why, and that a recording which
  cannot be written reports once rather than once a frame.

## 0.13.0 — unreleased

The pass over what the last two releases left behind: a defect in `--doctor`
found by asking whether its tests would survive CI on three platforms, the
documentation for the three modes added since, and the ignore rules for the
files a fleet now writes.

### Fixed

- **`--doctor` reported on the wrong config file.** It read the default path
  whatever `--config` named, so the one use it was built for — checking a
  config before deploying it — reported on the operator's own file instead,
  while quoting the rule count from the file actually loaded. It now reports
  on the config in force, and says "1 alert rule" rather than "1 alert rules".
- **Two new tests would have failed on Windows or a slow runner.** The hook
  test built a shell command around a scratch-file path, which is not portable
  to the `sh` that happens to be on a Windows runner's PATH, so the variables
  are now asserted on every platform and only the spawn-and-read-back half is
  Unix-only. A watch-recording test asserted a frame count derived from
  sampling speed, which is the pacing test's subject and not this one's, and
  another looked for the absence of a fixed filename in the working directory.
- **`.gitignore` said nothing about what a fleet writes.** The rules already
  covered it — `*.jsonl` is unanchored — but nothing said why that has to stay
  true now that a fleet export writes `run-web-1.jsonl` beside the path
  `--record` names and a directory per host under the one `--record-rotate`
  names. Both shapes are derived from the naming code and pinned by
  `tests/docs.rs`.

### Changed

- **The design notes and README cover the three modes added since 0.11.0.**
  `--doctor` has a section of its own in both, explaining the marks and why
  the exit code distinguishes "this host cannot do that" from "the config
  asked for something this host cannot do". What a hook is given sits beside
  the alert-rule reasoning it belongs with.

## 0.12.0 — unreleased

Three things an operator needs that crabmon could measure and could not say.
A hook knew a rule had fired and could not tell anyone which; a fleet could be
exported and not recorded; and the care taken never to report zero for
something unreadable left "is that nothing, or something I am not allowed to
see?" as a question with no answer anywhere.

### Added

- **Alert hooks are told what fired.** A hook was a bare command string run
  with nothing but the shell's own environment, so the command was the only
  place the context could live: every rule carried its own message, and even
  then the hook could not say what the measurement was or find the recording
  written beside it. A hook now runs with `CRABMON_RULE`, `CRABMON_KIND`,
  `CRABMON_STATE` (`firing` or `cleared`), `CRABMON_VALUE`,
  `CRABMON_THRESHOLD`, `CRABMON_MESSAGE`, `CRABMON_HOST`, `CRABMON_AT` and —
  where `[record] flight` opened one — `CRABMON_DUMP` and `CRABMON_SUMMARY`
  pointing at the evidence. Every variable is always set, empty where it does
  not apply, so a hook written with `set -u` need not guess which exist. One
  `page-oncall` script now serves every rule on every host.

- **`crabmon --doctor`.** crabmon is careful never to report zero for
  something it could not read — an unreadable fd table is unknown, a root-only
  `/dev/kmsg` says so rather than showing an empty list, a rule with nothing
  to measure is skipped. That is right, and it leaves "is that nothing, or
  something I am not allowed to see?" as a question nothing answered.
  `--doctor` asks all of them up front: the platform, privileges, pressure
  stall, `/dev/kmsg`, RAPL, cgroup version, GPU, fd tables, `smaps_rollup`,
  per-process IO, whether the config parses, and which rules can measure
  nothing on this host. It exits **1** when something the config actually
  asks for cannot work here, so it is a pre-deploy check and not only
  something to read; a missing GPU is not a failure, a `kind = "fd"` rule on
  a host whose fd tables are unreadable is. `--format json` emits the same
  report for something other than a person.

- **A fleet export records a timeline per host.** `--serve --remote` refused
  `--record` and `--record-rotate` outright, because a fleet has one timeline
  per machine and the flags name one place. Each host now gets its own:
  `run.jsonl` becomes `run-web-1.jsonl`, and a rotating set gets a directory
  per host under the one named. A directory rather than a name, because a
  segment is `crabmon-<unix>.jsonl` while a flight dump is
  `crabmon-<rule>-<unix>.jsonl` — putting the host in the name would make the
  two indistinguishable, and `rotate_dir` defaults to `flight_dir`, so
  rotation would have pruned the evidence sitting beside it. Only frames that
  actually arrived are recorded, so a host that has gone quiet leaves a gap
  rather than a flat line of its last good sample.


## 0.11.0 — unreleased

Recording that does not need somebody watching, growth measured in more than
one dimension, and the exporter defect a one-core machine found that faster
ones had been hiding for as long as the exporter has existed. The
always-on recorder was the feature that could not be deployed — it was driven
by the interface's event loop, so the one place it belongs, a unit file, was
the one place it did nothing. The rest came out of testing those two: four of
the eight fixes below are things the new tests tripped over rather than
anything anyone set out to look for.

### Added

- **Recording without a terminal.** `--record` and `--record-rotate` now work
  in every mode that samples on a loop — `--stream`, `--serve` and `--watch`
  as well as the interface — through one `record::Recordings` rather than four
  copies of the flag-and-config resolution. That is what makes an always-on
  recorder deployable: a unit running
  `crabmon --serve :9100 --alerts --record-rotate /var/log/crabmon` exports
  metrics, evaluates the rules and keeps the last hour on disk with nobody
  watching. Under `--stream` the frame on stdout keeps the full inventory a
  collector needs while the one on disk is trimmed by `[record]`; under
  `--serve` the cadence is the scrape's, because that is when a sample is
  actually taken. `--once` and `--diff` refuse both flags — a recording is a
  sequence — as does `--serve --remote`, where a fleet has a timeline per host
  and one file to put them in.

- **`trend` rules can follow CPU and disk IO, not only RSS.** A leak is
  usually spelled in resident memory, but it is not the only thing that
  climbs: a process whose disk rate has risen all week, or whose CPU has crept
  up since a deploy, is the same shape of problem and no threshold catches
  either until it is too late to act on. A `trend` rule with a `query` now
  reads `target` as the series to follow — `rss` (the default), `cpu` or `io`
  — with the threshold in that series' own unit: bytes, percentage points, or
  bytes per second. From a shell the series goes in front of the query,
  `crabmon --watch-rule 'trend:io:postgres>10M'`, and a bare `trend:postgres`
  still means RSS. Only the series some rule actually names are kept, so
  asking about IO does not silently start tracking three series per process.

### Fixed

- **The exporter reset roughly half its scrapes on a slow machine.** `handle`
  read only the request line when no token was configured — "an exporter with
  no token has no reason to read past it" — then wrote the response and closed.
  Closing a socket that still has unread bytes in its receive buffer makes
  Linux send RST instead of FIN, and the RST discards the response that was
  just written, so the client sees a reset connection rather than the 200 the
  exporter sent. It is a race, which is why it hid on a developer's machine
  and was routine on a single-core one: measured there with the e2e suite's
  own client, 11 of 25 scrapes were answered with the headers unread against
  25 of 25 with them read — same binary, same client, back to back. Prometheus
  would have recorded the rest as failed scrapes. The headers are now read to
  the blank line whether or not anything looks at one, still bounded by
  `MAX_HEADERS`; only the `Authorization` value is kept, and only when a token
  is configured.

- **`--record` and `--record-rotate` silently did nothing in a headless
  mode.** Recording was driven by the interface's own sampling loop, so
  `crabmon --once --record run.jsonl` printed a snapshot, wrote no recording,
  said nothing and exited 0 — the failure `--alerts` is already refused for,
  in its worse form, because the file that was asked for simply never
  appeared. Every mode that samples on a loop now records (see **Added**),
  and the two that take a single sample refuse the flags instead.

- **A machine-wide `trend` rule could name a process as though it were the
  cause.** The subject appended to a trend alert's message was gated on the
  rule having a parsed filter rather than on it being about processes — and
  an empty query parses into the filter that matches *everything*, so with
  any per-process trend rule also configured, "machine memory climbing"
  arrived as `leak +4.0 GiB ≥ 1.0 GiB (chrome over 10m)`. It now names a
  process only when the rule selected one.

- **The exporter's constant-time token comparison folded its length check
  into a `u8`.** Everything above the low eight bits of the length difference
  was thrown away, so two lengths exactly 256 apart folded to zero and left
  the byte loop as the only check. Not a bypass on its own — the bytes still
  have to match the secret — but the length half of the comparison was not
  doing its job. It folds the full `usize` now.

- **The detail pane showed `STRIPE_KEY` and friends in full.** The
  environment mask matched `API_KEY` and `SECRET` as fragments but nothing
  ending in `_KEY`, so an ordinary credential variable reached a pane people
  screenshot into tickets. `_KEY` and `_PWD` are matched as suffixes — a bare
  `KEY` fragment would mask `KEYMAP` and teach people to ignore the mask —
  and `PASSPHRASE` joins the fragment list.

- **Four exporter tests raced the whole machine for a port.** Each asked the
  kernel for a free ephemeral port, dropped the listener, and handed the
  number to `serve`. A full `cargo test` run binds a lot of ephemeral ports,
  and the loser of that race does not fail to connect — it connects to
  whatever took the port and asserts on somebody else's response, which is
  how `a_scrape_over_a_socket_must_carry_the_token` failed a parallel run.
  `serve::serve_listener` now takes an already-bound listener, so the socket
  the test probed with is the one the exporter serves on and there is no
  window at all. `serve_rendered` is that plus `TcpListener::bind` and the
  startup notices.

- **`stream_keeps_to_the_interval_it_was_asked_for` failed a correct binary on
  a loaded machine.** `Pacer` runs as fast as it can when a sample costs more
  than the interval it was given, so the honest expectation is
  `max(interval, sample_cost)` — but the test bounded the gap at twice a
  300 ms refresh, which a few thousand tasks exceed on their own. At load 8 it
  measured 1.17 s and failed the run for being right. It now measures what a
  sample costs on the machine it is running on, by streaming once at the
  refresh floor, and bounds the real run against that. Confirmed still to
  catch the regression it exists for by reinstating the old sleep-then-sample
  behaviour, which it reports as 1.69 s against a 1.5 s bound.

- **`a_process_that_execs_is_reported_as_what_it_became` raced its own first
  sample.** The wrapper exec'd after a fixed `sleep 1` while the test's first
  snapshot walks every process on the machine, which on a loaded box takes
  longer than that — so the run failed on the *setup* assertion, saying
  nothing about crabmon, roughly one time in three. The wrapper now blocks on
  stdin and the test closes the pipe when it is ready, the same frame-driven
  hand-off `tests/e2e_docker.rs` already uses.

## 0.10.0 — unreleased

The measurements that were wrong, and the questions no threshold could ask.
RSS double-counts shared pages, so a process tree could claim more memory than
the machine has; a percentage cannot say whether a disk fills in twenty
minutes or next week; a sum of per-process figures cannot see a container's
limit at all. Plus the four defects the previous release's own design notes
listed as open, and the one they turned up: process control acting on a local
PID while the screen showed another machine's.

### Changed

- **Rust edition 2024, and the dependencies brought up to date.** `sysinfo`
  0.30 → 0.39, `ratatui` 0.26 → 0.30, `crossterm` 0.27 → 0.29, `toml` 0.8 →
  1.1, `dirs` 5 → 7, `nix` 0.29 → 0.31, `signal-hook` 0.3 → 0.4. `rust-version
  = "1.85"` is now declared, so an older toolchain is refused with a sentence
  rather than a parse error. Every `if let` that existed only to be nested
  inside another is now a let-chain.
- **The documentation tests assumed LF line endings.** git checks out with
  CRLF on Windows by default, and the helpers slice on `"\n\n"` and count
  blank-line-separated blocks — so `.gitignore` read as a single section and
  a `between()` marker spanning a line break stopped matching. Repository
  files are now read LF-normalised: these tests are about what the
  documentation says, not how a checkout spells a line break.
- **Three tests encoded assumptions that only hold on Linux.** The RAPL
  fixtures build directories named as the kernel names them — `intel-rapl:0`,
  `intel-rapl:0:1` — and `:` cannot appear in a Windows filename at all, so
  they are Linux-only now rather than renamed into a layout the kernel never
  emits. And `renice_changes_the_process_priority_for_real` asserted that a
  fresh child starts at nice 0, which is true only when the parent is at 0:
  GitHub's macOS runners sit at -10, so it failed on a healthy machine. It
  now compares against the parent's own value, clamped at the kernel's 19.
  `once_honours_the_filter_language` filtered on `pid:1` because init always
  exists — on Linux. Windows has no PID 1, so the filter matched nothing; it
  now filters on the test process's own PID, which exists everywhere and
  checks the filter language rather than the platform.
- **The cfg gates actually hold on every target CI builds.** `-D warnings`
  makes a platform-specific constant that nothing uses on that platform a
  build failure, and three had crept in: `MAX_RECORDS_PER_DRAIN` and the
  `identity` test module (Linux-only), `SHUTDOWN_GRACE` (Unix-only). The
  numeric `uid` is now `None` on Windows, where a user is a SID rather than a
  number, instead of failing to compile. `libc` is pinned to 0.2.189 for the
  transitive copy: 0.2.190 dropped FreeBSD's `NOTE_PCTRLMASK`, which the only
  `mio` that `signal-hook-mio 0.2` permits still uses.
- **`libc` moved to `=1.0.0-alpha.5`.** A pre-release, pinned exactly so a
  routine `cargo update` cannot swap the definitions under this crate's
  `unsafe` syscalls; `nix`, `mio`, `dirs-sys` and `signal-hook` still want
  0.2, so both libc crates are in the tree. Every call through it was checked
  against a live host rather than against the test suite: `ioprio_get` and
  `ioprio_set` round-trip, `getpriority` agrees with `/proc/<pid>/stat` for
  2569 processes including negative nice values (the case that needs
  `__errno_location`), `statvfs` returns inode counts, and `/dev/kmsg` opens
  non-blocking.
- **The thread count was short by one on the new sysinfo.** `tasks()` is the
  set of *sub*tasks and no longer includes the main thread, so THR read 0 for
  a single-threaded process — a number no live process can have — and one
  short for every other. Found by cross-checking a live sample against
  `/proc/<pid>/task`, which is now a test.

### Added

- **PSS: the honest per-process memory figure.** Every memory number was RSS,
  which counts a page shared by eight Chromium processes eight times — so
  adding up a process tree could exceed the memory the machine has. `[procs]
  pss` reads `/proc/<pid>/smaps_rollup` for proportional and unique set size,
  adds the PSS column and the `pss>` and `uss>` filters, and exports both. On
  one real machine the summed figures differ by nearly half: 15.0 GiB of RSS
  against 8.1 GiB of PSS, on a host with 10.3 GiB in use. Opt-in like `fds`,
  and threads are skipped since they share their process's address space.
- **A GROWTH column.** The per-process history crabmon already keeps for the
  TREND column and the detail pane, surfaced and sortable, so a leak is found
  by pressing a key rather than by writing a rule and waiting for it.
- **`kind = "fillin"`: time to full, not percent full.** 95% of a 4 TB array
  is days of warning and 95% of a 20 GB root is twenty minutes; the same
  threshold fires far too late on one and far too early on the other. The
  disks panel shows `full in 02:41:00` for any mount that will fill within a
  week, so the figure is there before anybody writes a rule.
- **Per-container accounting from the kernel, not from sums.** Each
  container's and unit's own cgroup is read directly — `memory.current`,
  `memory.max`, `cpu.stat`, `cpu.max` — so the grouped view carries a LIMIT
  column, `crabmon_cgroup_*` series are exported, and `kind = "cgroup"` can
  say "this container is at 98% of what it is allowed", which a sum of
  per-process figures cannot express at all.
- **`--format text`.** A `top -b -n1`-shaped table for a person to read, laid
  out by the same column planner the interface uses. Getting numbers out of
  crabmon previously meant JSON, CSV, or a full-screen interface.
- **`Ctrl-R` reloads the config.** Thresholds, colours, panels, columns and
  `[[alert]]` rules are re-read and applied; the filter, tags, pins,
  selection, layout and sort are session state and are kept. A file that does
  not parse changes nothing. Editing a config mid-incident is exactly when
  restarting costs the most.
- **`o` folds a subtree in the tree view**, marking the row with how many
  rows it stands in for (`+40 chrome`). A browser with two hundred children
  otherwise buries everything else, which is the opposite of what the tree
  view is for.
- **`--serve unix:/run/crabmon.sock`.** The companion to the bearer token: no
  port, no network, filesystem permissions do the work. Created mode `0660`
  rather than at the mercy of the umask; a stale socket is replaced, a live
  one is never stolen.
- **`kind = "trend"`: alerts on growth, not level.** Every other rule compares
  a measurement against a line, which catches a machine that is already in
  trouble and never catches the process whose RSS has gone up 200 MB every ten
  minutes since Tuesday. `over_secs` is the window, `threshold` is in bytes,
  and a `query` picks the processes whose RSS is watched — the worst grower
  wins and is named in the message. A recycled PID starts a new series, a
  series reports nothing until it spans half its window, and per-process
  series are only kept when a rule asks for them.
- **Kernel events: OOM kills, segfaults, oopses and block-IO errors.** Read
  from `/dev/kmsg` into a panel, exported as `crabmon_kernel_events`, and
  alertable with `kind = "kernel"`. The dashboard shows memory *now* and the
  process the OOM killer chose is no longer in the table, so "what got killed,
  and when" was the one question nothing on screen could answer. Opening the
  device replays the ring, so a kill from twenty minutes ago is visible the
  moment crabmon starts; where `dmesg_restrict` makes it root-only the panel
  says so rather than showing an empty list.
- **`--serve --remote web-1,web-2,db-1`: a fleet exporter.** crabmon could
  watch several hosts over SSH and could export what it measured; the two
  could not meet. Every series now carries `host="..."`, each host evaluates
  the rules on its own engine, and flight dumps are named per host.
- **`I`: IO priority.** Signals, renice and affinity were three of the four
  knobs; CPU priority does nothing for a process saturating a disk, which is
  the one anybody actually wants to turn down. Takes what `ionice(1)` takes.
- **The detail pane carries OOM score, IO priority, cgroup path and
  environment.** Values of variables that look like credentials are withheld:
  a monitor is a thing people screenshot into tickets and project onto a wall
  during an incident.
- **An incident summary beside every flight dump.** `crabmon-<rule>-<unix>.md`:
  what fired and by how much, the machine at that moment, what the kernel
  said, the busiest processes, and what changed over the frames before it. A
  JSONL recording is evidence nobody can read without crabmon in front of it.
- **`[serve] token`, and `CRABMON_SERVE_TOKEN`.** A scrape carries the argv of
  every process on the machine, and binding the exporter off loopback was one
  flag away with no way to ask for a password. `/healthz` stays exempt and the
  comparison is constant-time.
- **`--record-rotate DIR`: a recording that never ends.** `--record` only
  helps if you started it before the incident, and the flight recorder only
  helps if an alert rule happened to describe what went wrong. This keeps the
  last `[record] rotate_keep` segments of `rotate_frames` frames on disk at
  all times, oldest deleted, so a replay of the last hour or two always
  exists. Rotation only deletes files of its own naming.

### Fixed

- **Process control acted on a *local* PID while showing another machine's
  processes.** Under `--remote` or `--replay` the rows on screen belong to
  another host or another time, and nothing stopped `t`, `r` or `A` from
  signalling, renicing or pinning whatever local process happened to hold that
  number. All three now refuse outright. The detail pane's sockets, open files
  and listening ports came from the local `/proc` for the same reason and now
  say "not on this host"; the nice value comes from the snapshot's own row.
- **An outage was charted as flat, real-looking measurement.**
  `ThreadedSource` counted frames *delivered* rather than measurements taken,
  and `RemoteSource` re-hands its last good frame whenever the far end stalls,
  so a dead host drew a steady line and the `sampling…` notice never appeared.
  A replay had the same shape from the other end: it held its final frame for
  ever while reporting no frame id at all, feeding copies into the history
  until the recorded incident scrolled out of the charts. Each source now says
  honestly when it has measured something new.
- **An explicit `[procs] columns` list blanked every process name on a narrow
  terminal.** The automatic planner reserves room for COMMAND as it adds
  columns; the configured path did not, so `name_width` reached zero and the
  table rendered as numbers with nothing beside them. Configured columns are
  now dropped from the right until COMMAND fits.
- **A `--watch` typo exited 1, which is also what a match exits.** So
  `crabmon --watch-rule 'psi:oi>20' || page-oncall` paged on a misspelling.
  Failures now exit 2, and a rule that names a pressure resource or a kernel
  event that does not exist is refused rather than measuring nothing for ever
  and then reporting a clean run.

- **A config file with one typo in it silently threw away every setting, then
  overwrote itself with the defaults.** A parse error was swallowed by
  `unwrap_or_default()`, so a stray bracket or a misspelled `kind` left
  crabmon running with none of the file's thresholds, columns or — worst of
  all — `[[alert]]` rules, with nothing on screen or on stderr to say so. The
  TUI then wrote the defaults back over the file on exit, so the user's
  settings were gone for good. The error is now reported on stderr and in the
  status line, and a file that could not be parsed is never written back over.
  Saving is also done through a temporary file and a rename, so a crash or a
  full disk can no longer leave a truncated config behind.
- **A parent cycle emptied the whole tree view.** Every process in a cycle has
  its parent inside the table, so none of them was a root, and the depth-first
  walk started from roots alone — the cycle and everything under it simply
  vanished from `T`. Processes a cycle leaves unreachable are now re-seeded as
  roots. The test that was meant to cover this asserted `len() <= 2`, which an
  empty table passes.
- **A panicking scrape cost the exporter a connection slot permanently.**
  `--serve` released a connection by decrementing a counter at the end of the
  handler thread, which a panic skips. Sixteen panics and the exporter
  answered `503` to everything for the rest of its life while still accepting
  connections. The slot is now released on drop, which also covers a thread
  that could not be spawned at all.
- **Alert hooks were left in the process table as zombies.** A hook is
  deliberately not waited on, so every firing left a defunct `sh` behind for
  the life of a process that is typically left running for weeks — unbounded
  with a flapping rule, in the one program whose own process table is on
  screen. Finished hooks are now reaped, without ever blocking on one.
- **A recycled PID inherited the previous process's history.** The per-process
  CPU, memory and IO series were keyed by PID alone, so a process that exited
  between two samples left its trend column and its detail-pane charts
  attached to whatever took its number. The series now belongs to a start
  time as well, and resets when the PID changes hands.
- **An unreachable `--remote` host was retried on every refresh.** A stream
  that dropped was reopened on the next tick, and an unreachable host fails in
  milliseconds, so a host that was down cost an `ssh` process and two reader
  threads several times a second for as long as it stayed down. Reconnects now
  back off from one second to thirty, resetting as soon as a frame arrives.

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
