#!/usr/bin/env bash
#
# The build driver. `make` delegates every target here so that the two cannot
# drift apart, and so the same commands work on a machine with no make.
#
#   ./scripts/build.sh <command> [args...]
#
# Environment:
#   CARGO     cargo to invoke                     (default: cargo)
#   PREFIX    install prefix                      (default: /usr/local)
#   DESTDIR   staging root prepended to PREFIX    (default: empty)
#   BIN_DIR   where `bin` drops the binary        (default: ./bin)
#   DIST_DIR  where `dist` writes the tarball     (default: ./dist)
#
#   DOCKER_SOCK  the daemon socket `e2e` hands its runner container
#                                                (default: /var/run/docker.sock)

set -euo pipefail

cd "$(dirname "$0")/.."

CARGO=${CARGO:-cargo}
PREFIX=${PREFIX:-/usr/local}
DESTDIR=${DESTDIR:-}
BIN_DIR=${BIN_DIR:-bin}
DIST_DIR=${DIST_DIR:-dist}

NAME=crabmon
RELEASE_BIN=target/release/$NAME

# The e2e harness image, and the volumes that keep its cargo cache off the
# mounted checkout. Named rather than anonymous so a second run reuses the
# compiled dependencies instead of rebuilding them from scratch.
E2E_RUNNER_IMAGE=${E2E_RUNNER_IMAGE:-crabmon-e2e-runner:local}
E2E_CARGO_VOLUME=${E2E_CARGO_VOLUME:-crabmon-e2e-cargo}
E2E_TARGET_VOLUME=${E2E_TARGET_VOLUME:-crabmon-e2e-target}

# The version lives in Cargo.toml and nowhere else; every other mention of it
# is derived, so a bump cannot half-happen.
version() {
  sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1
}

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }

# `install -D` is a GNU extension: BSD and macOS do not have it, and this is a
# project that cares about FreeBSD. Make the directory ourselves instead.
install_file() { # mode source destination
  mkdir -p "$(dirname "$3")"
  install -m "$1" "$2" "$3"
}

# ----------------------------------------------------------------- commands

cmd_build() {
  say "building (debug)"
  $CARGO build "$@"
}

cmd_release() {
  say "building (release)"
  $CARGO build --release "$@"
}

# The release binary somewhere predictable, so a shell alias or a $PATH entry
# does not have to name target/release.
cmd_bin() {
  cmd_release
  mkdir -p "$BIN_DIR"
  cp -f "$RELEASE_BIN" "$BIN_DIR/$NAME"
  say "$BIN_DIR/$NAME  ($(version))"
}

# Cross builds. Each platform is a rustc target and a linker:
#
#   arm64    aarch64-unknown-linux-musl     aarch64-linux-gnu-gcc   static; the uConsole, Raspberry Pi
#   armhf    armv7-unknown-linux-gnueabihf  arm-linux-gnueabihf-gcc dynamic glibc, 32-bit ARM
#   riscv64  riscv64gc-unknown-linux-musl   riscv64-linux-gnu-gcc   static
#   static   x86_64-unknown-linux-musl      (the default cc)        static, runs on any x86-64 Linux
#
# The musl ones are fully static, so the binary does not care which glibc the
# machine ships. Each needs `rustup target add <target>` and, except `static`,
# the matching gcc to link with (Debian/Ubuntu: gcc-aarch64-linux-gnu,
# gcc-arm-linux-gnueabihf, gcc-riscv64-linux-gnu). Override with <NAME>_TARGET
# and <NAME>_LINKER, e.g. ARM64_TARGET=aarch64-unknown-linux-gnu.
ARM64_TARGET=${ARM64_TARGET:-aarch64-unknown-linux-musl}
ARM64_LINKER=${ARM64_LINKER:-aarch64-linux-gnu-gcc}
ARMHF_TARGET=${ARMHF_TARGET:-armv7-unknown-linux-gnueabihf}
ARMHF_LINKER=${ARMHF_LINKER:-arm-linux-gnueabihf-gcc}
RISCV64_TARGET=${RISCV64_TARGET:-riscv64gc-unknown-linux-musl}
RISCV64_LINKER=${RISCV64_LINKER:-riscv64-linux-gnu-gcc}
STATIC_TARGET=${STATIC_TARGET:-x86_64-unknown-linux-musl}
STATIC_LINKER=${STATIC_LINKER:-}

# The variable names for a platform: the target triple and the linker.
platform_target() { local v=${1^^}_TARGET; echo "${!v}"; }
platform_linker() { local v=${1^^}_LINKER; echo "${!v}"; }

# Tell the person what is missing instead of letting rustc or the linker say
# it in their own words, which is "linker `riscv64-linux-gnu-gcc` not found"
# after a minute of compiling dependencies.
check_platform() { # platform
  local target linker
  target=$(platform_target "$1")
  linker=$(platform_linker "$1")
  if command -v rustup >/dev/null 2>&1 &&
    ! rustup target list --installed 2>/dev/null | grep -qx "$target"; then
    echo "the rustc target $target is not installed: rustup target add $target" >&2
    exit 1
  fi
  if [ -n "$linker" ] && ! command -v "$linker" >/dev/null 2>&1; then
    echo "the linker $linker is not installed (Debian/Ubuntu: apt install gcc-${linker%-gcc}), or set ${1^^}_LINKER" >&2
    exit 1
  fi
}

build_cross() { # platform [cargo args...]
  local platform=$1 target linker
  shift
  check_platform "$platform"
  target=$(platform_target "$platform")
  linker=$(platform_linker "$platform")
  say "building (release, $target)"
  local var=CARGO_TARGET_$(echo "$target" | tr 'a-z-' 'A-Z_')_LINKER
  if [ -n "$linker" ]; then
    env "$var=${!var:-$linker}" $CARGO build --release --target "$target" "$@"
  else
    $CARGO build --release --target "$target" "$@"
  fi
}

bin_cross() { # platform [cargo args...]
  local platform=$1 target
  shift
  build_cross "$platform" "$@"
  target=$(platform_target "$platform")
  mkdir -p "$BIN_DIR"
  cp -f "target/$target/release/$NAME" "$BIN_DIR/$NAME-$platform"
  say "$BIN_DIR/$NAME-$platform  ($(version), $target)"
}

# A static binary links nothing, so cargo-deb's `$auto` finds nothing and the
# package carries an empty `Depends:` field, which lintian flags and some
# tools reject. cargo-deb has no way to omit the field, so the package is
# unpacked, the empty line removed and the package rebuilt.
strip_empty_depends() { # deb
  local deb=$1 tmp
  dpkg-deb -f "$deb" Depends 2>/dev/null | grep -q '[^[:space:]]' && return 0
  command -v dpkg-deb >/dev/null 2>&1 || return 0
  tmp=$(mktemp -d)
  dpkg-deb -R "$deb" "$tmp"
  sed -i '/^Depends:[[:space:]]*$/d' "$tmp/DEBIAN/control"
  dpkg-deb --root-owner-group -b "$tmp" "$deb" >/dev/null
  rm -rf "$tmp"
}

# cargo-deb maps the target/release assets onto target/<triple>/release and
# takes the architecture from the target. It cannot inspect a foreign binary,
# so it warns that it found no $auto dependencies; a static binary has none.
deb_cross() { # platform
  local platform=$1 target
  target=$(platform_target "$platform")
  if ! command -v cargo-deb >/dev/null 2>&1; then
    say "installing cargo-deb"
    $CARGO install cargo-deb
  fi
  build_cross "$platform"
  say "packaging ($platform)"
  $CARGO deb --no-build --target "$target"
  local deb
  for deb in "target/$target"/debian/*.deb; do strip_empty_depends "$deb"; done
  ls -1 "target/$target"/debian/*.deb
}

# An rpm from the same assets, by cargo-generate-rpm, which is pure Rust: it
# needs neither rpmbuild nor the rpm package. The architecture comes from the
# target; the binary is stripped with the host's strip only where it is the
# host's architecture, so a foreign one ships unstripped rather than corrupted.
rpm_cross() { # platform
  local platform=$1 target
  target=$(platform_target "$platform")
  if ! command -v cargo-generate-rpm >/dev/null 2>&1; then
    say "installing cargo-generate-rpm"
    $CARGO install cargo-generate-rpm
  fi
  build_cross "$platform"
  say "packaging ($platform rpm)"
  $CARGO generate-rpm --target "$target" --auto-req disabled
  ls -1 "target/$target"/generate-rpm/*.rpm 2>/dev/null || ls -1 target/generate-rpm/*.rpm
}

cmd_bin_arm64() { bin_cross arm64 "$@"; }
cmd_bin_armhf() { bin_cross armhf "$@"; }
cmd_bin_riscv64() { bin_cross riscv64 "$@"; }
cmd_bin_static() { bin_cross static "$@"; }

cmd_deb_arm64() { deb_cross arm64; }
cmd_deb_armhf() { deb_cross armhf; }
cmd_deb_riscv64() { deb_cross riscv64; }
cmd_deb_static() { deb_cross static; }

cmd_rpm() {
  if ! command -v cargo-generate-rpm >/dev/null 2>&1; then
    say "installing cargo-generate-rpm"
    $CARGO install cargo-generate-rpm
  fi
  cmd_release
  say "packaging (rpm)"
  $CARGO generate-rpm --auto-req disabled
  ls -1 target/generate-rpm/*.rpm
}
cmd_rpm_arm64() { rpm_cross arm64; }
cmd_rpm_armhf() { rpm_cross armhf; }
cmd_rpm_riscv64() { rpm_cross riscv64; }
cmd_rpm_static() { rpm_cross static; }

cmd_run() {
  $CARGO run -- "$@"
}

cmd_test() {
  say "running the test suite"
  # --all-targets is what CI runs: the in-crate unit tests plus every
  # integration suite, rather than the lib tests alone.
  $CARGO test --all-targets "$@"
}

# The end-to-end suite builds a container and runs the binary inside it, which
# costs minutes on a cold cache — so `test` leaves it switched off and this is
# how you ask for it. `cargo test` is still the thing being run: the suite
# skips itself unless CRABMON_E2E says otherwise.
#
# The toolchain is containerised too, so this works on a machine that has
# Docker and nothing else — which is the point of an end-to-end suite that
# anyone can run. `e2e-local` is the same tests driven by the toolchain you
# already have, which is faster to iterate on; both need a Docker daemon,
# because containers are what they test against.
cmd_e2e() {
  require_docker
  local sock=${DOCKER_SOCK:-/var/run/docker.sock}
  if [ ! -S "$sock" ]; then
    echo "no Docker socket at $sock; set DOCKER_SOCK to point at one" >&2
    exit 1
  fi
  say "running the end-to-end suite in Docker, toolchain included"
  # The checkout is mounted read-only and cargo's output goes to named
  # volumes, so a run cannot leave a root-owned `target/` — or anything else —
  # in the tree it was pointed at.
  #
  # The daemon is the host's, reached through its socket, so the containers
  # the tests create are siblings of this one rather than nested: no
  # privileged mode, no daemon inside a daemon. Handing over that socket is
  # root-equivalent access to the host, which is the usual arrangement for a
  # build agent and the reason `e2e-local` exists for anyone who would rather
  # not.
  docker build -q -f tests/e2e/Dockerfile.runner -t "$E2E_RUNNER_IMAGE" . >/dev/null
  docker run --rm \
    -v "$PWD:/src:ro" \
    -v "$sock:/var/run/docker.sock" \
    -v "$E2E_CARGO_VOLUME:/cargo" \
    -v "$E2E_TARGET_VOLUME:/target" \
    "$E2E_RUNNER_IMAGE" test --test e2e_docker "$@"
}

# The same tests, driven by the toolchain already on this machine. No runner
# container, nothing handed the Docker socket, and the output lands in the
# ordinary `target/` — which is what makes it the one to iterate in.
cmd_e2e_local() {
  require_docker
  if ! command -v "$CARGO" >/dev/null 2>&1; then
    echo "$CARGO is not installed; use \`make e2e\`, which brings its own" >&2
    exit 1
  fi
  say "running the end-to-end suite with the local toolchain"
  CRABMON_E2E=1 $CARGO test --test e2e_docker "$@"
}

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not installed; the e2e suite needs it" >&2
    exit 1
  fi
}

cmd_test_unit() {
  say "running the in-crate unit tests"
  $CARGO test --lib "$@"
}

cmd_fmt() {
  say "formatting"
  $CARGO fmt --all
}

cmd_fmt_check() {
  say "checking formatting"
  $CARGO fmt --all -- --check
}

cmd_lint() {
  say "clippy"
  $CARGO clippy --all-targets -- -D warnings
}

cmd_check() {
  say "type-checking"
  $CARGO check --all-targets
}

# Everything CI enforces, in the order that fails cheapest first.
cmd_ci() {
  cmd_fmt_check
  cmd_lint
  cmd_test
}

cmd_deb() {
  if ! command -v cargo-deb >/dev/null 2>&1; then
    say "installing cargo-deb"
    $CARGO install cargo-deb
  fi
  cmd_release
  say "packaging"
  $CARGO deb
  ls -1 target/debian/*.deb
}

# A tarball of exactly what `install` would put on a machine, for the platforms
# the deb does not cover.
cmd_dist() {
  cmd_release
  local v arch stage tarball
  v=$(version)
  arch=$(uname -m)
  stage="$DIST_DIR/$NAME-$v-$arch"
  tarball="$NAME-$v-$arch-$(uname -s | tr '[:upper:]' '[:lower:]').tar.gz"

  rm -rf "$stage"
  mkdir -p "$stage/completions"
  cp "$RELEASE_BIN" "$stage/"
  cp "docs/$NAME.1" "$stage/"
  cp completions/* "$stage/completions/"
  cp README.md LICENSE "$stage/"
  cp docs/CHANGELOG.md docs/DESIGN.md "$stage/"

  tar -czf "$DIST_DIR/$tarball" -C "$DIST_DIR" "$NAME-$v-$arch"
  rm -rf "$stage"
  say "$DIST_DIR/$tarball"
}

cmd_install() {
  # Always rebuild first: an install that ships a stale binary from an earlier
  # checkout is worse than one that takes a second longer.
  cmd_release
  local root="$DESTDIR$PREFIX"
  say "installing to $root"
  install_file 755 "$RELEASE_BIN" "$root/bin/$NAME"
  install_file 644 "docs/$NAME.1" "$root/share/man/man1/$NAME.1"
  install_file 644 "completions/$NAME.bash" "$root/share/bash-completion/completions/$NAME"
  install_file 644 "completions/$NAME.zsh" "$root/share/zsh/site-functions/_$NAME"
  install_file 644 "completions/$NAME.fish" "$root/share/fish/vendor_completions.d/$NAME.fish"
}

cmd_uninstall() {
  local root="$DESTDIR$PREFIX"
  say "removing from $root"
  rm -f \
    "$root/bin/$NAME" \
    "$root/share/man/man1/$NAME.1" \
    "$root/share/bash-completion/completions/$NAME" \
    "$root/share/zsh/site-functions/_$NAME" \
    "$root/share/fish/vendor_completions.d/$NAME.fish"
}

# Redraw the interface snapshot the README carries.
#
# Through tmux rather than by rendering into a buffer: `capture-pane` hands
# back what is actually on a pty, so the braille sparklines and the box drawing
# come out as they are drawn rather than as something that approximates them.
# From the release build, because a debug binary spends enough CPU to put
# itself at the top of its own process table, which is both unflattering and
# untrue.
#
# The capture is of whatever machine runs it, so re-running this replaces the
# committed one with yours. That is the point — it is a demonstration of the
# interface, not a fixture — and it is why the README's copy is replaced in
# place between its markers rather than added to. What would identify the
# machine (hostname, MAC and IP addresses, container ids) is masked by
# scripts/embed-snapshot.py before it is written.
SNAPSHOT_COLS=${SNAPSHOT_COLS:-100}
SNAPSHOT_ROWS=${SNAPSHOT_ROWS:-30}
SNAPSHOT_WAIT=${SNAPSHOT_WAIT:-6}

cmd_snapshot() {
  command -v tmux >/dev/null 2>&1 || {
    echo "tmux is needed to capture a pty; install it, or edit README.md by hand" >&2
    exit 1
  }
  command -v python3 >/dev/null 2>&1 || {
    echo "python3 is needed to replace the block in README.md" >&2
    exit 1
  }
  say "building the release binary the snapshot is taken from"
  $CARGO build --release --quiet
  session="crabmon-snapshot-$$"
  capture=$(mktemp)
  say "capturing ${SNAPSHOT_COLS}x${SNAPSHOT_ROWS} after ${SNAPSHOT_WAIT}s"
  tmux new-session -d -s "$session" -x "$SNAPSHOT_COLS" -y "$SNAPSHOT_ROWS" \
    "exec ./target/release/$NAME $*" 2>/dev/null
  # Long enough for a second sample, so the charts have something in them.
  sleep "$SNAPSHOT_WAIT"
  tmux capture-pane -p -t "$session" | sed -e 's/[[:space:]]*$//' >"$capture"
  tmux kill-session -t "$session" 2>/dev/null || true

  if [ ! -s "$capture" ]; then
    echo "the capture came back empty; can the binary start here?" >&2
    rm -f "$capture"
    exit 1
  fi
  SNAPSHOT_CAPTURE="$capture" python3 scripts/embed-snapshot.py
  rm -f "$capture"
  say "README.md updated"
}

cmd_clean() {
  say "cleaning"
  $CARGO clean
  rm -rf "$BIN_DIR" "$DIST_DIR"
}

cmd_version() {
  version
}

cmd_help() {
  cat <<EOF
$NAME $(version) — build driver

Usage: ./scripts/build.sh <command>       (or: make <command>)

  build        debug build
  release      optimised build
  bin          release build, binary copied to $BIN_DIR/$NAME
  run          run the debug build (extra arguments are passed through)

  test         the whole suite: unit tests and every integration suite
  test-unit    the in-crate unit tests only
  e2e          the end-to-end suite, toolchain and all, in containers —
               needs Docker and nothing else, not even Rust
  e2e-local    the same tests, driven by the toolchain on this machine
  check        type-check without producing a binary
  fmt          reformat the tree
  fmt-check    fail if the tree is not formatted
  lint         clippy, warnings denied
  ci           fmt-check, lint and test — what CI runs

  deb          Debian package into target/debian
  rpm          RPM package into target/generate-rpm (needs no rpmbuild)

  Cross builds — `bin-X` copies a binary to $BIN_DIR/$NAME-X, `deb-X` and
  `rpm-X` build the package for it. X is arm64 (static, uConsole and
  Raspberry Pi), armhf (32-bit ARM, glibc), riscv64 (static) or static
  (x86-64, static; runs on any distribution):

  bin-arm64    cross-compile for aarch64
  bin-armhf    cross-compile for 32-bit ARM
  bin-riscv64  cross-compile for riscv64
  bin-static   static x86-64 binary (musl)
  deb-arm64    Debian package for arm64
  deb-armhf    Debian package for armhf
  deb-riscv64  Debian package for riscv64
  deb-static   Debian package of the static x86-64 binary
  rpm-arm64    RPM package for aarch64
  rpm-armhf    RPM package for armv7hl
  rpm-riscv64  RPM package for riscv64
  rpm-static   RPM package of the static x86-64 binary
  dist         release tarball into $DIST_DIR
  install      install under PREFIX=$PREFIX (honours DESTDIR)
  uninstall    remove what install put there

  snapshot     redraw the interface snapshot in README.md (needs tmux)
  clean        cargo clean, plus $BIN_DIR and $DIST_DIR
  version      print the version from Cargo.toml
  help         this message
EOF
}

# -------------------------------------------------------------------- entry

main() {
  local command=${1:-help}
  shift || true
  # Targets are spelled with dashes; functions cannot be.
  local fn="cmd_${command//-/_}"
  if ! declare -F "$fn" >/dev/null; then
    echo "unknown command: $command" >&2
    echo >&2
    cmd_help >&2
    exit 2
  fi
  "$fn" "$@"
}

main "$@"
