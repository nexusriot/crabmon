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

set -euo pipefail

cd "$(dirname "$0")/.."

CARGO=${CARGO:-cargo}
PREFIX=${PREFIX:-/usr/local}
DESTDIR=${DESTDIR:-}
BIN_DIR=${BIN_DIR:-bin}
DIST_DIR=${DIST_DIR:-dist}

NAME=crabmon
RELEASE_BIN=target/release/$NAME

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

cmd_run() {
  $CARGO run -- "$@"
}

cmd_test() {
  say "running the test suite"
  # --all-targets is what CI runs: the in-crate unit tests plus every
  # integration suite, rather than the lib tests alone.
  $CARGO test --all-targets "$@"
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
  cp README.md CHANGELOG.md LICENSE "$stage/"

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
  check        type-check without producing a binary
  fmt          reformat the tree
  fmt-check    fail if the tree is not formatted
  lint         clippy, warnings denied
  ci           fmt-check, lint and test — what CI runs

  deb          Debian package into target/debian
  dist         release tarball into $DIST_DIR
  install      install under PREFIX=$PREFIX (honours DESTDIR)
  uninstall    remove what install put there

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
