#!/usr/bin/env bash
set -euxo pipefail

cd "$(dirname "$0")/.."


if ! command -v cargo-deb >/dev/null 2>&1; then
  cargo install cargo-deb
fi

cargo build --release
cargo deb

echo "Built debs:"
ls -1 target/debian/*.deb
