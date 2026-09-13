#!/usr/bin/env bash
#
# Kept as the name CI and the README have always used. The packaging itself
# lives in build.sh, so there is one implementation of it rather than two.

set -euo pipefail

exec "$(dirname "$0")/build.sh" deb
