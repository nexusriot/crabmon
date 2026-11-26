#!/bin/sh
# Build a FreeBSD `.pkg` without FreeBSD.
#
# A pkg(8) package is an xz-compressed tar archive whose first members are two
# manifests and whose remaining members are the files, so it needs `tar`, `xz`
# and `sha256sum` and nothing from the ports tree. `pkg add` on the target
# verifies each file against the checksum in the manifest, which is the only
# thing that makes this more than a tarball.
#
# usage: mkpkg.sh <repo-root> <binary> <version> <abi> <out.pkg>
#   abi is what `pkg config ABI` prints on the target: FreeBSD:15:amd64
set -eu

root=$1 bin=$2 version=$3 abi=$4 out=$5
prefix=/usr/local

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

put() { # mode source destination-below-prefix
  mkdir -p "$stage$prefix/$(dirname "$3")"
  install -m "$1" "$2" "$stage$prefix/$3"
}

put 755 "$bin" bin/crabmon
# Man pages are installed compressed on FreeBSD, and pkg expects the name it
# records to be the name on disk.
mkdir -p "$stage$prefix/share/man/man1"
gzip -9nc "$root/docs/crabmon.1" > "$stage$prefix/share/man/man1/crabmon.1.gz"
chmod 644 "$stage$prefix/share/man/man1/crabmon.1.gz"
put 644 "$root/completions/crabmon.bash" share/bash-completion/completions/crabmon
put 644 "$root/completions/crabmon.zsh" share/zsh/site-functions/_crabmon
put 644 "$root/completions/crabmon.fish" share/fish/vendor_completions.d/crabmon.fish
put 644 "$root/README.md" share/doc/crabmon/README.md
put 644 "$root/docs/CHANGELOG.md" share/doc/crabmon/CHANGELOG.md
put 644 "$root/docs/DESIGN.md" share/doc/crabmon/DESIGN.md
put 644 "$root/LICENSE" share/licenses/crabmon-$version/LICENSE

# `files` maps each installed path to its checksum; flatsize is their total.
files=$(cd "$stage" && find . -type f | sort | sed 's,^\.,,' | while read -r f; do
  printf '    "%s": "%s",\n' "$f" "$(sha256sum "$stage$f" | cut -d' ' -f1)"
done | sed '$ s/,$//')
flatsize=$(cd "$stage" && find . -type f -exec wc -c {} + | awk 'END{print $1}')
# `wc -c` over many files ends with a total line, over one it is the file's
# own size; either way the last number is the sum.

desc="crabmon is a Rust terminal system monitor: per-core CPU history, a process table with tree view and grouping, per-disk and network throughput, listening ports and open sockets, renice and signal control, alert rules with webhook notification, a flight recorder and a fleet view over SSH."

cat > "$stage/+MANIFEST" <<EOF
{
  "name": "crabmon",
  "origin": "sysutils/crabmon",
  "version": "$version",
  "comment": "Fast, keyboard-driven terminal system monitor",
  "maintainer": "nexus.riot@gmail.com",
  "www": "https://github.com/nexusriot/crabmon",
  "abi": "$abi",
  "arch": "$abi",
  "prefix": "$prefix",
  "flatsize": $flatsize,
  "licenselogic": "single",
  "licenses": ["MIT"],
  "desc": "$desc",
  "categories": ["sysutils"],
  "files": {
$files
  }
}
EOF

# The compact manifest is the same without the file list; repository tooling
# reads it without unpacking the archive.
sed '/"files": {/,$d' "$stage/+MANIFEST" | sed '$ s/,$//' > "$stage/+COMPACT_MANIFEST"
echo "}" >> "$stage/+COMPACT_MANIFEST"

mkdir -p "$(dirname "$out")"
# Manifests first, then the tree with paths rooted at `/`, as pkg writes them.
# Files only: pkg creates the directories it needs, and a directory entry it
# did not record is a warning at best.
(cd "$stage" && { echo +COMPACT_MANIFEST; echo +MANIFEST; find usr -type f | sort; } |
  tar --owner=0 --group=0 --numeric-owner --transform "s,^usr,/usr," \
    --absolute-names -cf - -T -) | xz -9 > "$out"
