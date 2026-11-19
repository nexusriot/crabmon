#!/usr/bin/env python3
"""Put a captured terminal frame into README.md, between its markers.

Called by `scripts/build.sh snapshot`, which does the capturing. Separate from
it because a here-document inside a here-document is a thing nobody should
have to read, and because the replacement is the half worth being careful
about: it rewrites a file that is otherwise hand-written.

The capture arrives through SNAPSHOT_CAPTURE rather than argv so that a path
with a space in it cannot turn into two arguments.

The frame is masked on the way in: the hostname, MAC and IPv4 addresses and
container ids are replaced, at their own width. SNAPSHOT_HOSTNAME names the
machine when the capture was taken somewhere other than here.
"""

import os
import re
import socket
import sys

BEGIN = "<!-- snapshot:begin -->"
END = "<!-- snapshot:end -->"


# A snapshot is a picture of a real machine, published. Anything in it that
# names or locates that machine is replaced before it reaches README.md.
# Every replacement keeps the length of what it replaces, so the box drawing
# around it still lines up.
MAC = re.compile(r"\b[0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5}\b")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# Container and image ids, full (64) or short (12): long runs of hex that
# carry at least one digit and one letter, so a word like `deadbeef` and a
# plain number are left alone.
HEX_ID = re.compile(r"\b(?=[0-9a-f]*[0-9])(?=[0-9a-f]*[a-f])[0-9a-f]{12,64}\b")
PLACEHOLDER_HOST = "demo-host.example.net"


def same_length(placeholder: str, original: str) -> str:
    """The placeholder cut or padded to the original's width."""
    if len(placeholder) >= len(original):
        return placeholder[: len(original)]
    return placeholder + " " * (len(original) - len(placeholder))


def hostnames() -> list[str]:
    """The names this machine goes by, longest first so the FQDN is replaced
    before its own first label. SNAPSHOT_HOSTNAME names a machine the capture
    was taken on other than this one, comma-separated."""
    names = {n for n in os.environ.get("SNAPSHOT_HOSTNAME", "").split(",") if n}
    if not names:
        names = {socket.gethostname(), socket.getfqdn()}
    names |= {n.split(".")[0] for n in names}
    # A one- or two-letter label would take bites out of unrelated words.
    return sorted((n for n in names if len(n) >= 3), key=len, reverse=True)


def mask(frame: str) -> str:
    for name in hostnames():
        frame = frame.replace(name, same_length(PLACEHOLDER_HOST, name))
    frame = MAC.sub(lambda m: re.sub(r"[0-9A-Fa-f]", "x", m.group()), frame)
    frame = IPV4.sub(lambda m: re.sub(r"\d", "x", m.group()), frame)
    frame = HEX_ID.sub(lambda m: "x" * len(m.group()), frame)
    return frame


def main() -> int:
    capture = os.environ.get("SNAPSHOT_CAPTURE")
    if not capture:
        print("SNAPSHOT_CAPTURE is not set", file=sys.stderr)
        return 1

    with open(capture, encoding="utf-8") as fh:
        frame = fh.read().rstrip("\n")
    if not frame.strip():
        print("the capture is empty", file=sys.stderr)
        return 1

    with open("README.md", encoding="utf-8") as fh:
        readme = fh.read()
    if BEGIN not in readme or END not in readme:
        print("README.md carries no snapshot markers to replace", file=sys.stderr)
        return 1

    frame = mask(frame)
    block = f"{BEGIN}\n```text\n{frame}\n```\n{END}"
    # A lambda, so a backslash or a `\g` in the captured frame is not read as
    # a replacement-template escape.
    readme = re.sub(
        re.escape(BEGIN) + ".*?" + re.escape(END), lambda _m: block, readme, flags=re.S
    )
    with open("README.md", "w", encoding="utf-8") as fh:
        fh.write(readme)
    print(f"  replaced the snapshot: {len(frame.splitlines())} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
