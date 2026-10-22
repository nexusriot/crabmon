#!/usr/bin/env python3
"""Put a captured terminal frame into README.md, between its markers.

Called by `scripts/build.sh snapshot`, which does the capturing. Separate from
it because a here-document inside a here-document is a thing nobody should
have to read, and because the replacement is the half worth being careful
about: it rewrites a file that is otherwise hand-written.

The capture arrives through SNAPSHOT_CAPTURE rather than argv so that a path
with a space in it cannot turn into two arguments.
"""

import os
import re
import sys

BEGIN = "<!-- snapshot:begin -->"
END = "<!-- snapshot:end -->"


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
