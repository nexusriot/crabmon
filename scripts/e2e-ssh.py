#!/usr/bin/env python3
"""Opt-in real-host checks. Run with --target user@host and --binary path.
Only creates a private remote /tmp directory and temporary test processes.
"""
import argparse
import json
from pathlib import Path
import shlex
import socket
import subprocess
import tempfile
import time
import urllib.request

REMOTE_TEST = r'''
import json, pathlib, subprocess, sys, time
root = pathlib.Path(sys.argv[1])
binary = str(root / "crabmon")
marker = "crabmon-e2e-workload"
secret = "test-credential-e2e-7d96"
children = []
stream = None

def wait_for(check, message):
    end = time.monotonic() + 20
    while time.monotonic() < end:
        if check(): return
        time.sleep(.1)
    raise AssertionError(message)

def hooks():
    p = root / "hooks"
    return p.read_text().splitlines() if p.exists() else []

try:
    for _ in range(3):
        children.append(subprocess.Popen(["python3", "-c", "import time; time.sleep(120)", marker, "--password=" + secret]))
    cfg = root / "config.toml"
    cfg.write_text('refresh_ms=200\n[record]\ntop_n=0\nmax_cmd_len=2000\nflight=2\nflight_after=1\nflight_dir=' + json.dumps(str(root)) + '\nrotate_frames=2\nrotate_keep=2\nrotate_dir=' + json.dumps(str(root / "rotate")) + '\n[[alert]]\nname="workloads"\nkind="proc"\nquery="cmd:' + marker + '"\nthreshold=3\nhysteresis=2\ncommand=' + json.dumps('echo "$CRABMON_STATE" >> ' + str(root / "hooks")) + '\ncommand_clear=' + json.dumps('echo "$CRABMON_STATE" >> ' + str(root / "hooks")) + '\n')
    for fmt in ["json", "csv", "text"]:
        out = subprocess.check_output([binary, "--once", "--refresh", "200", "--filter", "cmd:" + marker + " !crabmon", "--format", fmt, "--config", str(cfg)], text=True)
        assert secret not in out, "credential leaked in " + fmt
        if fmt == "json":
            rows = json.loads(out)["procs"]
            assert len(rows) == 3 and all("[REDACTED]" in p["cmd"] for p in rows), "expected three redacted workload rows; got " + str(len(rows))
    raw = root / "raw.toml"
    raw.write_text('[export]\nredact_commands=false\n')
    out = subprocess.check_output([binary, "--once", "--refresh", "200", "--filter", "cmd:" + marker + " !crabmon", "--config", str(raw)], text=True)
    assert secret in out, "opt-out did not retain the raw command"
    with (root / "stdout.jsonl").open("w") as output, (root / "stderr.log").open("w") as errors:
        stream = subprocess.Popen([binary, "--stream", "--alerts", "--record", str(root / "record.jsonl"), "--config", str(cfg)], stdout=output, stderr=errors)
        wait_for(lambda: hooks() == ["firing"], "alert never fired")
        for child in children[:2]:
            child.terminate(); child.wait(timeout=5)
            time.sleep(.7)
            assert hooks() == ["firing"], "hysteresis cleared inside margin"
        children[-1].terminate(); children[-1].wait(timeout=5)
        wait_for(lambda: hooks() == ["firing", "cleared"], "recovery hook never ran")
        time.sleep(.5)
        stream.terminate(); stream.wait(timeout=5); stream = None
    files = list(root.glob("*.jsonl")) + list((root / "rotate").glob("*.jsonl")) + list(root.glob("*.md"))
    assert len(files) >= 5, "recording/rotation/flight evidence missing"
    for path in files:
        assert secret not in path.read_text(), "credential leaked to " + path.name
    frames = [json.loads(line) for line in (root / "record.jsonl").read_text().splitlines()]
    assert any(p["pid"] in [c.pid for c in children] and "[REDACTED]" in p["cmd"] for frame in frames for p in frame["procs"])
    # A private wrapper lets the collector exercise failures without disrupting SSH.
    wrapper = root / "sample.sh"
    wrapper.write_text('#!/bin/sh\n[ ! -e ' + str(root / "failed") + ' ] || exit 1\nif [ -e ' + str(root / "stalled") + ' ]; then sleep 7; fi\nexec ' + binary + ' --once --refresh 200 --config ' + str(raw) + '\n')
    print(json.dumps({"remote": "passed", "checks": ["exports", "redaction-opt-out", "hysteresis-hooks", "stream", "recording", "rotation", "flight-evidence"]}))
finally:
    if stream is not None:
        stream.terminate(); stream.wait(timeout=5)
    for child in children:
        if child.poll() is None:
            child.terminate(); child.wait(timeout=5)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--binary", type=Path, default=Path("target/release/crabmon"))
    args = parser.parse_args()
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "--", args.target]

    def remote(command, data=None):
        result = subprocess.run(ssh + [command], input=data, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=90)
        if result.returncode:
            raise RuntimeError("remote check failed: " + result.stderr.decode(errors="replace"))
        return result.stdout

    root = remote("mktemp -d /tmp/crabmon-e2e.XXXXXXXX").decode().strip()
    exporter = None
    try:
        remote("cat > " + shlex.quote(root + "/crabmon"), args.binary.read_bytes())
        remote("chmod 700 " + shlex.quote(root + "/crabmon"))
        remote("cat > " + shlex.quote(root + "/test.py"), REMOTE_TEST.encode())
        print(remote("python3 " + shlex.quote(root + "/test.py") + " " + shlex.quote(root)).decode().strip())
        with tempfile.TemporaryDirectory(prefix="crabmon-ssh-") as local:
            cfg = Path(local) / "config.toml"
            cfg.write_text('refresh_ms=200\n')
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            addr = "127.0.0.1:" + str(port)
            with (Path(local) / "exporter.log").open("w") as log:
                exporter = subprocess.Popen([str(args.binary.resolve()), "--serve", addr,
                    "--remote", args.target, "--remote-command", "sh " + shlex.quote(root + "/sample.sh"),
                    "--config", str(cfg)], stdout=subprocess.DEVNULL, stderr=log)
                def scrape():
                    try:
                        with urllib.request.urlopen("http://" + addr + "/metrics", timeout=3) as response:
                            return response.read().decode()
                    except OSError:
                        return ""
                label = '{host="' + args.target + '"}'
                def value(body, name):
                    prefix = "crabmon_" + name + label + " "
                    return next((float(line[len(prefix):]) for line in body.splitlines() if line.startswith(prefix)), None)
                def wait_metric(name, expected, seconds=25):
                    end = time.monotonic() + seconds
                    while time.monotonic() < end:
                        body = scrape()
                        if value(body, name) == expected:
                            return body
                        assert exporter.poll() is None, "exporter exited"
                        time.sleep(.2)
                    raise AssertionError(name + " did not become " + str(expected))
                wait_metric("sample_stale", 0)
                remote("touch " + shlex.quote(root + "/stalled"))
                body = wait_metric("sample_stale", 1)
                assert value(body, "sample_age_seconds") >= 5
                assert value(body, "up") == 0
                remote("rm " + shlex.quote(root + "/stalled"))
                wait_metric("sample_stale", 0)
                remote("touch " + shlex.quote(root + "/failed"))
                body = wait_metric("up", 0)
                assert value(body, "sample_stale") == 1
                remote("rm " + shlex.quote(root + "/failed"))
                wait_metric("up", 1)
                print(json.dumps({"ssh_collector": "passed", "checks": ["custom-command", "sample-age", "stalled-host", "failed-host", "recovery"]}))
    finally:
        if exporter is not None:
            exporter.terminate()
            try:
                exporter.wait(timeout=10)
            except subprocess.TimeoutExpired:
                exporter.kill(); exporter.wait()
        remote("rm -rf -- " + shlex.quote(root))


if __name__ == "__main__":
    main()
