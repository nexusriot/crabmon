#!/usr/bin/env python3
"""Opt-in real-host checks. Run with --target user@host and --binary path.
Only creates a private remote /tmp directory and temporary test processes.
--units additionally needs root on the target: it starts transient systemd
units (stopped and reset afterwards) to check failure and crash-loop
detection end to end.
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


UNITS_TEST = r'''
import json, pathlib, subprocess, sys, time
root = pathlib.Path(sys.argv[1])
binary = str(root / "crabmon")
tag = sys.argv[2]
loop, fail = "cm-e2e-loop-" + tag, "cm-e2e-fail-" + tag

def run(*args, **kw):
    return subprocess.run(list(args), check=True, text=True, capture_output=True, **kw).stdout

try:
    run("systemd-run", "--quiet", "--unit=" + loop, "-p", "Restart=always", "-p", "RestartSec=1",
        "-p", "StartLimitIntervalSec=0", "/bin/sh", "-c", "sleep 0.2; exit 1")
    run("systemd-run", "--quiet", "--unit=" + fail, "/bin/sh", "-c", "exit 3")
    time.sleep(2)
    units = {u["name"]: u for u in json.loads(run(binary, "--once"))["units"]}
    assert units[fail + ".service"]["state"] == "failed", units
    assert units[fail + ".service"]["exit_status"] == 3, units
    assert units[loop + ".service"]["state"] in ("restarting", "crashloop"), units
    # The loop is promoted once crabmon has watched it restart three times.
    out = subprocess.run([binary, "--watch-rule", "unit:crashloop:" + loop + ">=1",
                          "--watch-timeout", "60", "--format", "text"],
                         text=True, capture_output=True, timeout=90)
    assert out.returncode == 1, "crash loop never detected: " + out.stdout + out.stderr
    # A unit alert names the units it is about, and recovers when they do.
    hooks = root / "unit-hooks"
    cfg = root / "units.toml"
    cfg.write_text('refresh_ms=500\n[[alert]]\nname="failed"\nkind="unit"\ntarget="failed"\nquery=' + json.dumps(fail)
        + '\nthreshold=1\ncommand=' + json.dumps('echo "$CRABMON_STATE|$CRABMON_MESSAGE" >> ' + str(hooks))
        + '\ncommand_clear=' + json.dumps('echo "$CRABMON_STATE|$CRABMON_MESSAGE" >> ' + str(hooks)) + '\n')
    stream = subprocess.Popen([binary, "--stream", "--alerts", "--config", str(cfg)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        end = time.monotonic() + 30
        while time.monotonic() < end and not (hooks.exists() and "firing" in hooks.read_text()):
            time.sleep(.2)
        assert fail in hooks.read_text(), "the firing hook does not name the unit"
        run("systemctl", "reset-failed", fail + ".service")
        end = time.monotonic() + 30
        while time.monotonic() < end and "cleared" not in hooks.read_text():
            time.sleep(.2)
        assert "cleared" in hooks.read_text(), "no recovery once the unit was reset"
    finally:
        stream.terminate(); stream.wait(timeout=5)
    # An OOM kill at a unit's own MemoryMax: the unit is failed with the
    # reason, `oomkill:<unit>` counts it although the cgroup is gone by the
    # next sample, and the kernel log yields one kill, not a kill and an oops.
    oom = "cm-e2e-oom-" + tag
    seqs = {e["seq"] for e in json.loads(run(binary, "--once"))["kernel"]}
    run("systemd-run", "--quiet", "--unit=" + oom, "-p", "MemoryMax=30M", "-p", "MemorySwapMax=0",
        "python3", "-c", "a = bytearray(200 << 20)")
    end = time.monotonic() + 20
    while time.monotonic() < end and "failed" not in run("systemctl", "show", "-p", "ActiveState", oom + ".service"):
        time.sleep(.2)
    snap = json.loads(run(binary, "--once"))
    unit = next(u for u in snap["units"] if u["name"] == oom + ".service")
    assert (unit["state"], unit["result"]) == ("failed", "oom-kill"), unit
    out = subprocess.run([binary, "--watch-rule", "oomkill:" + oom + ">=1", "--watch-timeout", "10"],
                         text=True, capture_output=True, timeout=30)
    assert out.returncode == 1, "oomkill:<unit> missed the kill: " + out.stdout + out.stderr
    if snap.get("kernel_unreadable") is None:
        new = [e for e in snap["kernel"] if e["seq"] not in seqs]
        assert [e["kind"] for e in new] == ["oomkill"], new
    print(json.dumps({"units": "passed", "checks": ["failed", "exit-status", "crash-loop", "named-in-alert", "recovery", "oom-kill"]}))
finally:
    for unit in (loop, fail, "cm-e2e-oom-" + tag):
        subprocess.run(["systemctl", "stop", unit + ".service"], capture_output=True)
        subprocess.run(["systemctl", "reset-failed", unit + ".service"], capture_output=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--binary", type=Path, default=Path("target/release/crabmon"))
    parser.add_argument("--units", action="store_true",
                        help="also check systemd unit detection (needs root on the target)")
    args = parser.parse_args()
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "--", args.target]

    def remote(command, data=None):
        result = subprocess.run(ssh + [command], input=data, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=180)
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
        if args.units:
            remote("cat > " + shlex.quote(root + "/units.py"), UNITS_TEST.encode())
            tag = root.rsplit(".", 1)[-1].lower()
            print(remote("python3 " + shlex.quote(root + "/units.py") + " " + shlex.quote(root)
                         + " " + shlex.quote(tag)).decode().strip())
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

            # A host that has never answered exports that it is not up, and
            # no zeros dressed as measurements: it once claimed no memory, no
            # load and no failed services.
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            addr = "127.0.0.1:" + str(port)
            never = subprocess.Popen([str(args.binary.resolve()), "--serve", addr,
                "--remote", args.target, "--remote-command", "false", "--config", str(cfg)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                time.sleep(6)
                body = scrape()
                series = sorted({line.split("{")[0] for line in body.splitlines()
                                 if line.startswith("crabmon_")})
                assert series == ["crabmon_sample_stale", "crabmon_up"], series
                assert value(body, "up") == 0, body
            finally:
                never.terminate(); never.wait(timeout=10)
            print(json.dumps({"never_sampled": "passed", "checks": ["up-only"]}))
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
