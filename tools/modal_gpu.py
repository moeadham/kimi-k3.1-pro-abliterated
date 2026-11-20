"""Interactive GPU sandbox on the session image + Volume, driven step by step from the Mac.

  M="uv run --no-project --with modal --env-file .env python tools/modal_gpu.py"
  $M start --gpu B300:2 --minutes 60 --cpu 16 --mem-gib 128   # prints sandbox id, saves it to .modal_gpu_sb
  $M exec "nvidia-smi" [--timeout 900]                          # streams output; exit code = command's
  $M put tools/foo.py /root/harness/tools/foo.py                # push a local file into the sandbox
  $M stop
  $M oneshot "python tests_gpu/full_smoke.py ..." --gpu B300:8 --minutes 45 --name full_smoke   # runs, then exits
  $M fanout --job analysis="python tools/analyze_refusal.py /vol/session/refusal /vol/session/analysis_refusal" \
            --job directions="python -m harness.session directions --model unused --out /vol/session \
                              --capture-from refusal --label-by both"      # parallel CPU sandboxes (default --gpu none)

Debuggable runs (engine loaded once, survives failures; see harness/live.py):
  $M start --gpu B300:8 --minutes 240 --cpu 32 --mem-gib 512
  $M live-start                                  # detached `python -m harness.live` (loads the engine)
  $M live-status                                 # status.json + log tail
  $M live "stage refusal" [--wait 540]           # queue a command, stream its log until done or --wait s
  $M put harness/refusal.py /root/harness/harness/refusal.py && $M live reload   # hot-fix
  $M live-interrupt                              # SIGINT: abort the running command, keep the engine

The sandbox's own `timeout` is a hard cap: Modal kills it at --minutes regardless of this script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modal_image as mi  # noqa: E402

STATE = Path(__file__).resolve().parent.parent / ".modal_gpu_sb"


def _secret() -> modal.Secret:
    """HF_TOKEN (+ ALLMODELS_API_KEY for the Jev judge, if set) as an ephemeral, per-sandbox secret."""
    env = {k: os.environ[k] for k in ("HF_TOKEN", "ALLMODELS_API_KEY") if os.environ.get(k)}
    return modal.Secret.from_dict(env)


def _sb() -> modal.Sandbox:
    return modal.Sandbox.from_id(STATE.read_text().split()[0])


def start(a) -> None:
    app = modal.App.lookup(mi.APP, create_if_missing=True)
    t = time.time()
    sb = modal.Sandbox.create(
        "sleep", "infinity", app=app, image=mi.image(), gpu=a.gpu,
        cpu=a.cpu, memory=a.mem_gib * 1024, timeout=a.minutes * 60,
        volumes={mi.MOUNT: mi.volume()},
        secrets=[_secret()],
    )
    STATE.write_text(f"{sb.object_id} {int(time.time())} {a.minutes}")
    p = sb.exec("nvidia-smi", "-L")  # blocks until the container is actually running
    p.wait()
    print(f"sandbox {sb.object_id} running after {time.time() - t:.0f}s; hard stop in {a.minutes} min")
    print(p.stdout.read())


CPU_MEM_MAX_GIB = 336  # Modal's limit for a sandbox without GPU (344064 MiB), seen 2026-11-19


def _oneshot_sb(app, name: str, cmd: str, gpu: str, cpu: float, mem_gib: int, minutes: int) -> modal.Sandbox:
    """Sandbox whose entrypoint IS the command, output teed to /vol/logs/<name>.log."""
    if gpu.lower() == "none" and mem_gib > CPU_MEM_MAX_GIB:
        sys.exit(f"--mem-gib {mem_gib} > {CPU_MEM_MAX_GIB}: Modal rejects CPU-only sandboxes above {CPU_MEM_MAX_GIB} GiB")
    log = f"{mi.MOUNT}/logs/{name}.log"
    return modal.Sandbox.create(
        "bash", "-c", f"cd /root/harness && ( {cmd} ) 2>&1 | tee {log}; rc=${{PIPESTATUS[0]}}; sync {mi.MOUNT}; exit $rc",
        app=app, image=mi.image(), gpu=None if gpu.lower() == "none" else gpu, cpu=cpu,
        memory=mem_gib * 1024, timeout=minutes * 60,
        volumes={mi.MOUNT: mi.volume()},
        secrets=[_secret()],
    )


def oneshot(a) -> None:
    """Sandbox whose entrypoint IS the command: GPUs are released the moment it exits (no idle shell).
    Output is teed to /vol/logs/<name>.log so it survives a dropped client."""
    app = modal.App.lookup(mi.APP, create_if_missing=True)
    t = time.time()
    sb = _oneshot_sb(app, a.name, a.cmd, a.gpu, a.cpu, a.mem_gib, a.minutes)
    # not written to STATE: that file names the interactive sandbox (start/exec/put/live/stop), and a oneshot
    # running beside it must not redirect those commands
    print(f"[{time.strftime('%H:%M:%S')}] oneshot {sb.object_id} ({a.gpu}, hard stop {a.minutes} min); "
          f"log {mi.MOUNT}/logs/{a.name}.log", flush=True)
    for chunk in sb.stdout:
        print(chunk, end="", flush=True)
    sb.wait()
    print(f"[{time.strftime('%H:%M:%S')}] exit {sb.returncode} after {(time.time() - t) / 60:.1f} min (incl. queue)")


def fanout(a) -> None:
    """Independent jobs in parallel, one oneshot sandbox each (same --gpu/--cpu/--mem-gib/--minutes).
    Jobs are NAME=CMD; output lines are prefixed [NAME] and teed to /vol/logs/NAME.log. Waits for all and
    exits non-zero if any job failed; Ctrl-C terminates every sandbox. Jobs must not write the same outputs."""
    import threading

    jobs = []
    for j in a.job:
        name, sep, cmd = j.partition("=")
        if not sep or not name or not cmd:
            sys.exit(f"--job must be NAME=CMD, got {j!r}")
        jobs.append((name, cmd))
    if len({n for n, _ in jobs}) != len(jobs):
        sys.exit("job names must be unique (they name the log files)")
    app = modal.App.lookup(mi.APP, create_if_missing=True)
    t = time.time()
    sbs = {n: _oneshot_sb(app, n, c, a.gpu, a.cpu, a.mem_gib, a.minutes) for n, c in jobs}
    for n, sb in sbs.items():
        print(f"[{time.strftime('%H:%M:%S')}] [{n}] {sb.object_id} ({a.gpu}, hard stop {a.minutes} min)", flush=True)
    lock = threading.Lock()

    def stream(n, sb):
        buf = ""
        for chunk in sb.stdout:
            buf += chunk
            *lines, buf = buf.split("\n")
            with lock:
                for ln in lines:
                    print(f"[{n}] {ln}", flush=True)
        sb.wait()
        with lock:
            if buf:
                print(f"[{n}] {buf}", flush=True)
            print(f"[{time.strftime('%H:%M:%S')}] [{n}] exit {sb.returncode} after {(time.time() - t) / 60:.1f} min",
                  flush=True)

    threads = [threading.Thread(target=stream, args=(n, sb), daemon=True) for n, sb in sbs.items()]
    for th in threads:
        th.start()
    try:
        while any(th.is_alive() for th in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        for n, sb in sbs.items():
            sb.terminate()
            print(f"terminated [{n}] {sb.object_id}", flush=True)
        sys.exit(130)
    failed = [n for n, sb in sbs.items() if sb.returncode != 0]
    print(f"[{time.strftime('%H:%M:%S')}] all {len(jobs)} done in {(time.time() - t) / 60:.1f} min; "
          f"failed: {failed or 'none'}", flush=True)
    sys.exit(1 if failed else 0)


def run(a) -> None:
    sb = _sb()
    sid, t0, mins = STATE.read_text().split()
    left = int(t0) + int(mins) * 60 - time.time()
    print(f"[{time.strftime('%H:%M:%S')}] {left / 60:.1f} min left on {sid}", flush=True)
    # bytes mode: log lines cut mid-UTF-8 character must not kill the stream
    p = sb.exec("bash", "-c", f"cd /root/harness && {a.cmd}", timeout=a.timeout, text=False)
    for chunk in p.stdout:
        print(chunk.decode("utf-8", "replace"), end="", flush=True)
    p.wait()
    err = p.stderr.read().decode("utf-8", "replace")
    if err:
        print(err[-6000:], file=sys.stderr)
    print(f"[exit {p.returncode}]")
    sys.exit(p.returncode or 0)


def put(a) -> None:
    data = Path(a.src).read_bytes()
    p = _sb().exec("bash", "-c", f"mkdir -p $(dirname {a.dst}) && cat > {a.dst}")
    p.stdin.write(data)
    p.stdin.write_eof()
    p.stdin.drain()
    p.wait()
    print(f"put {a.src} -> {a.dst} ({len(data)} B) exit {p.returncode}")


def stop(a) -> None:
    sb = _sb()
    sb.terminate()
    print(f"terminated {sb.object_id}; poll={sb.poll()}")


LIVE_ROOT = os.environ.get("K3_LIVE_OUT", "/vol/session") + "/live"  # = harness.live --out + /live
LIVE_LOG = "/vol/logs/live.log"


def _sh(cmd: str, timeout: int = 120) -> tuple[int, str]:
    p = _sb().exec("bash", "-c", cmd, timeout=timeout, text=False)
    out = p.stdout.read().decode("utf-8", "replace")
    p.wait()
    return p.returncode, out


def live_start(a) -> None:
    cmd = (f"cd /root/harness && mkdir -p /vol/logs {a.out}/live && nohup python -m harness.live --model {a.model} "
           f"--out {a.out} --tp {a.tp} --engine-mode {a.engine_mode} {'--no-engine' if a.no_engine else ''} >> {LIVE_LOG} 2>&1 < /dev/null & "
           f"sleep 5; cat {a.out}/live/pid; tail -n 20 {LIVE_LOG}")
    print(_sh(cmd)[1])


def live_status(a) -> None:
    print(_sh(f"cat {LIVE_ROOT}/status.json; echo; ls {LIVE_ROOT}/inbox; tail -n {a.lines} {LIVE_LOG}; "
              f"nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader")[1])


def live_interrupt(a) -> None:
    print(_sh(f"kill -INT $(cat {LIVE_ROOT}/pid) && echo sent SIGINT to $(cat {LIVE_ROOT}/pid)")[1])


def live_cmd(a) -> None:
    """Queue one command for harness.live and stream the log until its result appears (or --wait s pass)."""
    name = time.strftime("%Y%m%d-%H%M%S")
    rc, off = _sh(f"stat -c %s {LIVE_LOG} 2>/dev/null || echo 0")
    off = int(off.strip() or 0)
    p = _sb().exec("bash", "-c", f"cat > {LIVE_ROOT}/inbox/{name}.tmp && mv {LIVE_ROOT}/inbox/{name}.tmp "
                                 f"{LIVE_ROOT}/inbox/{name}.cmd")
    p.stdin.write(a.cmd.encode())
    p.stdin.write_eof()
    p.stdin.drain()
    p.wait()
    print(f"[{time.strftime('%H:%M:%S')}] queued {name}: {a.cmd}", flush=True)
    t0 = time.time()
    while True:
        _, out = _sh(f"tail -c +{off + 1} {LIVE_LOG} | head -c 400000; echo; echo @@OFF $(stat -c %s {LIVE_LOG}); "
                     f"cat {LIVE_ROOT}/done/{name}.json 2>/dev/null")
        body, _, tail = out.partition("@@OFF ")
        new_off, _, result = tail.partition("\n")
        if body.strip():
            print(body.rstrip(), flush=True)
        off = int(new_off.strip() or off)
        if result.strip():
            r = json.loads(result)
            print(f"[{time.strftime('%H:%M:%S')}] {name} {'OK' if r['ok'] else 'FAILED: ' + r['error']} "
                  f"({r.get('seconds')}s)")
            sys.exit(0 if r["ok"] else 1)
        if time.time() - t0 > a.wait:
            print(f"[{time.strftime('%H:%M:%S')}] still running after {a.wait}s; check with live-status "
                  f"(result will be {LIVE_ROOT}/done/{name}.json)")
            if a.strict:  # scripts: a command that has not finished is not a success
                sys.exit(2)
            return
        time.sleep(a.poll)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd_name", required=True)
    s = sp.add_parser("start"); s.add_argument("--gpu", default="B300:2"); s.add_argument("--minutes", type=int, default=60)
    s.add_argument("--cpu", type=float, default=16); s.add_argument("--mem-gib", type=int, default=128); s.set_defaults(f=start)
    s = sp.add_parser("oneshot"); s.add_argument("cmd"); s.add_argument("--name", default="oneshot")
    s.add_argument("--gpu", default="B300:8"); s.add_argument("--minutes", type=int, default=45)
    s.add_argument("--cpu", type=float, default=32); s.add_argument("--mem-gib", type=int, default=512); s.set_defaults(f=oneshot)
    s = sp.add_parser("fanout"); s.add_argument("--job", action="append", required=True, help="NAME=CMD (repeat)")
    s.add_argument("--gpu", default="none"); s.add_argument("--minutes", type=int, default=60)
    s.add_argument("--cpu", type=float, default=16); s.add_argument("--mem-gib", type=int, default=64)
    s.set_defaults(f=fanout)
    s = sp.add_parser("exec"); s.add_argument("cmd"); s.add_argument("--timeout", type=int, default=1200); s.set_defaults(f=run)
    s = sp.add_parser("put"); s.add_argument("src"); s.add_argument("dst"); s.set_defaults(f=put)
    s = sp.add_parser("stop"); s.set_defaults(f=stop)
    s = sp.add_parser("live-start"); s.add_argument("--model", default="/vol/Kimi-K3.1-Pro")
    s.add_argument("--out", default=LIVE_ROOT[: -len("/live")]); s.add_argument("--tp", type=int, default=8)
    s.add_argument("--engine-mode", default="eager", choices=["eager", "decode-graphs", "prod"])
    s.add_argument("--no-engine", action="store_true"); s.set_defaults(f=live_start)
    s = sp.add_parser("live-status"); s.add_argument("--lines", type=int, default=40); s.set_defaults(f=live_status)
    s = sp.add_parser("live-interrupt"); s.set_defaults(f=live_interrupt)
    s = sp.add_parser("live"); s.add_argument("cmd"); s.add_argument("--wait", type=int, default=540)
    s.add_argument("--strict", action="store_true", help="exit 2 if the command is still running after --wait")
    s.add_argument("--poll", type=float, default=10); s.set_defaults(f=live_cmd)
    a = ap.parse_args()
    a.f(a)
