"""Long-lived session driver: load the engine ONCE, then run commands against it until told to quit.

Loading K3.1 at TP8 costs up to ~50 min (Volume read speed), so a failure in our code must not cost a
reload. This process keeps the engine in memory (harness.live_state, never reloaded) and takes commands
from files in OUT/live/inbox/. A failing command is logged with its traceback and the process goes
back to waiting, with the engine intact. Fixed code is pushed into the sandbox (tools/modal_gpu.py put)
and picked up with `reload`, on the driver AND on every TP worker.

  python -m harness.live --model /vol/Kimi-K3.1-Pro --out /vol/session --tp 8     (start detached)

Commands: one per file OUT/live/inbox/<name>.cmd, run in name order (tools/modal_gpu.py live "<cmd>"):
  stage <session argv>   e.g. `stage refusal --n-per-class 20`; --model/--out/--tp are added.
                         Stages call session.make_engine, which returns the live engine.
  reload                 importlib.reload harness.* here (except live, live_state) and harness.tp /
                         harness.capture on every worker (tp's saved writer originals are kept)
  py <file>              exec a Python file with `llm`, `tok`, `session`, `args`, `log` in scope (probes)
  engine [mode]          (re)load the engine, optionally in another mode (eager | decode-graphs | prod):
                         in-process teardown, best effort; the sure way is `quit` + a new driver
  quit                   exit (the sandbox keeps running until stopped)
SIGINT to the pid in OUT/live/pid (tools/modal_gpu.py live-interrupt): the running generate stops at the next
engine step, its requests are aborted, the command fails, the engine stays usable. Outside generation (judging,
file I/O) the first SIGINT only sets the flag; a second one interrupts immediately, which can corrupt the engine
if it lands mid-forward (seen on 1xB300: MoE shared-experts assert afterwards) -> probe, then `engine`.
Results: OUT/live/done/<name>.json {cmd, ok, error, traceback, seconds}; state: OUT/live/status.json.
"""

from __future__ import annotations

import argparse
import gc
import os
import importlib
import json
import signal
import sys
import time
import traceback
from pathlib import Path

from harness import live_state

KEEP = {"harness.live", "harness.live_state", "harness.vllm_plugin"}
WORKER_MODULES = ("harness.capture", "harness.tp")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), "[live]", msg, flush=True)


def reload_keeping_tp_state(modules) -> list[str]:
    """Reload modules, carrying over tp's per-process state (saved writer originals, open capture)."""
    import harness.tp as tp_mod

    keep = (dict(tp_mod._ORIG), dict(tp_mod._CAP))
    done = []
    for name in modules:
        if name in sys.modules:
            importlib.reload(sys.modules[name])
            done.append(name)
    import harness.tp as tp_new

    tp_new._ORIG.update(keep[0])
    tp_new._CAP.update(keep[1])
    return done


def w_reload(worker, modules=WORKER_MODULES) -> list[str]:
    """Runs in each TP worker: reload the worker-side harness modules, keeping tp's per-worker state."""
    return reload_keeping_tp_state(modules)


_STOP = {"requested": False, "hard": False}


def _on_sigint(signum, frame) -> None:
    """First SIGINT: stop at the next engine-step boundary (safe). Second: interrupt immediately (may corrupt the
    engine: on the 1xB300 check an interrupt mid-forward broke the MoE shared-experts state -> `engine` reload)."""
    if _STOP["requested"]:
        _STOP["hard"] = True
        raise KeyboardInterrupt("hard interrupt (engine state may be corrupt: run a probe, `engine` if broken)")
    _STOP["requested"] = True
    log("interrupt requested: stopping at the next engine step (SIGINT again to force)")


def abort_unfinished(llm) -> int:
    """Abort every request still in the engine (left over after an exception inside llm.generate)."""
    eng = llm.llm_engine
    ids = list(eng.output_processor.request_states.keys())
    if ids:
        eng.abort_request(ids, internal=True)
    return len(ids)


def install_step_guard(llm) -> None:
    """Interrupts land only between engine steps: after a full step, if a stop was requested, abort all requests
    (so the engine is clean) and raise KeyboardInterrupt out of llm.generate."""
    eng = llm.llm_engine
    orig = eng.step

    def step(*a, **k):
        out = orig(*a, **k)
        if _STOP["requested"]:
            n = abort_unfinished(llm)
            _STOP["requested"] = False
            raise KeyboardInterrupt(f"interrupted at an engine step boundary ({n} requests aborted)")
        return out
    eng.step = step


class Live:
    def __init__(self, args, root: Path):
        self.args, self.root = args, root
        for d in ("inbox", "done"):
            (root / d).mkdir(parents=True, exist_ok=True)
        self.tok = None
        self.state = "starting"

    def status(self, **kw) -> None:
        st = {"state": self.state, "engine": live_state.DESC if live_state.LLM is not None else None,
              "time": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}
        (self.root / "status.json").write_text(json.dumps(st, indent=1))

    def base_argv(self) -> list[str]:
        a = self.args
        return ["--model", a.model, "--out", str(a.out), "--tp", str(a.tp), "--max-model-len", str(a.max_model_len),
                "--max-num-seqs", str(a.max_num_seqs)]

    def drop_engine(self) -> None:
        """Best-effort in-process teardown (engine core, TP workers, process groups, GPU memory). Unverified on
        TP8: if GPU memory is not freed, `quit` and start the driver again with --engine-mode instead."""
        llm = live_state.LLM
        if llm is None:
            return
        log(f"dropping the current engine ({live_state.DESC})")
        live_state.LLM, live_state.DESC, live_state.MODE = None, "", ""
        try:
            llm.llm_engine.engine_core.shutdown()
        except Exception as e:  # noqa: BLE001
            log(f"engine_core.shutdown: {e!r}")
        del llm
        gc.collect()
        try:
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

            cleanup_dist_env_and_memory()
        except Exception as e:  # noqa: BLE001
            log(f"cleanup_dist_env_and_memory: {e!r}")

    def load_engine(self, mode: str | None = None) -> None:
        from harness import session

        mode = mode or self.args.engine_mode
        self.drop_engine()
        eargs = session.build_parser().parse_args(["refusal", *self.base_argv()])
        self.state = f"loading engine ({mode})"
        self.status()
        t = time.time()
        llm = session.make_engine_mode(eargs, mode)
        install_step_guard(llm)
        live_state.LLM, live_state.MODE = llm, mode
        live_state.DESC = (f"tp={self.args.tp} {mode}"
                           f"{' capture-ready' if mode in session.CAPTURE_READY else ''}, loaded in {time.time() - t:.0f}s")
        log(f"engine ready: {live_state.DESC}")
        self.tok = session.tokenizer()

    def run(self, line: str) -> None:
        cmd, _, rest = line.strip().partition(" ")
        if cmd == "stage":
            from harness import session

            argv = rest.split()
            extra = [x for x in self.base_argv()[::2] if x not in argv]
            base = dict(zip(self.base_argv()[::2], self.base_argv()[1::2]))
            for k in extra:
                argv += [k, base[k]]
            session.main(argv)
        elif cmd == "reload":
            names = sorted(n for n in sys.modules if n.startswith("harness.") and n not in KEEP)
            reload_keeping_tp_state(names)  # TP=1 runs the worker in this process, so keep its state too
            log(f"reloaded here: {names}")
            if live_state.LLM is not None:
                log(f"reloaded on workers: {live_state.LLM.collective_rpc(w_reload)}")
        elif cmd == "py":
            from harness import session

            code = Path(rest.strip()).read_text()
            exec(compile(code, rest.strip(), "exec"), {"llm": live_state.LLM, "tok": self.tok, "session": session,
                                                       "args": self.args, "log": log, "__name__": "__live__"})
        elif cmd == "engine":  # `engine` reloads the current mode; `engine decode-graphs` swaps modes
            self.load_engine(rest.strip() or live_state.MODE or None)
        elif cmd == "quit":
            raise SystemExit(0)
        else:
            raise ValueError(f"unknown command {cmd!r}")

    def loop(self, poll_s: float = 2.0) -> None:
        while True:
            self.state = "idle"
            self.status()
            todo = sorted(p for p in (self.root / "inbox").glob("*.cmd"))
            if not todo:
                time.sleep(poll_s)
                continue
            f = todo[0]
            line = f.read_text().strip()
            f.unlink()
            self.state = f"running {f.stem}: {line}"
            self.status()
            log(f">>> {f.stem}: {line}")
            t, res = time.time(), {"cmd": line, "ok": True}
            _STOP.update(requested=False, hard=False)
            try:
                self.run(line)
            except SystemExit as e:
                if line.strip() == "quit":
                    (self.root / "done" / f"{f.stem}.json").write_text(json.dumps(res))
                    self.state = "quit"
                    self.status()
                    log("quit")
                    return
                res.update(ok=False, error=f"SystemExit({e.code})", traceback=traceback.format_exc())
            except BaseException as e:  # noqa: BLE001 -- keep the engine alive whatever our code does
                res.update(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
            if not res["ok"] and live_state.LLM is not None:
                try:  # a failure inside llm.generate leaves its requests queued: the next generate would run them
                    n = abort_unfinished(live_state.LLM)
                    res["aborted_requests"] = n
                    if n:
                        log(f"aborted {n} leftover requests")
                except Exception as e:  # noqa: BLE001
                    res["abort_error"] = repr(e)
            res["hard_interrupt"] = _STOP["hard"]
            _STOP.update(requested=False, hard=False)
            res["seconds"] = round(time.time() - t, 1)
            log(f"<<< {f.stem}: {'OK' if res['ok'] else 'FAILED ' + res['error']} ({res['seconds']}s)")
            if not res["ok"]:
                print(res["traceback"], flush=True)
            (self.root / "done" / f"{f.stem}.json").write_text(json.dumps(res, indent=1))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--engine-mode", default="eager", help="eager | decode-graphs | prod (harness.session)")
    ap.add_argument("--no-engine", action="store_true", help="start idle; load later with the `engine` command")
    args = ap.parse_args(argv)
    signal.signal(signal.SIGINT, _on_sigint)
    live = Live(args, args.out / "live")
    (live.root / "pid").write_text(str(os.getpid()))  # for `kill -INT`: interrupts the running command only
    if not args.no_engine:
        try:
            live.load_engine()
        except BaseException:  # noqa: BLE001 -- stay up so the failure can be fixed and `engine` retried
            live.state = "engine load FAILED"
            live.status(error=traceback.format_exc()[-4000:])
            log("engine load FAILED; waiting for commands (fix, then `engine`)\n" + traceback.format_exc())
    live.loop()


if __name__ == "__main__":
    main()
