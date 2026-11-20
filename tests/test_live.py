"""CPU checks for the long-lived driver: failures keep the engine, reload keeps worker state, quit exits."""
import json
import os
import signal
import threading
import time
import types

from harness import live, live_state


class FakeEngine:
    def __init__(self):
        self.output_processor = types.SimpleNamespace(request_states={})
        self.steps, self.aborted = 0, []

    def step(self):
        time.sleep(0.01)
        self.steps += 1

    def abort_request(self, ids, internal=False):
        self.aborted += ids
        for i in ids:
            self.output_processor.request_states.pop(i)


class FakeLLM:
    def __init__(self):
        self.llm_engine = FakeEngine()

    def collective_rpc(self, fn, args=()):
        return [fn(None, *args)]  # one "worker", in-process


def _run(tmp_path, cmds, engine=True, mode="eager"):
    args = types.SimpleNamespace(model="/m", out=tmp_path, tp=1, max_model_len=4096, max_num_seqs=4)
    lv = live.Live(args, tmp_path / "live")
    live_state.LLM, live_state.DESC, live_state.MODE = (FakeLLM(), "fake", mode) if engine else (None, "", "")
    if engine:
        live.install_step_guard(live_state.LLM)
    signal.signal(signal.SIGINT, live._on_sigint)
    for i, c in enumerate(cmds):
        (lv.root / "inbox" / f"{i:03d}.cmd").write_text(c)
    lv.loop(poll_s=0.01)
    return lv, [json.loads((lv.root / "done" / f"{i:03d}.json").read_text()) for i in range(len(cmds))]


def test_failures_keep_engine_and_reload_keeps_worker_state(tmp_path):
    from harness import tp

    tp._ORIG["w"] = "original weights"
    try:
        _check(tmp_path)
    finally:
        import harness.tp as tp_now

        tp._ORIG.pop("w", None)
        tp_now._ORIG.pop("w", None)


def _check(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text("open(args.out / 'probe_ran', 'w').write(str(llm is not None))\n")
    engine = None
    lv, res = _run(tmp_path, [
        "stage refusal --harmful nope.jsonl --n-per-class 1",  # fails before touching the engine
        "py " + str(probe),
        "reload",
        "bogus",
        "quit",
    ])
    engine = live_state.LLM
    assert [r["ok"] for r in res] == [False, True, True, False, True]
    assert "nope.jsonl" in res[0]["error"] and "Traceback" in res[0]["traceback"]
    assert (tmp_path / "probe_ran").read_text() == "True"
    import harness.tp as tp2

    assert tp2._ORIG.get("w") == "original weights"  # survived the worker reload
    assert isinstance(engine, FakeLLM)  # engine never dropped
    assert json.loads((lv.root / "status.json").read_text())["state"] == "quit"


def test_sigint_stops_generation_at_a_step_boundary_and_aborts(tmp_path):
    gen = tmp_path / "gen.py"  # stands in for llm.generate: queued requests, engine steps until done
    gen.write_text("eng = llm.llm_engine\neng.output_processor.request_states.update({'r1': 1, 'r2': 2})\n"
                   "for _ in range(1000): eng.step()\n")
    threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    t = time.time()
    _, res = _run(tmp_path, ["py " + str(gen), "quit"])
    eng = live_state.LLM.llm_engine
    assert not res[0]["ok"] and "step boundary (2 requests aborted)" in res[0]["error"]
    assert not res[0]["hard_interrupt"] and sorted(eng.aborted) == ["r1", "r2"] and res[1]["ok"]
    assert time.time() - t < 4


def test_second_sigint_interrupts_outside_generation(tmp_path):
    slow = tmp_path / "slow.py"  # no engine steps (e.g. judging): first SIGINT only sets the flag
    slow.write_text("import time\nfor _ in range(500): time.sleep(0.01)\n")
    for d in (0.4, 0.8):
        threading.Timer(d, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    t = time.time()
    _, res = _run(tmp_path, ["py " + str(slow), "quit"])
    assert not res[0]["ok"] and "KeyboardInterrupt" in res[0]["error"] and res[0]["hard_interrupt"]
    assert time.time() - t < 4


def test_failure_inside_generate_aborts_leftover_requests(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("llm.llm_engine.output_processor.request_states['r9'] = 1\nraise RuntimeError('boom')\n")
    _, res = _run(tmp_path, ["py " + str(bad), "quit"])
    assert not res[0]["ok"] and res[0]["aborted_requests"] == 1
    assert live_state.LLM.llm_engine.output_processor.request_states == {}


def test_refusal_capture_refused_on_a_prod_engine(tmp_path):
    _, res = _run(tmp_path, ["stage refusal --n-per-class 1", "stage refusal --phase capture", "quit"], mode="prod")
    assert "engine mode 'prod' cannot" in res[0]["error"]
    assert "engine mode 'prod' cannot" in res[1]["error"]


def test_engine_modes():
    from harness import session

    a = session.build_parser().parse_args(["refusal", "--model", "m", "--out", "o"])
    e, d, p = (session.engine_kwargs(a, m) for m in ("eager", "decode-graphs", "prod"))
    assert e["eager"] and e["max_num_batched_tokens"] == a.max_model_len
    assert not d["eager"] and d["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert d["max_num_batched_tokens"] == a.max_model_len
    assert "eager" not in p and "max_num_batched_tokens" not in p and "compilation_config" not in p
