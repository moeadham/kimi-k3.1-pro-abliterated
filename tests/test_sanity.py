import json

from harness import sanity
from tools.compare_sanity import main as compare_main


def _res(acc, rep=0.05, kl=None):
    r = {"mmlu_pro": {"accuracy": acc, "health": {"hit_max_tokens": 0.0, "repetition": rep, "reasoning_closed": 1.0}},
         "bfcl": {"accuracy": 0.88, "health": {"hit_max_tokens": 0.0, "repetition": 0.01}}}
    if kl is not None:
        r["kl"] = {"kl_per_token": kl}
    return r


def test_compare_pass_and_fail():
    assert sanity.compare(_res(0.92), _res(0.90, kl=0.03))["pass"]
    r = sanity.compare(_res(0.92), _res(0.80, kl=0.03))
    assert not r["pass"] and [c["check"] for c in r["checks"] if not c["pass"]] == ["mmlu_pro.accuracy_drop"]
    assert not sanity.compare(_res(0.92), _res(0.92, kl=0.5))["pass"]
    assert not sanity.compare(_res(0.92), _res(0.92, rep=0.3))["pass"]


def test_panel_items_match_hosted_reference():
    ids = [it["id"] for _, it in sanity.panel_items()]
    ref = [json.loads(l)["item_id"] for l in open("results/sanity_panel_hosted/responses.jsonl")]
    assert ids == ref and len(ids) == 350


def test_compare_cli_on_hosted_reference(tmp_path, capsys):
    ref = "results/sanity_panel_hosted/summary.json"
    assert compare_main([ref, ref]) == 0  # a model compared with itself passes
    worse = json.loads(open(ref).read())
    worse["sets"]["ifeval"]["accuracy"] -= 0.2
    p = tmp_path / "worse.json"
    p.write_text(json.dumps(worse))
    assert compare_main([ref, str(p)]) == 1
    assert "FAIL  ifeval.accuracy_drop" in capsys.readouterr().out
