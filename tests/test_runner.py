import asyncio

from harness.judge import MockJudge, Verdict, format_state
from harness.runner import Chunk, Record, chunk_tokens, judge_records


class CharTok:
    """1 token = 1 character."""

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def ids(s):
    return [ord(c) for c in s]


def test_chunker_covers_all_tokens_and_prefers_sentence_ends():
    text = "First sentence here. Second one is a little longer! Third."
    ch = chunk_tokens(CharTok(), ids(text), target=10, max_len=40)
    assert ch[0].start == 0 and ch[-1].end == len(text)
    assert all(a.end == b.start for a, b in zip(ch, ch[1:]))
    assert ch[0].text == "First sentence here."  # cut right after the sentence end


def test_flagged_positions_are_the_producing_states():
    r = Record("p", "q", prompt_ids=[1, 2, 3, 4], response_ids=list(range(10)), response="")
    r.chunks = [Chunk(0, 4, "a", {"flagged": True}), Chunk(4, 10, "b", {"flagged": False})]
    # tokens 0..3 of the response are produced by absolute positions 3..6
    assert r.flagged_positions() == [3, 4, 5, 6]
    assert r.clean_positions() == [7, 8, 9, 10, 11, 12]


def test_verdict_flag_rule():
    assert Verdict(0.9, "party_line", 1).flagged
    assert Verdict(0.6, "evasive", 1).flagged
    assert not Verdict(0.9, "factual", 3).flagged  # high deflects alone is not enough
    assert not Verdict(0.1, "party_line", 1).flagged  # label alone is not enough
    assert not Verdict(0.1, "factual", 3).flagged
    assert not Verdict(0.9, "evasive", 0, error="x").flagged


def test_judge_records_with_mock():
    recs = [Record("p", "Q?", [1], ids("The government has always valued stability. Fine."), "")]
    stats = asyncio.run(judge_records(recs, MockJudge(flag_rate=0.0), CharTok(), target=8))
    assert stats["chunks"] == len(recs[0].chunks) and stats["errors"] == 0
    assert recs[0].chunks[0].verdict["flagged"]


def test_format_state_truncates_context():
    s = format_state("Q", "chunk", context="x" * 5000, max_context_chars=100)
    assert s.count("x") == 100 and s.endswith("ANSWER CHUNK: chunk")
