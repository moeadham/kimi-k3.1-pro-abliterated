"""Build and freeze the English run-3 prompt corpus (data/refusal_v3/), spec: docs/run3_plan.md section 2.1.

Files (every row: id, prompt, split = harmful | harmless, kind, source; plus category / pair / pair_sim when known):

  build_harmful / build_harmless   same length, same kind order: row k of one file is row k's counterpart in the
                                   other (twins share `pair`; WildJailbreak rows are length-bin counterparts only)
      wjb_adversarial_*   WildJailbreak TRAIN, 1,000 + 1,000, length-matched (<= 300 words)
      wjb_vanilla_*       WildJailbreak TRAIN, 500 + 500, length-matched
      residual_<theme>    ~100 harmful per theme, mined by keyword from WildJailbreak train (adversarial + vanilla
                          harmful) and the enriched set; each gets a benign WildJailbreak twin of the same length
                          family by TF-IDF nearest neighbour (one-to-one), preferring benign rows on the same theme
      enriched_<frame>    Bahushruth/abliteration-harmful-enriched framed prompts not used in v1, 250 per frame; the
                          enriched set has no harmless side, so each gets a WildJailbreak vanilla_benign twin by TF-IDF
                          nearest neighbour, preferring the same framing
      v1_pairs_<source>   data/refusal_v1/pairs_* as-is (same ids, pair, category, score)
  reference_harmless      ~2,000 non-adversarial prompts for the projection reference (see the dataset doc for gaps)

Held out (decontaminated against, never sampled): HarmBench standard, XSTest safe, data/evals/*.jsonl,
data/refusal_v1/heldout_wjb_*.jsonl and the whole WildJailbreak eval split, mlabonne/harmful_behaviors test.
v1 borderline prompts are excluded from the new samples (so overlap with them is 0 by construction).
Decontamination, English filter, pinned revisions and OBLITERATUS parsing (ast.literal_eval, never executed) are
v1's (tools/build_refusal_v1.py).

Usage: uv run --env-file .env --with pyarrow --with pandas --with huggingface_hub python tools/build_refusal_v3.py [out_dir]
"""

from __future__ import annotations

import ast
import collections
import hashlib
import heapq
import json
import math
import random
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.build_refusal_v1 import (ADV_MAX_WORDS, FRAMES, JAILBREAK, OBLIT_COMMIT, OBLIT_URL, REV, SEMANTIC_MIN_SCORE,  # noqa: E402
                                    Heldout, hf, is_english, length_matched, norm, words)

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/refusal_v3"
SEED = 0
V1 = ROOT / "data/refusal_v1"

HELDOUT_FILES = ["results/harmbench_standard/prompts.jsonl", "results/xstest_safe/prompts.jsonl",
                 "data/evals/harmful.jsonl", "data/evals/kl_prompts.jsonl", "data/evals/mmlu_pro.jsonl",
                 "data/evals/cmmlu.jsonl", "data/evals/ifeval.jsonl", "data/evals/math500.jsonl",
                 "data/evals/bfcl.jsonl", "data/refusal_v1/heldout_wjb_harmful.jsonl",
                 "data/refusal_v1/heldout_wjb_benign.jsonl"]

N_WJB_ADV = 1000          # per side
N_WJB_VAN = 500           # per side
N_RESIDUAL_PER_THEME = 100
RESIDUAL_QUOTA = {"enriched": 20, "vanilla": 40}   # per theme; adversarial fills the rest
N_ENRICHED_PER_FRAME = 250
N_REFERENCE = 2000
ADV_BENIGN_POOL = 20000   # random adversarial_benign rows (plus all on-theme ones) searched for twins
THEME_BONUS = 0.10        # added to the cosine of a twin candidate on the same theme / framing
LEN_RATIO = {"vanilla": (0.6, 1.67), "adversarial": (0.67, 1.5)}  # twin words / query words

# ---------------------------------------------------------------- residual themes (lowercase regexes)
GROUP = (r"\b(immigrants?|immigration|refugees?|migrants?|asylum|ethnic\w*|racial\w*|races\b|religio\w*|muslims?|"
         r"islam\w*|jews?\b|jewish|christians?|hindus?|sikhs?|buddhists?|atheists?|black (people|men|women|individuals|"
         r"community|communities|americans?)|african\w*|asians?\b|latin[oa]s?\b|latinx|hispanics?|mexicans?|arabs?\b|"
         r"roma\b|gypsies|indigenous|native americans?|minorit\w*|chinese|indians?\b|middle eastern|nationalit\w*|"
         r"foreigners?|people of (a )?(certain )?(race|color|colour|religion|ethnicity|nationality))")
WRITE = (r"\b(article|essay|blog|speech|op-?ed|persuasive|story|post|narrative|argu\w*|piece|letter|tweets?|campaign|"
         r"portray\w*|depict\w*|pamphlet|manifesto|editorial|paragraph|poem|jokes?|script|dialogue|social media)")
ATROC = (r"\b(genocid\w*|holocaust|massacre\w*|atrocit\w*|nanking|nanjing|armenian|khmer rouge|rwanda\w*|slavery|"
         r"slave trade|apartheid|tiananmen|uyghurs?|uighurs?|xinjiang|famine|holodomor|ethnic cleansing|war crimes?|"
         r"internment|trail of tears|srebrenica|bosnian|cultural revolution|great leap|gulag|pol pot|stalin\w*|hitler|"
         r"nazi\w*|mao\b|colonial\w*|residential schools?|lynch\w*|jim crow|hiroshima|nagasaki|my lai|darfur|rohingya)")
BALANCE = (r"\b(balanced|both sides|two sides|other side|perspectives?|justif\w*|den(y|ial|ier|ied)\w*|exaggerat\w*|"
           r"hoax|revis\w*|positive (light|lens|aspects?|side|impact)|benefi\w*|necessary|nuanc\w*|alternative (view|"
           r"narrative|history|perspective)|controversial|never happened|myth|glorif\w*|downplay\w*|minimi[sz]\w*|"
           r"fabricat\w*|propaganda|debat\w*|objective|neutral|unbiased|impartial|fair)")
SEC = (r"(\bexploit\w*\W+(\w+\W+){0,3}(vulnerab|flaw|bug|cve|system|server|network|software|website|database|router)|"
       r"\bexploit (code|kit)|\bmalware|\bransomware|\bkeyloggers?|computer virus|\bviruses\b|\btrojans?\b|\brootkits?|"
       r"\bbotnets?|\bshellcode|\bpayloads?\b|\bzero.?days?\b|\bbackdoors?\b|\bddos\b|denial.of.service|sql injection|"
       r"\bxss\b|cross.site scripting|\bphishing|\bspyware|reverse shell|brute.?forc\w*|buffer overflow|"
       r"privilege escalation|\bhack\w*|\bcrack\w* (a |the )?(passwords?|wi.?fi|software|encryption)|"
       r"password stealer|cryptojack\w*|bypass\w* (the )?(antivirus|security|authentication|firewall|2fa))")
WRAP = (r"\b(educational|education|defensive|defen[sc]e|security research\w*|researchers?|penetration test\w*|"
        r"pen.?test\w*|awareness|training|red.?team\w*|ethical\w*|white.?hat|cybersecurity|cyber security|for testing|"
        r"simulat\w*|protect\w*|defend\w*|detect\w*|understand\w*|learn\w*|demonstrat\w*|hypothetical\w*|academic|"
        r"course|class|students?|professor)")
CODE = (r"\b(code|script|program\w*|write|create|develop|build|craft|generate|implement|step.by.step|how to|guide|"
        r"tutorial|instructions)")
BLACKMKT = (r"\b(black.?market\w*|dark ?web|darknet|smuggl\w*|traffick\w*|launder\w*|fenc(e|ing) (stolen|goods)|"
            r"stolen (goods|property|merchandise|cars?|art)|counterfeit\w*|"
            r"cover(ing)?.?up (a |an |the )?(crime|murder|killing|death|accident|abuse|assault|evidence|incident|"
            r"hit.and.run|bod(y|ies))|"
            r"hid(e|ing) (the |any )?(evidence|body|bodies|crime)|destroy\w* (the )?evidence|"
            r"dispose of (a |the )?(body|corpse|evidence)|untraceable|brib\w*|hit.?man|hire (a |an )?(assassin|killer)|"
            r"contacts? (for|of|with|to)|suppliers?\b|dealers?\b|connect me|get in touch with|"
            r"where (can|could|do|should) i (buy|get|find|obtain|purchase|source)|illegal (arms|weapons|guns|"
            r"firearms|drugs|organs?)|organ (trade|harvest\w*)|poach\w*|ivory)")
HAZ = (r"\b(nerve agents?|sarin|vx\b|tabun|soman|novichok|ricin|abrin|botulinum|anthrax|mustard gas|chlorine gas|"
       r"phosgene|chemical weapons?|chemical warfare|bio.?weapons?|biological weapons?|explosives?|tatp|petn|rdx|c-4|"
       r"nitroglycerin|thermite|napalm|pipe bombs?|bombs?|methamphetamine|meth\b|fentanyl|carfentanil|lsd|mdma|heroin|"
       r"cocaine|ghb|cyanide|toxins?|pathogens?|precursors?|synthesi[sz]\w*|synthesis|poisons?|poisonous|toxic gas|"
       r"radioactive|uranium|plutonium|dirty bomb)")
ACAD = (r"\b(academic\w*|research\w*|chemistry|chemist\w*|thesis|dissertation|professor|paper|scien\w*|"
        r"educational|students?|study|studies|studying|theor\w*|mechanism\w*|reactions?|class|course|textbook|"
        r"lecture|universit\w*|scholar\w*|curriculum|pharmacolog\w*|toxicolog\w*|molecul\w*|compound\w*)")
CHEM = (r"\b(chemi\w*|compound\w*|molecul\w*|reactions?|synthesi\w*|toxi\w*|explosive\w*|pharmac\w*|drugs?|"
        r"laborator\w*|lab\b|poison\w*|combust\w*|oxidi[sz]\w*|acids?|periodic table|catalys\w*)")

# harmful side: all patterns must match; benign twins prefer rows matching the benign patterns (same topic)
THEMES = {  # first match wins, most specific first
    "chemistry": {"harmful": [HAZ, ACAD], "benign": [CHEM, ACAD]},
    "exploit": {"harmful": [SEC, WRAP, CODE], "benign": [SEC]},
    "revisionism": {"harmful": [ATROC, BALANCE], "benign": [ATROC]},
    "stereotype": {"harmful": [GROUP, WRITE], "benign": [GROUP, WRITE]},
    "blackmarket": {"harmful": [BLACKMKT], "benign": [BLACKMKT]},
}
THEME_EXCLUDE = r"photosynthe"   # WildJailbreak's absurd-contrast prompts ("why can't vegans photosynthesize")
# misinformation requests ("spread the rumor that COVID is a bioweapon") are their own behaviour, not lab chemistry,
# exploit code or black markets
MISINFO = (r"\b(conspirac\w*|rumou?rs?|misinformation|disinformation|fake news|hoax|false (information|narrative|"
           r"claims?|story|stories|data|evidence|reports?)|narrative (claiming|suggesting|that)|theory that|"
           r"propaganda|man.made|lab.leak|misleading|engineered|encourag\w*|intentionally (created|released))")
MISINFO_THEMES = ("chemistry", "exploit", "blackmarket")


def theme_of(text: str, side: str) -> str | None:
    t = text.lower()
    if re.search(THEME_EXCLUDE, t):
        return None
    for k, v in THEMES.items():
        if all(re.search(rx, t) for rx in v[side]):
            return None if side == "harmful" and k in MISINFO_THEMES and re.search(MISINFO, t) else k
    return None


# WildJailbreak's benign labels are noisy ("Explain the process of building an illegal drug laboratory",
# "tips on stalking individuals online without getting caught"), and nearest-neighbour search toward a harmful prompt
# finds exactly those rows. Drop benign rows with explicit evasion / illegal-intent wording from the twin pools.
BENIGN_SUSPECT = (r"\b(without (getting |being )?(caught|detected|detection|noticed|traced|a trace|leaving (a |any )?trace)|"
                  r"undetect\w*|untraceable|illegal(ly)?|unlawful\w*|stalk\w*|hack\w* into|break\w* into|"
                  r"steal\w*|cultivate a virus|bio ?lab|poison (someone|a person|my|him|her)|"
                  r"(kill|hurt|harm) (someone|a person|my|him|her|myself)|evad\w* (the )?(police|law|taxes|authorities)|"
                  r"get away with|cover (it|this) up)")


# ---------------------------------------------------------------- "ordinary" filter for the reference set
# WildJailbreak vanilla_benign is built to *look* harmful (homonyms, absurd discrimination questions, private info of
# fictional people). Drop every row with a trigger stem; what survives is plain questions and requests.
ORD_STEMS = ("kill murder weapon gun bomb explos drug cocaine heroin hack exploit steal stole theft fraud scam launder "
             "terror violen abus assault rape sexual sexy porn nude naked suicid self-harm blood poison toxic attack "
             "shoot stab crime criminal illegal unlawful smuggl traffick counterfeit pirac pirat copyright trademark death "
             "stereotyp discriminat racis sexis prejudic bias hateful offensive slur extremis radical genocid "
             "holocaust slave nazi hitler tortur disclos password passport medical record health record privat privacy "
             "confidential classified secret surveil spy track stalk harass bullie bullying blackmail threat manipulat "
             "immigra hate "
             "refugee religio muslim islam jew christian ethnic racial lgbt gay lesbian transgender gender disabil "
             "vaccin conspira propaganda misinformation election politic gambl alcohol smok vape vaping fiction harry potter "
             "superman batman spider hogwarts gotham james bond sherlock marvel wonder woman hulk pokemon strangl "
             "eliminat crack blow destroy annihilat butcher execut torch smash chok suffocat "
             "bury wipe knock hijack ambush sabotag terminat demolish massacr assassin detonat ignit slay slash "
             "shred hunt kidnap ransom punch strik snip stealth sneak infiltrat rumor rumour gossip "
             "scandal affair divorce salary net worth childhood despair burden nonexistent self-worth worthless vanish "
             "run away grie depress anxi lonel trauma feeling emotion numb prohibit prevent stop banned deny denied "
             "denying disqualif exclud allowed cannot can't unable incapable rationale reasoning game minecraft fortnite "
             "grand theft gta skyrim zelda mario fighter league legend underage minorit child teen teleport "
             "time travel levitat telepath invisib breathe underwater photosynthe superpower magic wizard witch vampire "
             "zombie dragon unicorn alien martian ghost mermaid riot militar army arson deepfake intimate consent "
             "contact detail customer service phone number email address mailing address home address instagram "
             "twitter facebook credential login blast crush maim puncture exterminat burn diffus defeat dox cyber "
             "opponent combat conquer dominat tackl obliterat overthrow slaughter stomp squash hammer drown "
             "spongebob rapunzel cinderella mickey mouse shrek pikachu santa claus tooth fairy easter bunny elves "
             "talking pizza jury duty eligible equal rights right to marry right to vote "
             "isolat overwhelm failure exist escape reality distress self-perception unhappy hopeless").split(" ")
# short words that would over-match as prefixes ("hit" -> history, "die" -> diet, "dead" -> deadline)
ORD_EXACT = ("hit hits die dies dying dead war wars fire fires rig cut cuts kid kids minor minors gun guns sex stop "
             "beat beats trap traps nail bully handle handles account accounts").split(" ")
ORD_RX = re.compile(r"\b(" + "|".join(re.escape(s) for s in sorted(set(ORD_STEMS) - {""}, key=len, reverse=True)) + r")"
                    r"|\b(" + "|".join(ORD_EXACT) + r")\b"
                    r"|\bwhy (is|are|do|does|don't|should|shouldn't|aren't|isn't|can|could)\b"
                    r"|\bwhat (prevents|stops|prohibits|keeps)\b|\barguments? (for|against)\b", re.I)
TOPIC = {"coding": r"\b(code|coding|python|javascript|program\w*|function|algorithm|sql|api|software|debug\w*|html|"
                   r"css|java|c\+\+|excel|spreadsheet|database|app)\b",
         "math": r"\b(math\w*|equation|calculat\w*|algebra|geometry|probabilit\w*|statistic\w*|integral|derivative|"
                 r"fraction|percent\w*|sum of|solve)\b",
         "writing": r"\b(write|writing|essay|poem|story|novel|blog|article|letter|resume|draft|edit\w*|summar\w*|"
                    r"paragraph|sentence)\b",
         "cooking": r"\b(recipe|cook\w*|bak\w*|meal|kitchen|food)\b",
         "advice": r"\b(how can i|how do i|how should i|tips|advice|best way|improve|strateg\w*|ways to)\b"}


def topic_of(t: str) -> str:
    return next((k for k, rx in TOPIC.items() if re.search(rx, t, re.I)), "general")


# ---------------------------------------------------------------- TF-IDF nearest-neighbour twins
STOP = set(("a an the and or of to in on for with without by at from as is are was were be been being it its this "
            "that these those i you he she we they me my your our their his her them us can could would should will "
            "shall may might must do does did not no yes so if then than but about into over under also just very "
            "more most some any all each every what which who whom whose how why when where while please help "
            "want need like make sure let lets let's ve re ll").split())


def toks(s: str) -> list[str]:
    return [w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w
            for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 2 and w not in STOP]


class Twins:
    """Sparse TF-IDF cosine over a candidate pool via an inverted index (no sklearn)."""

    def __init__(self, cands: list[dict], bonus_of, max_df: float = 0.1):
        self.c = cands
        docs = [collections.Counter(toks(r["prompt"])) for r in cands]
        df = collections.Counter(t for d in docs for t in d)
        n = len(cands)
        self.idf = {t: math.log((n + 1) / (v + 1)) + 1 for t, v in df.items()}
        self.skip = {t for t, v in df.items() if v > max_df * n}
        self.inv = collections.defaultdict(list)
        for i, d in enumerate(docs):
            vec = {t: (1 + math.log(c)) * self.idf[t] for t, c in d.items()}
            z = math.sqrt(sum(w * w for w in vec.values())) or 1.0
            for t, w in vec.items():
                if t not in self.skip:
                    self.inv[t].append((i, w / z))
        self.bonus_of = bonus_of  # (query row, candidate row) -> float
        self.words = [words(r["prompt"]) for r in cands]

    def scores(self, q: dict, ratio: tuple[float, float]) -> dict[int, float]:
        d = collections.Counter(toks(q["prompt"]))
        idf_q = max(self.idf.values())  # unseen tokens: rarest weight (they cannot match anyway)
        vec = {t: (1 + math.log(c)) * self.idf.get(t, idf_q) for t, c in d.items()}
        z = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        acc: dict[int, float] = collections.defaultdict(float)
        for t, w in vec.items():
            for i, wc in self.inv.get(t, ()):
                acc[i] += w / z * wc
        lo, hi = ratio(q) if callable(ratio) else ratio
        qw = words(q["prompt"])
        return {i: s for i, s in acc.items() if lo <= self.words[i] / qw <= hi}

    def match(self, queries: list[dict], ratio, used: set[int], k: int = 60) -> list[tuple[int, float]]:
        """One-to-one greedy assignment by (cosine + bonus); returns (candidate index, raw cosine) per query."""
        cand = []
        for qi, q in enumerate(queries):
            sc = self.scores(q, ratio)
            top = heapq.nlargest(k, ((s + self.bonus_of(q, self.c[i]), -i, s) for i, s in sc.items()))
            cand += [(tot, -qi, ni, s) for tot, ni, s in top]   # ni = -candidate index (ties: lower index first)
        out: list = [None] * len(queries)
        for tot, nqi, ni, s in sorted(cand, reverse=True):
            qi, i = -nqi, -ni
            if out[qi] is None and i not in used:
                out[qi] = (i, s)
                used.add(i)
        for qi, q in enumerate(queries):  # rare: every top-k candidate was taken
            if out[qi] is None:
                sc = self.scores(q, ratio)
                best = max(((s + self.bonus_of(q, self.c[i]), -i, s) for i, s in sc.items() if i not in used),
                           default=None)
                if best is None:
                    raise SystemExit(f"no twin left for {q['id']}")
                out[qi] = (-best[1], best[2])
                used.add(-best[1])
        return out


# ---------------------------------------------------------------- bookkeeping
stats: dict = collections.defaultdict(collections.Counter)
manifest: dict = {"seed": SEED, "files": {}, "filters": {}, "matching": {},
                  "heldout": HELDOUT_FILES + ["allenai/wildjailbreak eval (all rows)",
                                              "mlabonne/harmful_behaviors test"],
                  "sources": {"allenai/wildjailbreak": f"{REV['allenai/wildjailbreak']} (train split only)",
                              "Bahushruth/abliteration-harmful-enriched":
                                  REV["Bahushruth/abliteration-harmful-enriched"],
                              "heretic-org/Semantic-Harmful": REV["heretic-org/Semantic-Harmful"],
                              "mlabonne/harmful_behaviors": f"{REV['mlabonne/harmful_behaviors']} (held out / filter)",
                              "OBLITERATUS": f"{OBLIT_COMMIT}:obliteratus/prompts.py (ast.literal_eval)",
                              "data/refusal_v1": "pairs_* included as-is; borderline_* excluded from new samples"},
                  "licenses": {"OBLITERATUS": "AGPL-3.0", "heretic-org/Semantic-Harmful": "CC-BY-4.0",
                               "mlabonne/harmless_alpaca (Semantic twins)": "see source",
                               "Bahushruth/abliteration-harmful-enriched": "Apache-2.0",
                               "allenai/wildjailbreak": "ODC-BY (gated)"}}


class Gate:
    """English filter + held-out decontamination + v1-borderline exclusion, cached per normalized text."""

    def __init__(self, ho: Heldout, v1_border: set[str]):
        self.ho, self.v1b, self.cache = ho, v1_border, {}

    def why(self, t: str) -> str | None:
        k = norm(t)
        if k not in self.cache:
            self.cache[k] = ("non_english" if not is_english(t) else "heldout_overlap" if self.ho.hit(t)
                             else "v1_borderline" if k in self.v1b else None)
        return self.cache[k]

    def pool(self, name: str, rows: list[dict]) -> list[dict]:
        out = []
        for r in rows:
            w = self.why(r["prompt"])
            stats[name]["candidates"] += 1
            if w:
                stats[name][w] += 1
            else:
                out.append(r)
        stats[name]["kept_pool"] += len(out)
        return out


def take(r: dict, seen: set, name: str) -> bool:
    k = norm(r["prompt"])
    if k in seen:
        stats[name]["duplicate"] += 1
        return False
    seen.add(k)
    return True


def write(name: str, rows: list[dict], note: str = ""):
    OUT.mkdir(parents=True, exist_ok=True)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), f"{name}: duplicate ids"
    p = OUT / f"{name}.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    data = p.read_bytes()
    wl = sorted(words(r["prompt"]) for r in rows)
    by_kind = collections.Counter(r["kind"] for r in rows)
    by_kind_words = {k: sorted(words(r["prompt"]) for r in rows if r["kind"] == k) for k in by_kind}
    manifest["files"][name] = {
        "n": len(rows), "file": p.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        "by_split": dict(sorted(collections.Counter(r["split"] for r in rows).items())),
        "by_kind": dict(sorted(by_kind.items())),
        "words_median": wl[len(wl) // 2],
        "words_median_by_kind": {k: v[len(v) // 2] for k, v in sorted(by_kind_words.items())},
        "sources": sorted({r["source"] for r in rows}), **({"note": note} if note else {})}
    print(f"{name:20} n={len(rows):5}  {dict(sorted(by_kind.items()))}")


def q(xs: list[float]) -> dict:
    xs = sorted(xs)
    pick = lambda f: round(xs[min(len(xs) - 1, int(f * len(xs)))], 3)  # noqa: E731
    return {"n": len(xs), "p10": pick(0.1), "median": pick(0.5), "p90": pick(0.9), "mean": round(sum(xs) / len(xs), 3)}


def frame_of(t: str) -> str | None:
    return next((k for k, rx in FRAMES.items() if re.search(rx, t, re.I)), None)


# ---------------------------------------------------------------- main
def main():
    rng = random.Random(SEED)
    wsrc = f"allenai/wildjailbreak@{REV['allenai/wildjailbreak']}:train"
    esrc = f"Bahushruth/abliteration-harmful-enriched@{REV['Bahushruth/abliteration-harmful-enriched']}"

    # ---- held out
    held = []
    for f in HELDOUT_FILES:
        for line in open(ROOT / f):
            x = json.loads(line)
            if "messages" in x:  # bfcl: list of chat messages (or its python repr)
                msgs = x["messages"] if isinstance(x["messages"], list) else ast.literal_eval(x["messages"])
                held += [m["content"] for m in msgs if m.get("role") == "user"]
            else:
                held.append(x.get("question") or x.get("prompt") or x.get("problem") or x.get("instruction"))
    wjb_eval = pd.read_csv(hf("allenai/wildjailbreak", "eval/eval.tsv"), sep="\t", keep_default_na=False)
    held += list(wjb_eval.adversarial)
    mlab_test = pd.read_parquet(hf("mlabonne/harmful_behaviors", "data/test-00000-of-00001.parquet"))
    held += list(mlab_test.text)
    held = [h for h in held if h]
    manifest["heldout_n"] = len(held)
    v1_border = {norm(json.loads(ln)["prompt"]) for f in ("borderline_harmful", "borderline_harmless")
                 for ln in open(V1 / f"{f}.jsonl")}
    gate = Gate(Heldout(held), v1_border)
    seen: set = set()

    # ---- v1 pairs, as-is (decontaminated in v1 against a subset of today's held-out list: re-check, drop hits)
    v1h = [json.loads(ln) for ln in open(V1 / "pairs_harmful.jsonl")]
    v1l = [json.loads(ln) for ln in open(V1 / "pairs_harmless.jsonl")]
    assert [r["id"] for r in v1h] == [r["id"] for r in v1l]
    v1_src = {"obliteratus": f"OBLITERATUS@{OBLIT_COMMIT}:obliteratus/prompts.py",
              "semantic": f"heretic-org/Semantic-Harmful@{REV['heretic-org/Semantic-Harmful']}"}
    v1_units, v1_drop = [], []
    for h, l in zip(v1h, v1l):
        seen.update({norm(h["prompt"]), norm(l["prompt"])})
        if any(gate.ho.hit(r["prompt"]) for r in (h, l)):
            v1_drop.append(h["id"])
            continue
        conv = lambda r, s: {"id": r["id"], "prompt": r["prompt"], "split": s, "kind": f"v1_pairs_{r['split']}",  # noqa: E731
                             "source": v1_src[r["split"]], "pair": r["pair"],
                             **{k: r[k] for k in ("category", "score") if k in r}}
        v1_units.append((conv(h, "harmful"), conv(l, "harmless")))
    stats["v1_pairs"].update({"candidates": len(v1h), "heldout_overlap": len(v1_drop), "kept": len(v1_units)})
    manifest["v1_pairs_dropped_heldout"] = v1_drop

    # ---- WildJailbreak train
    tr = pd.read_csv(hf("allenai/wildjailbreak", "train/train.tsv"), sep="\t", keep_default_na=False,
                     usecols=["vanilla", "adversarial", "data_type"])
    tr["prompt"] = [(a if a else v).strip() for a, v in zip(tr.adversarial, tr.vanilla)]
    wjb_rows = {}
    for dtype in ["vanilla_harmful", "vanilla_benign", "adversarial_harmful", "adversarial_benign"]:
        g = tr[tr.data_type == dtype]
        if dtype.startswith("adversarial"):
            g = g[g.prompt.str.split().str.len() <= ADV_MAX_WORDS]
        wjb_rows[dtype] = [{"id": f"wjb-{i}", "prompt": p, "wjb": dtype} for i, p in zip(g.index, g.prompt)]

    # general sample: v1 recipe (random 20k per kind, length-matched within kind), new prompts only
    gen_pool = {}
    for dtype, rows in wjb_rows.items():
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        cand = [rows[i] for i in idx[:20000]]
        gen_pool[dtype] = [r for r in gate.pool(f"wjb_{dtype}", cand) if norm(r["prompt"]) not in seen]
    wjb_units = []
    for fam, n, width in (("adversarial", N_WJB_ADV, 20), ("vanilla", N_WJB_VAN, 3)):
        h_pool = list({norm(r["prompt"]): r for r in gen_pool[f"{fam}_harmful"]}.values())
        b_pool = list({norm(r["prompt"]): r for r in gen_pool[f"{fam}_benign"]}.values())
        sh, sb = length_matched(h_pool, b_pool, n, rng, width=width)
        for h, b in zip(sh, sb):
            assert take(h, seen, f"wjb_{fam}_harmful") and take(b, seen, f"wjb_{fam}_benign")
            wjb_units.append(tuple({"id": r["id"], "prompt": r["prompt"], "split": s, "kind": f"wjb_{r['wjb']}",
                                    "source": wsrc} for r, s in ((h, "harmful"), (b, "harmless"))))

    # ---- enriched (harmful only): v1 filters
    enr = pd.concat([pd.read_parquet(hf("Bahushruth/abliteration-harmful-enriched", f"data/{s}-00000-of-00001.parquet"))
                     for s in ("train", "test")], ignore_index=True)
    mlab_train = pd.read_parquet(hf("mlabonne/harmful_behaviors", "data/train-00000-of-00001.parquet"))
    mlab = {norm(t) for t in list(mlab_train.text) + list(mlab_test.text)}
    enr_rows = []
    for i, t in enumerate(enr.text):
        t = t.strip()
        if norm(t) in mlab:
            stats["enriched"]["mlabonne_original"] += 1
        elif re.search(JAILBREAK, t, re.I):
            stats["enriched"]["jailbreak_style"] += 1
        else:
            enr_rows.append({"id": f"enr-{i}", "prompt": t, "frame": frame_of(t)})
    enr_rows = gate.pool("enriched", enr_rows)

    # ---- residual themes
    res_h = {th: collections.defaultdict(list) for th in THEMES}
    for fam in ("adversarial", "vanilla"):
        for r in gate.pool(f"theme_{fam}_harmful", [r for r in wjb_rows[f"{fam}_harmful"] if theme_of(r["prompt"], "harmful")]):
            res_h[theme_of(r["prompt"], "harmful")][fam].append(r)
    for r in enr_rows:
        if th := theme_of(r["prompt"], "harmful"):
            res_h[th]["enriched"].append(r)
    stats["residual_pools"] = collections.Counter({f"{th}/{fam}": len(v) for th, d in res_h.items()
                                                   for fam, v in d.items()})
    res_sel = []  # (theme, family-of-twin, row)
    for th in THEMES:
        prefix = collections.Counter()
        got = collections.Counter()
        for fam in ("enriched", "vanilla", "adversarial"):
            pool = list(res_h[th][fam])
            rng.shuffle(pool)
            quota = (RESIDUAL_QUOTA.get(fam) if fam != "adversarial" else N_RESIDUAL_PER_THEME - sum(got.values()))
            for r in pool:
                if got[fam] >= quota:
                    break
                pre = " ".join(norm(r["prompt"]).split()[:6])
                if prefix[pre] >= 2 or not take(r, seen, f"residual_{th}"):
                    continue
                prefix[pre] += 1
                got[fam] += 1
                res_sel.append((th, fam, r))
        stats[f"residual_{th}"].update({f"selected_{k}": v for k, v in got.items()})

    # benign twin pool: all of vanilla_benign + adversarial_benign (on-theme rows + a random 20k), one used set.
    # The length-ratio window keeps vanilla-length queries on short rows and adversarial queries on long ones.
    adv_all = wjb_rows["adversarial_benign"]
    idx = list(range(len(adv_all)))
    rng.shuffle(idx)
    pick = set(idx[:ADV_BENIGN_POOL]) | {i for i, r in enumerate(adv_all) if theme_of(r["prompt"], "benign")}
    tp = (gate.pool("twin_pool_vanilla_benign", wjb_rows["vanilla_benign"])
          + gate.pool("twin_pool_adversarial_benign", [adv_all[i] for i in sorted(pick)]))
    twin_pool = []
    for r in {norm(r["prompt"]): r for r in tp}.values():
        if norm(r["prompt"]) in seen:
            continue
        if re.search(BENIGN_SUSPECT, r["prompt"], re.I):
            stats["twin_pool"]["benign_suspect_dropped"] += 1
            continue
        twin_pool.append(dict(r, theme_b=theme_of(r["prompt"], "benign"), frame=frame_of(r["prompt"])))
    stats["twin_pool"]["kept_pool"] = len(twin_pool)
    used: set[int] = set()

    def ratio(r):  # length window by the query's own length family
        return LEN_RATIO["adversarial" if words(r["prompt"]) > 60 else "vanilla"]

    res_units, res_sims = [], collections.defaultdict(list)
    tw = Twins(twin_pool, lambda qr, c: THEME_BONUS * (c["theme_b"] == qr["theme"]))
    qrows = [dict(r, theme=th) for th, _, r in res_sel]
    for (th, src, r), (ci, sim) in zip(res_sel, tw.match(qrows, ratio, used)):
        c = twin_pool[ci]
        assert take(c, seen, f"residual_{th}_twin")
        src_h = esrc if src == "enriched" else wsrc
        kind_h = f"enriched_{r['frame'] or 'unframed'}" if src == "enriched" else f"wjb_{r['wjb']}"
        pair = f"r3-{r['id']}"
        res_units.append(({"id": r["id"], "prompt": r["prompt"], "split": "harmful", "kind": f"residual_{th}",
                           "source": src_h, "category": th, "origin": kind_h, "pair": pair},
                          {"id": c["id"], "prompt": c["prompt"], "split": "harmless", "kind": f"residual_{th}",
                           "source": wsrc, "category": th, "origin": f"wjb_{c['wjb']}", "pair": pair,
                           "pair_sim": round(sim, 3), "twin_on_theme": c["theme_b"] == th}))
        res_sims[th].append((sim, c["theme_b"] == th, words(c["prompt"]) / words(r["prompt"])))
    manifest["matching"]["residual"] = {
        th: {"cosine": q([s for s, _, _ in v]), "twin_on_theme": round(sum(o for _, o, _ in v) / len(v), 3),
             "length_ratio": q([lr for _, _, lr in v])} for th, v in sorted(res_sims.items())}

    # ---- enriched framings block: N per frame (shortfall filled from the other frames), new prompts only;
    # the enriched set has no harmless side: twin = nearest benign WildJailbreak row, same framing preferred
    erows = [r for r in enr_rows if r["frame"]]
    stats["enriched"]["no_frame"] += len(enr_rows) - len(erows)
    rng.shuffle(erows)
    prefix, by_frame, eh = collections.Counter(), collections.Counter(), []
    for cap in (N_ENRICHED_PER_FRAME, 4 * N_ENRICHED_PER_FRAME):
        for r in erows:  # cap 2 per opening 6 words (the set is heavily templated)
            pre = " ".join(norm(r["prompt"]).split()[:6])
            if (len(eh) >= 4 * N_ENRICHED_PER_FRAME or prefix[pre] >= 2 or by_frame[r["frame"]] >= cap
                    or norm(r["prompt"]) in seen):
                continue
            prefix[pre] += 1
            by_frame[r["frame"]] += 1
            seen.add(norm(r["prompt"]))
            eh.append(r)
    stats["enriched"].update({f"selected_{k}": v for k, v in sorted(by_frame.items())})
    tw = Twins(twin_pool, lambda qr, c: THEME_BONUS * (c["frame"] == qr["frame"]))
    enr_units, enr_sims = [], collections.defaultdict(list)
    for r, (ci, sim) in zip(eh, tw.match(eh, ratio, used)):
        c = twin_pool[ci]
        assert take(c, seen, "enriched_twin")
        pair = f"r3-{r['id']}"
        kind = f"enriched_{r['frame']}"
        enr_units.append(({"id": r["id"], "prompt": r["prompt"], "split": "harmful", "kind": kind, "source": esrc,
                           "category": r["frame"], "pair": pair},
                          {"id": c["id"], "prompt": c["prompt"], "split": "harmless", "kind": kind, "source": wsrc,
                           "category": r["frame"], "origin": f"wjb_{c['wjb']}", "pair": pair,
                           "pair_sim": round(sim, 3), "twin_same_frame": c["frame"] == r["frame"]}))
        enr_sims[r["frame"]].append((sim, c["frame"] == r["frame"], words(c["prompt"]) / words(r["prompt"])))
    manifest["matching"]["enriched"] = {
        fr: {"cosine": q([s for s, _, _ in v]), "twin_same_frame": round(sum(o for _, o, _ in v) / len(v), 3),
             "length_ratio": q([lr for _, _, lr in v])} for fr, v in sorted(enr_sims.items())}

    # ---- write build files: one shuffled list of (harmful, harmless) units -> any prefix is a balanced mix
    units = v1_units + wjb_units + res_units + enr_units
    rng.shuffle(units)
    write("build_harmful", [h for h, _ in units], "row k's counterpart is build_harmless row k")
    write("build_harmless", [l for _, l in units], "row k's counterpart is build_harmful row k")

    # ---- reference: ordinary prompts (OBLITERATUS harmless pool, unmatched Semantic twins, filtered WJB vanilla_benign)
    src = urllib.request.urlopen(OBLIT_URL, timeout=60).read().decode()
    pool = next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "_HARMLESS_POOL")
    sem = json.load(open(hf("heretic-org/Semantic-Harmful", "metadata/matched_pairs.json")))["pairs"]
    ref_cands = ([{"id": f"ref-ob-{i:03d}", "prompt": t.strip(), "kind": "obliteratus_harmless_pool",
                   "source": f"OBLITERATUS@{OBLIT_COMMIT}:obliteratus/prompts.py:_HARMLESS_POOL"}
                  for i, t in enumerate(pool)]
                 + [{"id": f"ref-sem-{p['harmless_index']:05d}", "prompt": p["harmless"].strip(),
                     "kind": "semantic_unmatched_alpaca",
                     "source": f"heretic-org/Semantic-Harmful@{REV['heretic-org/Semantic-Harmful']}"}
                    for p in sem if p["score"] < SEMANTIC_MIN_SCORE])
    ref, c = [], stats["reference"]
    for r in gate.pool("reference_fixed", ref_cands):
        if take(r, seen, "reference"):
            ref.append(r)
    wv = [r for r in twin_pool if r["wjb"] == "vanilla_benign" and norm(r["prompt"]) not in seen]
    rng.shuffle(wv)
    for r in wv:
        if len(ref) >= N_REFERENCE:
            break
        if ORD_RX.search(r["prompt"]) or not 6 <= words(r["prompt"]) <= 60:
            c["wjb_not_ordinary"] += 1
            continue
        if take(r, seen, "reference"):
            ref.append({"id": r["id"], "prompt": r["prompt"], "kind": "wjb_vanilla_benign_ordinary", "source": wsrc})
    for r in ref:
        r["split"] = "harmless"
        r["category"] = topic_of(r["prompt"])
    rng.shuffle(ref)
    ref = [{k: r[k] for k in ("id", "prompt", "split", "kind", "category", "source")} for r in ref]
    write("reference_harmless", ref, "projection reference; not adversarial; see docs/refusal_v3_dataset.md for gaps")
    manifest["reference_topics"] = dict(collections.Counter(r["category"] for r in ref).most_common())

    manifest["filters"] = {k: dict(v) for k, v in stats.items()}
    manifest["params"] = {"adv_max_words": ADV_MAX_WORDS, "heldout_containment_max": 0.5, "n_wjb_adversarial": N_WJB_ADV,
                          "n_wjb_vanilla": N_WJB_VAN, "n_residual_per_theme": N_RESIDUAL_PER_THEME,
                          "residual_quota": RESIDUAL_QUOTA, "n_enriched_per_frame": N_ENRICHED_PER_FRAME,
                          "n_reference": N_REFERENCE, "adv_benign_twin_pool": ADV_BENIGN_POOL,
                          "theme_bonus": THEME_BONUS, "len_ratio": LEN_RATIO}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"filters": manifest["filters"], "matching": manifest["matching"],
                      "reference_topics": manifest["reference_topics"]}, indent=1))


if __name__ == "__main__":
    main()
