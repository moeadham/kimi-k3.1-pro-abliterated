"""Raw-vs-summary check for tools/probe_reasoning.py output.

Re-tokenizes each returned reasoning trace with the real K3.1 tokenizer and compares it with
the provider-reported usage.completion_tokens_details.reasoning_tokens. A raw trace matches to
within a few delimiter tokens; a summary would be much shorter than the billed count.

Usage: uv run --env-file .env --with transformers --with tiktoken --with blobfile \
         python tools/check_reasoning_tokens.py [out/probe_reasoning/responses.jsonl]
"""

import json
import sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("moonshotai/Kimi-K3.1-Pro", trust_remote_code=True)
path = sys.argv[1] if len(sys.argv) > 1 else "out/probe_reasoning/responses.jsonl"
for line in open(path):
    x = json.loads(line)
    if x.get("status") != 200:
        continue
    u = x["usage"]
    rt = u["completion_tokens_details"].get("reasoning_tokens")
    n_r = len(tok.encode(x["reasoning"], add_special_tokens=False))
    n_c = len(tok.encode(x["content"], add_special_tokens=False))
    print(f"{x['provider']:17} {x['prompt']:13} {x['effort']:6} reasoning: billed={rt} retok={n_r} (diff {rt - n_r:+d}) | "
          f"completion: billed={u['completion_tokens']} retok r+c={n_r + n_c} (diff {u['completion_tokens'] - n_r - n_c:+d})")
