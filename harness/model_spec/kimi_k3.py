"""Kimi K3 / K3.1 residual-stream spec for vLLM 0.34.0. Derived in docs/k3_residual_map.md.

Pure data + helpers (no vLLM import) so it can be unit-tested on CPU and shared by the
steering plugin, capture, and bake code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Writer(str, Enum):
    EMBED = "embed"  # model.embed_tokens
    ATTN = "attn"  # layers.i.self_attn (KDA / MLA / DSA), weight: self_attn.o_proj
    MLP = "mlp"  # layers.i.mlp (dense KimiMLP or KimiMoE)


# Module paths to hook (outputs), relative to the inner KimiLinearModel (`.model`).
EMBED_MODULE = "embed_tokens"
ATTN_MODULE = "layers.{i}.self_attn"
MLP_MODULE = "layers.{i}.mlp"

# Checkpoint tensor names (HF/safetensors naming, without the "language_model." prefix)
# whose output space is the residual stream. These are what the bake edits.
EMBED_WEIGHT = "model.embed_tokens.weight"
ATTN_WEIGHTS = ("model.layers.{i}.self_attn.o_proj.weight",)
DENSE_MLP_WEIGHTS = ("model.layers.{i}.mlp.down_proj.weight",)
MOE_WEIGHTS = (
    "model.layers.{i}.block_sparse_moe.routed_expert_up_proj.weight",
    "model.layers.{i}.block_sparse_moe.shared_experts.down_proj.weight",
)

# Read-only stream consumers that must NOT be edited (score vectors for AttnRes mixing).
NON_WRITER_SUFFIXES = (
    "self_attention_res_proj.weight",
    "mlp_res_proj.weight",
    "output_attn_res_proj.weight",
)

# Env / config constraints the harness enforces.
REQUIRED_ENV = {"VLLM_KIMI_K3_AUX_ATTN_RES_STREAM": "1"}  # capture the mixed stream
FORBIDDEN_DECODE_BACKEND = "latency_specialized"


@dataclass(frozen=True)
class K3Layout:
    num_layers: int
    first_k_dense_replace: int
    attn_res_block_size: int | None
    kda_layers_1idx: frozenset[int]
    dsa_layers_1idx: frozenset[int]
    checkpoint_prefix: str = "language_model."

    @classmethod
    def from_config(cls, cfg: dict) -> "K3Layout":
        t = cfg.get("text_config", cfg)
        lin = t.get("linear_attn_config") or {}
        dsa = t.get("dsa_config") or {}
        return cls(
            num_layers=t["num_hidden_layers"],
            first_k_dense_replace=t.get("first_k_dense_replace", 0),
            attn_res_block_size=t.get("attn_res_block_size"),
            kda_layers_1idx=frozenset(lin.get("kda_layers", [])),
            dsa_layers_1idx=frozenset(dsa.get("dsa_layers", [])),
            checkpoint_prefix="language_model." if "text_config" in cfg else "",
        )

    def attn_type(self, i: int) -> str:
        if (i + 1) in self.kda_layers_1idx:
            return "kda"
        return "dsa" if (i + 1) in self.dsa_layers_1idx else "mla"

    def is_moe(self, i: int) -> bool:
        return i >= self.first_k_dense_replace

    def is_block_write(self, i: int) -> bool:
        return self.attn_res_block_size is not None and i % self.attn_res_block_size == 0

    def writer_weights(self, i: int) -> dict[Writer, tuple[str, ...]]:
        p = self.checkpoint_prefix
        mlp = MOE_WEIGHTS if self.is_moe(i) else DENSE_MLP_WEIGHTS
        return {
            Writer.ATTN: tuple(p + w.format(i=i) for w in ATTN_WEIGHTS),
            Writer.MLP: tuple(p + w.format(i=i) for w in mlp),
        }

    def all_writer_weights(self) -> list[str]:
        names = [self.checkpoint_prefix + EMBED_WEIGHT]
        for i in range(self.num_layers):
            for ws in self.writer_weights(i).values():
                names.extend(ws)
        return names

    def hook_modules(self) -> list[tuple[str, Writer, int]]:
        """(module path under the inner model, writer kind, layer index; -1 for embed)."""
        out = [(EMBED_MODULE, Writer.EMBED, -1)]
        for i in range(self.num_layers):
            out.append((ATTN_MODULE.format(i=i), Writer.ATTN, i))
            out.append((MLP_MODULE.format(i=i), Writer.MLP, i))
        return out
