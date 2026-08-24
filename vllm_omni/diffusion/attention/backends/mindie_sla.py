# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MindIE-SD SparseLinearAttention backend for HunyuanImage3 diffusion."""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl

logger = init_logger(__name__)

_FORMAT_VERSION = 1
_ARCHITECTURE = "HunyuanImage3SparseLinearAttentionAdapter"
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_VALID_MASK_POLICIES = frozenset({"hybrid", "error", "dense_fallback"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_adapter_paths(adapter_path: str) -> tuple[Path, Path]:
    path = Path(adapter_path).expanduser().resolve()
    if path.is_dir():
        weights = path / "adapter.safetensors"
        config = path / "adapter_config.json"
    else:
        weights = path
        config = path.with_name("adapter_config.json")
    if not weights.is_file():
        raise FileNotFoundError(f"MindIE SLA adapter weights do not exist: {weights}")
    if not config.is_file():
        raise FileNotFoundError(f"MindIE SLA adapter config does not exist: {config}")
    return weights, config


@lru_cache(maxsize=8)
def _load_adapter(adapter_path: str) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    weights_path, config_path = _resolve_adapter_paths(adapter_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"MindIE SLA adapter config must be a JSON object: {config_path}")
    if config.get("format_version") != _FORMAT_VERSION:
        raise ValueError(
            f"Unsupported SLA adapter format_version={config.get('format_version')!r}; expected {_FORMAT_VERSION}."
        )
    if config.get("architecture") != _ARCHITECTURE:
        raise ValueError(
            f"Invalid SLA adapter architecture={config.get('architecture')!r}; expected {_ARCHITECTURE!r}."
        )
    expected_sha = config.get("adapter_sha256")
    actual_sha = _sha256(weights_path)
    if not isinstance(expected_sha, str) or actual_sha != expected_sha:
        raise ValueError(
            f"SLA adapter SHA256 mismatch for {weights_path}: expected {expected_sha!r}, got {actual_sha}."
        )

    num_layers = int(config.get("num_layers", 0))
    head_dim = int(config.get("head_dim", 0))
    if num_layers <= 0 or head_dim <= 0:
        raise ValueError(f"Invalid SLA adapter geometry: num_layers={num_layers}, head_dim={head_dim}.")
    tensors = load_file(str(weights_path), device="cpu")
    expected_keys = {
        f"layers.{layer}.sla.proj_l.{parameter}" for layer in range(num_layers) for parameter in ("weight", "bias")
    }
    missing = sorted(expected_keys - set(tensors))
    unexpected = sorted(set(tensors) - expected_keys)
    if missing or unexpected:
        raise ValueError(f"Invalid SLA adapter keys: missing={missing or 'none'}, unexpected={unexpected or 'none'}")
    for name, tensor in tensors.items():
        expected_shape = (head_dim,) if name.endswith(".bias") else (head_dim, head_dim)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Invalid SLA tensor shape for {name}: expected {expected_shape}, got {tuple(tensor.shape)}"
            )
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all().item():
            raise ValueError(f"SLA adapter tensor must be finite floating point: {name} ({tensor.dtype})")
    if int(config.get("tensor_count", -1)) != len(tensors):
        raise ValueError("SLA adapter tensor_count does not match adapter.safetensors.")
    parameter_count = sum(tensor.numel() for tensor in tensors.values())
    if int(config.get("parameter_count", -1)) != parameter_count:
        raise ValueError("SLA adapter parameter_count does not match adapter.safetensors.")
    logger.info(
        "Validated HunyuanImage3 SLA adapter %s: layers=%d, tensors=%d, parameters=%d, sha256=%s",
        weights_path,
        num_layers,
        len(tensors),
        parameter_count,
        actual_sha,
    )
    return config, tensors


def _layer_index(prefix: str) -> int:
    match = _LAYER_RE.search(prefix)
    if match is None:
        raise ValueError(f"Cannot determine Hunyuan layer index from SLA attention prefix: {prefix!r}")
    return int(match.group(1))


def _repeat_gqa_kv(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q_heads, kv_heads = query.shape[2], key.shape[2]
    if key.shape[2] != value.shape[2]:
        raise ValueError(f"K/V head mismatch: key={key.shape[2]}, value={value.shape[2]}")
    if q_heads == kv_heads:
        return key, value
    if q_heads % kv_heads != 0:
        raise ValueError(f"GQA requires Q heads divisible by KV heads; got q_heads={q_heads}, kv_heads={kv_heads}.")
    repeat = q_heads // kv_heads
    return key.repeat_interleave(repeat, dim=2), value.repeat_interleave(repeat, dim=2)


class MindIESLABackend(AttentionBackend):
    supported_platforms = ("npu",)

    @classmethod
    def validate_available(cls) -> None:
        if find_spec("mindiesd") is None:
            raise ImportError("MINDIE_SLA requires MindIE-SD. Install it so `import mindiesd` succeeds.")

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "MINDIE_SLA"

    @staticmethod
    def get_impl_cls() -> type[MindIESLAImpl]:
        return MindIESLAImpl


class MindIESLAImpl(nn.Module, AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        backend_kwargs: dict[str, Any] | None = None,
        **extra_impl_args: Any,
    ) -> None:
        super().__init__()
        opts = backend_kwargs or {}
        role = extra_impl_args.get("role")
        if role != "hunyuan.diffusion":
            raise ValueError(f"MINDIE_SLA is restricted to role='hunyuan.diffusion'; got {role!r}.")
        if causal:
            raise ValueError("MINDIE_SLA does not support causal attention.")
        if head_size not in (64, 128):
            raise ValueError(f"MINDIE_SLA supports head_size 64 or 128; got {head_size}.")
        adapter_path = opts.get("adapter_path")
        if not isinstance(adapter_path, str) or not adapter_path:
            raise ValueError("MINDIE_SLA requires backend option adapter_path.")
        self.mask_policy = str(opts.get("mask_policy", "hybrid"))
        if self.mask_policy not in _VALID_MASK_POLICIES:
            raise ValueError(f"Invalid MINDIE_SLA mask_policy: {self.mask_policy!r}")

        from mindiesd.layers import SparseLinearAttention

        self.sla = SparseLinearAttention(
            head_dim=head_size,
            topk=float(opts.get("topk", 0.125)),
            BLKQ=int(opts.get("blkq", 64)),
            BLKK=int(opts.get("blkk", 128)),
            use_bf16=bool(opts.get("use_bf16", True)),
            inner_precise=opts.get("inner_precise"),
        )
        self.dense_fallback = SDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
        )
        config, tensors = _load_adapter(adapter_path)
        layer = _layer_index(prefix)
        if layer >= int(config["num_layers"]):
            raise ValueError(f"SLA layer {layer} is outside adapter num_layers={config['num_layers']}.")
        if int(config["head_dim"]) != head_size:
            raise ValueError(f"SLA adapter head_dim={config['head_dim']} does not match model head_size={head_size}.")
        runtime_signature = {
            "topk": float(opts.get("topk", 0.125)),
            "blkq": int(opts.get("blkq", 64)),
            "blkk": int(opts.get("blkk", 128)),
            "compute_dtype": "bfloat16" if bool(opts.get("use_bf16", True)) else "float16",
        }
        artifact_signature = {name: config.get(name) for name in runtime_signature}
        if artifact_signature != runtime_signature:
            raise ValueError(
                "MINDIE_SLA runtime settings do not match the trained adapter: "
                f"runtime={runtime_signature}, adapter={artifact_signature}."
            )
        # HunyuanImage3Pipeline calls post_init() after attention construction,
        # which reinitializes nn.Linear modules. Install the adapter lazily after
        # model initialization and checkpoint loading have both completed.
        self.register_buffer(
            "_adapter_weight",
            tensors[f"layers.{layer}.sla.proj_l.weight"].clone(),
            persistent=False,
        )
        self.register_buffer(
            "_adapter_bias",
            tensors[f"layers.{layer}.sla.proj_l.bias"].clone(),
            persistent=False,
        )
        self._adapter_installed = False

    def get_externally_loaded_parameter_names(self, prefix: str) -> set[str]:
        """Return parameters supplied by the validated SLA artifact."""
        return {
            f"{prefix}.sla.proj_l.weight",
            f"{prefix}.sla.proj_l.bias",
        }

    def _install_adapter(self) -> None:
        if self._adapter_installed:
            return
        weight = self.sla.proj_l.weight
        bias = self.sla.proj_l.bias
        if weight.is_meta or (bias is not None and bias.is_meta):
            raise RuntimeError("MINDIE_SLA proj_l parameters are still on the meta device at first forward.")
        if bias is None:
            raise RuntimeError("MINDIE_SLA SparseLinearAttention.proj_l must have a bias parameter.")
        with torch.no_grad():
            weight.copy_(self._adapter_weight.to(device=weight.device, dtype=weight.dtype))
            bias.copy_(self._adapter_bias.to(device=bias.device, dtype=bias.dtype))
        self._adapter_installed = True

    @staticmethod
    def _reject_parallel_metadata(attn_metadata: AttentionMetadata | None) -> None:
        if attn_metadata is not None and any(
            tensor is not None
            for tensor in (attn_metadata.joint_query, attn_metadata.joint_key, attn_metadata.joint_value)
        ):
            raise NotImplementedError("MINDIE_SLA first version requires sequence_parallel_size=1 (no joint Q/K/V).")

    def _run_sla(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                f"MINDIE_SLA expects [B, L, H, D] Q/K/V; got {query.shape=}, {key.shape=}, {value.shape=}."
            )
        if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
            raise ValueError(f"MINDIE_SLA batch mismatch: Q={query.shape[0]}, K={key.shape[0]}, V={value.shape[0]}.")
        if query.shape[-1] != key.shape[-1] or key.shape[-1] != value.shape[-1]:
            raise ValueError(
                f"MINDIE_SLA head dimension mismatch: Q={query.shape[-1]}, K={key.shape[-1]}, V={value.shape[-1]}."
            )
        if key.shape[1] != value.shape[1]:
            raise ValueError(f"MINDIE_SLA K/V sequence mismatch: K={key.shape[1]}, V={value.shape[1]}.")
        self._install_adapter()
        key, value = _repeat_gqa_kv(query, key, value)
        q, k, v = (tensor.transpose(1, 2).contiguous() for tensor in (query, key, value))
        output = self.sla(q, k, v)
        return output.transpose(1, 2).contiguous()

    def _hybrid_masked_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if self.mask_policy == "error":
            raise ValueError("MINDIE_SLA received an attention mask while mask_policy='error'.")
        if self.mask_policy == "dense_fallback":
            logger.warning_once("MINDIE_SLA is using the explicitly configured dense_fallback mask policy.")
            return self.dense_fallback._forward_impl(query, key, value, attn_metadata, mask_mode="full_qk")
        spans_by_batch = attn_metadata.full_attn_spans
        if spans_by_batch is None or len(spans_by_batch) != query.shape[0]:
            raise ValueError("MINDIE_SLA hybrid mask policy requires full_attn_spans for every batch row.")

        query_offset = key.shape[1] - query.shape[1]
        if query_offset < 0:
            raise ValueError(
                "MINDIE_SLA hybrid mode requires KV length >= query length; "
                f"got Lq={query.shape[1]}, Lk={key.shape[1]}."
            )
        rows = []
        for batch_index, spans in enumerate(spans_by_batch):
            intersections = [
                (max(start, query_offset), min(end, key.shape[1]))
                for start, end in spans
                if max(start, query_offset) < min(end, key.shape[1])
            ]
            if len(intersections) != 1 or intersections[0][1] != key.shape[1]:
                raise ValueError(
                    "MINDIE_SLA hybrid mode requires one full-attention image suffix ending at the KV length; "
                    f"batch={batch_index}, spans={spans}, q_offset={query_offset}, kv_len={key.shape[1]}."
                )
            local_start = intersections[0][0] - query_offset
            if local_start:
                row_metadata = AttentionMetadata(
                    attn_mask=attn_metadata.attn_mask[batch_index : batch_index + 1, :, :local_start, :]
                )
                dense_prefix = self.dense_fallback._forward_impl(
                    query[batch_index : batch_index + 1, :local_start],
                    key[batch_index : batch_index + 1],
                    value[batch_index : batch_index + 1],
                    row_metadata,
                    mask_mode="full_qk",
                )
            else:
                dense_prefix = query[batch_index : batch_index + 1, :0]
            sparse_suffix = self._run_sla(
                query[batch_index : batch_index + 1, local_start:],
                key[batch_index : batch_index + 1],
                value[batch_index : batch_index + 1],
            )
            rows.append(torch.cat([dense_prefix, sparse_suffix], dim=1))
        return torch.cat(rows, dim=0)

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        self._reject_parallel_metadata(attn_metadata)
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            return self._hybrid_masked_forward(query, key, value, attn_metadata)
        return self._run_sla(query, key, value)
