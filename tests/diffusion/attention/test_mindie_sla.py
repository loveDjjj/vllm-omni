# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import hashlib
import json
import sys
from types import ModuleType

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.mindie_sla import (
    MindIESLAImpl,
    _load_adapter,
    _repeat_gqa_kv,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _FakeSLA(nn.Module):
    last_shapes = None

    def __init__(self, head_dim, **_kwargs):
        super().__init__()
        self.proj_l = nn.Linear(head_dim, head_dim, dtype=torch.float32)

    def forward(self, q, k, v):
        type(self).last_shapes = (tuple(q.shape), tuple(k.shape), tuple(v.shape))
        return q


@pytest.fixture
def fake_mindiesd(monkeypatch):
    module = ModuleType("mindiesd")
    layers = ModuleType("mindiesd.layers")
    layers.SparseLinearAttention = _FakeSLA
    module.layers = layers
    monkeypatch.setitem(sys.modules, "mindiesd", module)
    monkeypatch.setitem(sys.modules, "mindiesd.layers", layers)


def _make_adapter(tmp_path, *, corrupt_sha=False):
    tensors = {}
    for layer in range(2):
        tensors[f"layers.{layer}.sla.proj_l.weight"] = torch.full((64, 64), float(layer + 1))
        tensors[f"layers.{layer}.sla.proj_l.bias"] = torch.full((64,), float(layer + 1))
    weights = tmp_path / "adapter.safetensors"
    save_file(tensors, str(weights))
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    config = {
        "format_version": 1,
        "architecture": "HunyuanImage3SparseLinearAttentionAdapter",
        "num_layers": 2,
        "head_dim": 64,
        "topk": 0.125,
        "blkq": 64,
        "blkk": 128,
        "compute_dtype": "bfloat16",
        "tensor_count": 4,
        "parameter_count": sum(t.numel() for t in tensors.values()),
        "adapter_sha256": "0" * 64 if corrupt_sha else digest,
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def _make_impl(adapter_path):
    return MindIESLAImpl(
        num_heads=4,
        head_size=64,
        softmax_scale=0.125,
        num_kv_heads=1,
        prefix="model.layers.1.self_attn.image_attn.attn",
        backend_kwargs={"adapter_path": str(adapter_path), "mask_policy": "hybrid"},
        role="hunyuan.diffusion",
    )


def test_adapter_is_installed_lazily_after_model_initialization(fake_mindiesd, tmp_path):
    _load_adapter.cache_clear()
    impl = _make_impl(_make_adapter(tmp_path))
    impl.sla.proj_l.weight.data.zero_()
    impl.sla.proj_l.bias.data.zero_()
    impl._install_adapter()
    torch.testing.assert_close(impl.sla.proj_l.weight, torch.full((64, 64), 2.0))
    torch.testing.assert_close(impl.sla.proj_l.bias, torch.full((64,), 2.0))
    assert impl._adapter_installed


def test_adapter_sha_mismatch_rejected(fake_mindiesd, tmp_path):
    _load_adapter.cache_clear()
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _make_impl(_make_adapter(tmp_path, corrupt_sha=True))


def test_gqa_repeat_is_idempotent_after_hunyuan_repeat():
    q = torch.zeros(1, 3, 4, 8)
    compressed_k = torch.zeros(1, 5, 1, 8)
    compressed_v = torch.ones_like(compressed_k)
    key, value = _repeat_gqa_kv(q, compressed_k, compressed_v)
    assert key.shape == value.shape == (1, 5, 4, 8)
    repeated_key, repeated_value = _repeat_gqa_kv(q, key, value)
    assert repeated_key is key
    assert repeated_value is value


def test_hybrid_supports_lq_not_equal_lk(fake_mindiesd, tmp_path):
    _load_adapter.cache_clear()
    impl = _make_impl(_make_adapter(tmp_path))
    query = torch.randn(1, 3, 4, 64)
    key = torch.randn(1, 5, 1, 64)
    value = torch.randn_like(key)
    mask = torch.ones(1, 1, 3, 5, dtype=torch.bool)
    metadata = AttentionMetadata(attn_mask=mask, full_attn_spans=[[(3, 5)]])

    output = impl._hybrid_masked_forward(query, key, value, metadata)

    assert output.shape == query.shape
    torch.testing.assert_close(output[:, 1:], query[:, 1:])
    assert _FakeSLA.last_shapes == ((1, 4, 2, 64), (1, 4, 5, 64), (1, 4, 5, 64))


def test_hybrid_rejects_non_suffix_span(fake_mindiesd, tmp_path):
    _load_adapter.cache_clear()
    impl = _make_impl(_make_adapter(tmp_path))
    query = torch.randn(1, 3, 4, 64)
    key = torch.randn(1, 5, 1, 64)
    value = torch.randn_like(key)
    metadata = AttentionMetadata(
        attn_mask=torch.ones(1, 1, 3, 5, dtype=torch.bool),
        full_attn_spans=[[(2, 4)]],
    )
    with pytest.raises(ValueError, match="image suffix ending at the KV length"):
        impl._hybrid_masked_forward(query, key, value, metadata)


def test_joint_sequence_parallel_is_rejected():
    metadata = AttentionMetadata(joint_query=torch.zeros(1))
    with pytest.raises(NotImplementedError, match="sequence_parallel_size=1"):
        MindIESLAImpl._reject_parallel_metadata(metadata)
