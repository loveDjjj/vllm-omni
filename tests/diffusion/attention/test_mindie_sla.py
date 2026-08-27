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
    apply_attention_deltas,
)
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _FakeSLA(nn.Module):
    last_shapes = None
    calls = []

    def __init__(self, head_dim, **_kwargs):
        super().__init__()
        self.proj_l = nn.Linear(head_dim, head_dim, dtype=torch.float32)

    def forward(self, q, k, v):
        type(self).last_shapes = (tuple(q.shape), tuple(k.shape), tuple(v.shape))
        type(self).calls.append(type(self).last_shapes)
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


def _make_v2_adapter(tmp_path):
    tensors = {}
    for layer in range(2):
        tensors[f"layers.{layer}.sla.proj_l.weight"] = torch.full((2, 2), float(layer + 1))
        tensors[f"layers.{layer}.sla.proj_l.bias"] = torch.full((2,), float(layer + 1))
        tensors[f"layers.{layer}.qkv_delta.weight"] = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        tensors[f"layers.{layer}.o_delta.weight"] = torch.full((4, 4), float(layer + 3))
    weights = tmp_path / "adapter.safetensors"
    save_file(tensors, str(weights))
    config = {
        "format_version": 2,
        "architecture": "HunyuanImage3SparseLinearAttentionAdapter",
        "num_layers": 2,
        "head_dim": 2,
        "hidden_size": 4,
        "q_heads": 2,
        "kv_heads": 1,
        "trained_components": ["proj_l", "qkv_delta", "o_delta"],
        "topk": 0.125,
        "blkq": 64,
        "blkk": 128,
        "compute_dtype": "bfloat16",
        "tensor_count": len(tensors),
        "parameter_count": sum(t.numel() for t in tensors.values()),
        "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path, tensors


def _make_v3_adapter(tmp_path):
    tensors = {}
    for layer in range(2):
        tensors[f"layers.{layer}.sla.proj_l.weight"] = torch.zeros(2, 2)
        tensors[f"layers.{layer}.sla.proj_l.bias"] = torch.zeros(2)
        tensors[f"layers.{layer}.qkv_lora.a.weight"] = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        tensors[f"layers.{layer}.qkv_lora.b.weight"] = torch.ones(8, 2)
        tensors[f"layers.{layer}.o_lora.a.weight"] = torch.ones(2, 4)
        tensors[f"layers.{layer}.o_lora.b.weight"] = torch.full((4, 2), 2.0)
        for expert in range(2):
            prefix = f"layers.{layer}.moe.experts.{expert}.down_lora"
            tensors[f"{prefix}.a.weight"] = torch.ones(1, 3)
            tensors[f"{prefix}.b.weight"] = torch.full((4, 1), float(expert + 1))
    weights = tmp_path / "adapter.safetensors"
    save_file(tensors, str(weights))
    config = {
        "format_version": 3,
        "architecture": "HunyuanImage3SparseLinearAttentionAdapter",
        "num_layers": 2,
        "head_dim": 2,
        "hidden_size": 4,
        "q_heads": 2,
        "kv_heads": 1,
        "num_experts": 2,
        "moe_intermediate_size": 3,
        "attention_lora_rank": 2,
        "attention_lora_alpha": 2,
        "moe_down_lora_rank": 1,
        "moe_down_lora_alpha": 1,
        "trained_components": ["proj_l", "qkv_lora", "o_lora", "moe_down_lora"],
        "topk": 0.125,
        "blkq": 64,
        "blkk": 128,
        "compute_dtype": "bfloat16",
        "tensor_count": len(tensors),
        "parameter_count": sum(t.numel() for t in tensors.values()),
        "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path, tensors


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


def test_mindie_sla_config_is_role_scoped_and_serialized(monkeypatch):
    monkeypatch.setenv("SLA_TEST_ADAPTER", "/tmp/sla-adapter")
    config = AttentionConfig(
        default={"backend": "TORCH_SDPA"},
        per_role={
            "hunyuan": {
                "diffusion": {
                    "backend": "MINDIE_SLA",
                    "mindie_sla": {"adapter_path": "$SLA_TEST_ADAPTER", "mask_policy": "hybrid"},
                }
            }
        },
    )
    spec, _ = config.resolve_with_source(role="hunyuan.diffusion")
    assert spec is not None
    assert spec.backend_kwargs() == {
        "adapter_path": "/tmp/sla-adapter",
        "topk": 0.125,
        "blkq": 64,
        "blkk": 128,
        "use_bf16": True,
        "inner_precise": None,
        "mask_policy": "hybrid",
    }
    ar_spec, _ = config.resolve_with_source(role="hunyuan.ar")
    assert ar_spec is not None and ar_spec.backend == "TORCH_SDPA"


def test_mindie_sla_requires_adapter_config():
    with pytest.raises(ValueError, match="requires a mindie_sla block"):
        AttentionSpec(backend="MINDIE_SLA")


def test_adapter_is_installed_lazily_after_model_initialization(fake_mindiesd, tmp_path):
    _load_adapter.cache_clear()
    impl = _make_impl(_make_adapter(tmp_path))
    impl.sla.proj_l.weight.data.zero_()
    impl.sla.proj_l.bias.data.zero_()
    impl._install_adapter()
    torch.testing.assert_close(impl.sla.proj_l.weight, torch.full((64, 64), 2.0))
    torch.testing.assert_close(impl.sla.proj_l.bias, torch.full((64,), 2.0))
    assert impl._adapter_installed


def test_adapter_declares_runtime_parameters_as_externally_loaded(fake_mindiesd, tmp_path):
    _load_adapter.cache_clear()
    impl = _make_impl(_make_adapter(tmp_path))

    assert impl.get_externally_loaded_parameter_names("model.layers.1.attention") == {
        "model.layers.1.attention.sla.proj_l.weight",
        "model.layers.1.attention.sla.proj_l.bias",
    }


def test_adapter_sha_mismatch_rejected(fake_mindiesd, tmp_path):
    _load_adapter.cache_clear()
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _make_impl(_make_adapter(tmp_path, corrupt_sha=True))


def test_v2_attention_deltas_are_added_before_tp_loading(tmp_path):
    _load_adapter.cache_clear()
    adapter, tensors = _make_v2_adapter(tmp_path)
    base_weights = [
        ("model.layers.0.self_attn.qkv_proj.weight", torch.zeros(8, 4)),
        ("model.layers.0.self_attn.o_proj.weight", torch.ones(4, 4)),
        ("model.layers.0.input_layernorm.weight", torch.ones(4)),
        ("vision_model.encoder.layers.0.self_attn.k_proj.weight", torch.full((6, 6), 7.0)),
        ("vision_model.encoder.layers.0.self_attn.o_proj.weight", torch.full((6, 6), 8.0)),
    ]

    loaded = dict(apply_attention_deltas(iter(base_weights), str(adapter)))

    torch.testing.assert_close(
        loaded["model.layers.0.self_attn.qkv_proj.weight"],
        tensors["layers.0.qkv_delta.weight"],
    )
    torch.testing.assert_close(
        loaded["model.layers.0.self_attn.o_proj.weight"],
        torch.ones(4, 4) + tensors["layers.0.o_delta.weight"],
    )
    torch.testing.assert_close(loaded["model.layers.0.input_layernorm.weight"], torch.ones(4))
    torch.testing.assert_close(
        loaded["vision_model.encoder.layers.0.self_attn.k_proj.weight"],
        torch.full((6, 6), 7.0),
    )
    torch.testing.assert_close(
        loaded["vision_model.encoder.layers.0.self_attn.o_proj.weight"],
        torch.full((6, 6), 8.0),
    )


def test_v2_packed_qkv_delta_supports_split_checkpoints(tmp_path):
    _load_adapter.cache_clear()
    adapter, tensors = _make_v2_adapter(tmp_path)
    names_and_shapes = {
        "q_proj": (4, 4),
        "k_proj": (2, 4),
        "v_proj": (2, 4),
    }
    base_weights = [
        (f"model.layers.0.self_attn.{name}.weight", torch.zeros(shape))
        for name, shape in names_and_shapes.items()
    ]

    loaded = dict(apply_attention_deltas(iter(base_weights), str(adapter)))
    packed = tensors["layers.0.qkv_delta.weight"].reshape(1, 4, 2, 4)
    q, k, v = torch.split(packed, (2, 1, 1), dim=1)
    expected = {
        "q_proj": q.reshape(4, 4),
        "k_proj": k.reshape(2, 4),
        "v_proj": v.reshape(2, 4),
    }
    for projection, tensor in expected.items():
        torch.testing.assert_close(loaded[f"model.layers.0.self_attn.{projection}.weight"], tensor)


def test_v3_attention_and_moe_lora_are_merged_before_parallel_loading(tmp_path):
    _load_adapter.cache_clear()
    adapter, tensors = _make_v3_adapter(tmp_path)
    base_weights = [
        ("model.layers.0.self_attn.qkv_proj.weight", torch.zeros(8, 4)),
        ("model.layers.0.self_attn.o_proj.weight", torch.ones(4, 4)),
        ("model.layers.0.mlp.experts.0.down_proj.weight", torch.zeros(4, 3)),
        ("model.layers.0.mlp.experts.1.down_proj.weight", torch.ones(4, 3)),
        ("model.layers.0.mlp.shared_mlp.down_proj.weight", torch.full((4, 3), 7.0)),
    ]

    loaded = dict(apply_attention_deltas(iter(base_weights), str(adapter)))

    qkv = tensors["layers.0.qkv_lora.b.weight"] @ tensors["layers.0.qkv_lora.a.weight"]
    output = tensors["layers.0.o_lora.b.weight"] @ tensors["layers.0.o_lora.a.weight"]
    expert0 = (
        tensors["layers.0.moe.experts.0.down_lora.b.weight"]
        @ tensors["layers.0.moe.experts.0.down_lora.a.weight"]
    )
    expert1 = (
        tensors["layers.0.moe.experts.1.down_lora.b.weight"]
        @ tensors["layers.0.moe.experts.1.down_lora.a.weight"]
    )
    torch.testing.assert_close(loaded[base_weights[0][0]], qkv)
    torch.testing.assert_close(loaded[base_weights[1][0]], torch.ones(4, 4) + output)
    torch.testing.assert_close(loaded[base_weights[2][0]], expert0)
    torch.testing.assert_close(loaded[base_weights[3][0]], torch.ones(4, 3) + expert1)
    torch.testing.assert_close(loaded[base_weights[4][0]], torch.full((4, 3), 7.0))


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
    _FakeSLA.calls = []
    impl = _make_impl(_make_adapter(tmp_path))
    first_query = torch.randn(1, 5, 4, 64)
    key = torch.randn(1, 5, 1, 64)
    value = torch.randn_like(key)
    first_mask = torch.ones(1, 1, 5, 5, dtype=torch.bool)
    impl._hybrid_masked_forward(
        first_query,
        key,
        value,
        AttentionMetadata(
            attn_mask=first_mask,
            full_attn_spans=[[(3, 5)]],
            extra={"sla_static_prefix_lens": [2]},
        ),
    )
    query = torch.randn(1, 3, 4, 64)
    mask = torch.ones(1, 1, 3, 5, dtype=torch.bool)
    metadata = AttentionMetadata(attn_mask=mask, full_attn_spans=[[(3, 5)]])

    output = impl._hybrid_masked_forward(query, key, value, metadata)

    assert output.shape == query.shape
    torch.testing.assert_close(output[:, 1:], query[:, 1:])
    assert _FakeSLA.last_shapes == ((1, 4, 5, 64), (1, 4, 5, 64), (1, 4, 5, 64))


def test_hybrid_query_prefix_excludes_three_dynamic_special_tokens(fake_mindiesd, tmp_path):
    _load_adapter.cache_clear()
    impl = _make_impl(_make_adapter(tmp_path))
    first_query = torch.randn(1, 8, 4, 64)
    first_key = torch.randn(1, 8, 1, 64)
    first_value = torch.randn_like(first_key)
    first_metadata = AttentionMetadata(
        attn_mask=torch.ones(1, 1, 8, 8, dtype=torch.bool),
        full_attn_spans=[[(5, 8)]],
        extra={"sla_static_prefix_lens": [2]},
    )
    impl._hybrid_masked_forward(first_query, first_key, first_value, first_metadata)

    query = torch.randn(1, 6, 4, 64)
    metadata = AttentionMetadata(
        attn_mask=torch.ones(1, 1, 6, 8, dtype=torch.bool),
        full_attn_spans=[[(5, 8)]],
        extra={"sla_static_prefix_lens": [2]},
    )
    output = impl._hybrid_masked_forward(query, first_key, first_value, metadata)

    assert impl._query_prefix_cache[0].shape[1] == 2
    assert output.shape == query.shape


def test_hybrid_static_prefix_may_contain_earlier_image_spans(fake_mindiesd, tmp_path):
    _load_adapter.cache_clear()
    impl = _make_impl(_make_adapter(tmp_path))
    first_query = torch.randn(1, 10, 4, 64)
    key = torch.randn(1, 10, 1, 64)
    value = torch.randn_like(key)
    spans = [[(1, 3), (8, 10)]]
    impl._hybrid_masked_forward(
        first_query,
        key,
        value,
        AttentionMetadata(
            attn_mask=torch.ones(1, 1, 10, 10, dtype=torch.bool),
            full_attn_spans=spans,
            extra={"sla_static_prefix_lens": [5]},
        ),
    )

    query = torch.randn(1, 5, 4, 64)
    output = impl._hybrid_masked_forward(
        query,
        key,
        value,
        AttentionMetadata(
            attn_mask=torch.ones(1, 1, 5, 10, dtype=torch.bool),
            full_attn_spans=spans,
            extra={"sla_static_prefix_lens": [5]},
        ),
    )

    assert impl._query_prefix_cache[0].shape[1] == 5
    assert output.shape == query.shape


def test_module_forward_dispatches_to_npu_implementation(fake_mindiesd, tmp_path):
    _load_adapter.cache_clear()
    impl = _make_impl(_make_adapter(tmp_path))
    query = torch.randn(1, 3, 4, 64)
    key = torch.randn(1, 3, 1, 64)
    value = torch.randn_like(key)

    output = impl(query, key, value)

    assert output.shape == query.shape
    torch.testing.assert_close(output, query)


def test_hybrid_supports_multiple_image_spans_and_trailing_tokens(fake_mindiesd, tmp_path):
    _load_adapter.cache_clear()
    _FakeSLA.calls = []
    impl = _make_impl(_make_adapter(tmp_path))
    query = torch.randn(1, 12, 4, 64)
    key = torch.randn(1, 12, 1, 64)
    value = torch.randn_like(key)
    metadata = AttentionMetadata(
        attn_mask=torch.ones(1, 1, 12, 12, dtype=torch.bool).tril(),
        full_attn_spans=[[(1, 4), (6, 9)]],
    )

    output = impl._hybrid_masked_forward(query, key, value, metadata)

    assert output.shape == query.shape
    assert _FakeSLA.calls == [
        ((1, 4, 3, 64), (1, 4, 4, 64), (1, 4, 4, 64)),
        ((1, 4, 3, 64), (1, 4, 9, 64), (1, 4, 9, 64)),
    ]


def test_joint_sequence_parallel_is_rejected():
    metadata = AttentionMetadata(joint_query=torch.zeros(1))
    with pytest.raises(NotImplementedError, match="sequence_parallel_size=1"):
        MindIESLAImpl._reject_parallel_metadata(metadata)
