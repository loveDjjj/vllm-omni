# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 as hy3_module
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import ImageInfo
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    _STEP_AR_KV,
    _STEP_CFG_FACTOR,
    _STEP_GENERATOR,
    _STEP_GUIDANCE_SCALE,
    _STEP_INPUT_IDS,
    _STEP_MODEL_KWARGS,
    _STEP_PROMPT_KV,
    _STEP_TEACHER_TRAJECTORY,
    HunyuanImage3Pipeline,
)
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _pipeline():
    pipeline = object.__new__(HunyuanImage3Pipeline)
    pipeline.hf_config = SimpleNamespace(cfg_distilled=False, use_meanflow=False)
    pipeline._tkwrapper = SimpleNamespace(pad_token_id=0)
    pipeline.od_config = SimpleNamespace(
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
        parallel_config=SimpleNamespace(sequence_parallel_size=1, cfg_parallel_size=1),
        cache_backend=None,
        diffusion_kv_cache_skip_step_indices=None,
    )
    pipeline._pipeline = SimpleNamespace()
    return pipeline


def _state(request_id: str, step_index: int) -> StepRequestState:
    state = StepRequestState(
        request_id=request_id,
        sampling=SimpleNamespace(),
        prompt="prompt",
    )
    state.step_index = step_index
    state.timesteps = torch.tensor([1.0, 0.5, 0.25, 0.0])
    state.latents = torch.zeros(1, 4, 8, 8)
    state.extra = {
        _STEP_CFG_FACTOR: 1,
        _STEP_AR_KV: None,
        _STEP_INPUT_IDS: None,
        _STEP_GUIDANCE_SCALE: 1.0,
        _STEP_MODEL_KWARGS: {
            "num_image_tokens": 17,
            "ar_kv_reuse_len": 0,
        },
    }
    return state


def _sampling_params(**extra_args):
    return SimpleNamespace(
        timesteps=None,
        sigmas=None,
        num_outputs_per_prompt=None,
        extra_args=extra_args,
        height=512,
        width=512,
        num_inference_steps=4,
        guidance_scale=1.0,
        guidance_scale_provided=True,
        guidance_rescale=0.0,
        generator=None,
    )


def test_meanflow_image_info_exports_timestep_r_layout_flag():
    image_info = ImageInfo(
        image_type="gen_image",
        token_width=8,
        token_height=8,
        add_timestep_token=True,
        add_guidance_token=True,
        add_timestep_r_token=True,
    )

    assert image_info.meta_info["add_timestep_r_token"] is True


def test_hunyuan_step_group_key_ignores_step_index_for_later_steps():
    pipeline = _pipeline()
    states = [_state("req-0", 1), _state("req-1", 3)]

    groups = pipeline._split_step_groups(states)

    assert len(groups) == 1
    assert [state.request_id for state in groups[0]] == ["req-0", "req-1"]


@pytest.mark.parametrize(
    ("sampling", "prompt_item", "expected_model_bot_task", "expected_system_bot_task"),
    [
        pytest.param(
            _sampling_params(bot_task="think_recaption", use_system_prompt="dynamic"),
            {"prompt": "prompt", "bot_task": "vanilla"},
            "think",
            "think",
            id="extra-args-precedence",
        ),
        pytest.param(
            _sampling_params(use_system_prompt="dynamic"),
            {"prompt": "prompt", "bot_task": "vanilla"},
            "image",
            "image",
            id="prompt-dict-fallback",
        ),
        pytest.param(
            _sampling_params(use_system_prompt="dynamic"),
            {"prompt": "prompt"},
            "auto",
            "image",
            id="default-auto-system-prompt",
        ),
    ],
)
def test_prepare_encode_preserves_normal_hunyuan_bot_task_semantics(
    monkeypatch,
    sampling,
    prompt_item,
    expected_model_bot_task,
    expected_system_bot_task,
):
    pipeline = _pipeline()
    captured: dict[str, object] = {}

    def fake_get_system_prompt(sys_type, bot_task, system_prompt=None):
        del sys_type, system_prompt
        captured["system_prompt_bot_task"] = bot_task
        return "system prompt"

    def fake_prepare_model_inputs(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after prepare_model_inputs")

    monkeypatch.setattr(hy3_module, "get_system_prompt", fake_get_system_prompt)
    pipeline.prepare_model_inputs = fake_prepare_model_inputs
    state = StepRequestState(
        request_id="req-bot-task",
        sampling=sampling,
        prompt=prompt_item,
    )

    with pytest.raises(RuntimeError, match="stop after prepare_model_inputs"):
        pipeline.prepare_encode(state)

    assert captured["bot_task"] == expected_model_bot_task
    assert captured["system_prompt_bot_task"] == expected_system_bot_task


def test_forward_uses_same_hunyuan_bot_task_semantics(monkeypatch):
    pipeline = _pipeline()
    captured: dict[str, object] = {}

    def fake_get_system_prompt(sys_type, bot_task, system_prompt=None):
        del sys_type, system_prompt
        captured["system_prompt_bot_task"] = bot_task
        return "system prompt"

    def fake_prepare_model_inputs(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after prepare_model_inputs")

    monkeypatch.setattr(hy3_module, "get_system_prompt", fake_get_system_prompt)
    pipeline.prepare_model_inputs = fake_prepare_model_inputs
    req = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="req-forward-bot-task",
                sampling_params=_sampling_params(bot_task="think_recaption", use_system_prompt="dynamic"),
                prompt={"prompt": "prompt", "bot_task": "vanilla"},
            )
        ]
    )

    with pytest.raises(RuntimeError, match="stop after prepare_model_inputs"):
        pipeline.forward(req)

    assert captured["bot_task"] == "think"
    assert captured["system_prompt_bot_task"] == "think"


def test_grouped_denoise_rejects_non_sdpa_attention_backend():
    pipeline = _pipeline()
    pipeline.od_config.diffusion_attention_config = AttentionConfig(default=AttentionSpec(backend="FLASH_ATTN"))

    with pytest.raises(ValueError, match="only supports TORCH_SDPA"):
        pipeline._ensure_grouped_attention_backend_supported(2)


def test_single_denoise_allows_non_sdpa_attention_backend():
    pipeline = _pipeline()
    pipeline.od_config.diffusion_attention_config = AttentionConfig(default=AttentionSpec(backend="FLASH_ATTN"))

    pipeline._ensure_grouped_attention_backend_supported(1)


def test_grouped_denoise_allows_sdpa_attention_backend():
    pipeline = _pipeline()

    pipeline._ensure_grouped_attention_backend_supported(2)


def test_step_scheduler_preserves_latent_dtype_for_mixed_progress_batches():
    pipeline = _pipeline()
    pipeline._pipeline = SimpleNamespace(prepare_extra_func_kwargs=lambda step, kwargs: {})

    class FakeScheduler:
        def step(self, noise_pred, timestep, latents, **kwargs):
            del timestep, kwargs
            return (latents.float() + noise_pred.float(),)

    state = _state("req", 0)
    state.timesteps = torch.tensor([1.0])
    state.scheduler = FakeScheduler()
    state.latents = torch.zeros(1, 4, 8, 8, dtype=torch.bfloat16)
    state.extra[_STEP_GENERATOR] = None

    pipeline.step_scheduler(state, torch.ones_like(state.latents, dtype=torch.float32))

    assert state.latents.dtype == torch.bfloat16
    assert state.step_index == 1


def test_step_scheduler_records_dense_teacher_trajectory():
    pipeline = _pipeline()
    pipeline.hf_config.use_meanflow = True
    pipeline._pipeline = SimpleNamespace(prepare_extra_func_kwargs=lambda step, kwargs: {})

    class FakeScheduler:
        def get_timestep_r(self, timestep):
            return timestep - 0.25

        def step(self, noise_pred, timestep, latents, **kwargs):
            del timestep, kwargs
            return (latents.float() + noise_pred.float(),)

    state = _state("teacher", 0)
    state.timesteps = torch.tensor([1.0])
    state.scheduler = FakeScheduler()
    state.latents = torch.zeros(1, 4, 8, 8, dtype=torch.bfloat16)
    state.extra[_STEP_GENERATOR] = None
    state.extra[_STEP_TEACHER_TRAJECTORY] = {
        "latents": [state.latents[0].float().clone()],
        "predictions": [],
        "timesteps": [],
        "timesteps_r": [],
        "condition": {"input_ids": torch.ones(1, 3, dtype=torch.long)},
        "metadata": {"prompt": "test"},
    }

    prediction = torch.ones_like(state.latents, dtype=torch.float32)
    pipeline.step_scheduler(state, prediction)
    result = pipeline.post_decode(state)
    trajectory = result.output["payload"]["trajectory"]

    assert trajectory["latents"].shape == (2, 4, 8, 8)
    assert trajectory["predictions"].shape == (1, 4, 8, 8)
    torch.testing.assert_close(trajectory["timesteps"], torch.tensor([1.0]))
    torch.testing.assert_close(trajectory["timesteps_r"], torch.tensor([0.75]))
    assert result.to_cpu is True


def test_later_step_merge_shifts_spans_without_polluting_request_state():
    pipeline = _pipeline()
    states = [_state("short", 2), _state("long", 4)]
    states[0].extra[_STEP_MODEL_KWARGS].update(
        {
            "attention_mask": torch.ones(1, 1, 3, 5, dtype=torch.bool),
            "full_attn_spans": [[(2, 5)]],
        }
    )
    states[1].extra[_STEP_MODEL_KWARGS].update(
        {
            "attention_mask": torch.ones(1, 1, 3, 7, dtype=torch.bool),
            "full_attn_spans": [[(4, 7)]],
        }
    )
    states[0].extra[_STEP_PROMPT_KV] = [{"lens": torch.tensor([2])}]
    states[1].extra[_STEP_PROMPT_KV] = [{"lens": torch.tensor([4])}]

    row_state_indexes = [0, 1]
    row_branches = [0, 0]
    _, merged = pipeline._merge_step_model_inputs(
        states,
        row_state_indexes,
        row_branches,
        first_step=False,
    )

    assert merged["attention_mask"].shape == (2, 1, 3, 7)
    assert merged["full_attn_spans"] == [[(4, 7)], [(4, 7)]]

    pipeline._split_merged_kwargs_to_states(states, merged, row_state_indexes, row_branches)

    assert states[0].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (1, 1, 3, 5)
    assert states[1].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (1, 1, 3, 7)
    assert states[0].extra[_STEP_MODEL_KWARGS]["full_attn_spans"] == [[(2, 5)]]
    assert states[1].extra[_STEP_MODEL_KWARGS]["full_attn_spans"] == [[(4, 7)]]


def test_later_step_merge_allows_request_local_step_counts_and_guidance_values():
    pipeline = _pipeline()
    states = [_state("req-0", 1), _state("req-1", 3)]
    for idx, state in enumerate(states):
        state.extra[_STEP_MODEL_KWARGS].update(
            {
                "attention_mask": torch.ones(1, 1, 2, 4, dtype=torch.bool),
                "full_attn_spans": [[(2, 4)]],
                "guidance_scale": 3.0 + idx,
                "num_inference_steps": 20 + idx,
            }
        )
        state.extra[_STEP_PROMPT_KV] = [{"lens": torch.tensor([2])}]

    _, merged = pipeline._merge_step_model_inputs(
        states,
        row_state_indexes=[0, 1],
        row_branches=[0, 0],
        first_step=False,
    )

    assert "guidance_scale" not in merged
    assert "num_inference_steps" not in merged


def test_distilled_later_step_injects_guidance_without_cfg_batch(monkeypatch):
    pipeline = _pipeline()
    pipeline.hf_config.cfg_distilled = True
    monkeypatch.setattr(HunyuanImage3Pipeline, "device", property(lambda self: torch.device("cpu")))
    state = _state("distilled", 1)
    state.extra[_STEP_GUIDANCE_SCALE] = 2.5
    state.extra[_STEP_MODEL_KWARGS].update(
        {
            "attention_mask": torch.ones(1, 1, 2, 5, dtype=torch.bool),
            "full_attn_spans": [[(3, 5)]],
        }
    )
    state.extra[_STEP_PROMPT_KV] = [
        {
            "key": torch.zeros(1, 3, 1, 1),
            "value": torch.zeros(1, 3, 1, 1),
            "lens": torch.tensor([3]),
        }
    ]
    captured: dict[str, object] = {}
    pipeline._restore_prompt_kv_cache = lambda *_args: None

    def fake_prepare_inputs(input_ids, images, timestep, **model_kwargs):
        captured.update(model_kwargs)
        captured["latent_batch"] = images.shape[0]
        return {}

    pipeline.prepare_inputs_for_generation = fake_prepare_inputs
    pipeline.forward_call = lambda **_kwargs: {"diffusion_prediction": torch.ones(1, 1)}
    pipeline._update_model_kwargs_for_generation = lambda _output, model_kwargs: model_kwargs

    output = pipeline._denoise_step_group([state])

    assert captured["latent_batch"] == 1
    assert captured["guidance"].dtype == torch.bfloat16
    torch.testing.assert_close(captured["guidance"], torch.tensor([2500.0], dtype=torch.bfloat16))
    torch.testing.assert_close(output, torch.ones(1, 1))


def test_meanflow_later_step_injects_scheduler_timestep_r(monkeypatch):
    pipeline = _pipeline()
    pipeline.hf_config.cfg_distilled = True
    pipeline.hf_config.use_meanflow = True
    monkeypatch.setattr(HunyuanImage3Pipeline, "device", property(lambda self: torch.device("cpu")))
    state = _state("meanflow", 1)
    state.scheduler = SimpleNamespace(get_timestep_r=lambda timestep: timestep - 0.125)
    state.extra[_STEP_MODEL_KWARGS].update(
        {
            "attention_mask": torch.ones(1, 1, 2, 5, dtype=torch.bool),
            "full_attn_spans": [[(3, 5)]],
        }
    )
    state.extra[_STEP_PROMPT_KV] = [
        {
            "key": torch.zeros(1, 3, 1, 1),
            "value": torch.zeros(1, 3, 1, 1),
            "lens": torch.tensor([3]),
        }
    ]
    captured = {}
    pipeline._restore_prompt_kv_cache = lambda *_args: None

    def fake_prepare_inputs(input_ids, images, timestep, **model_kwargs):
        captured.update(model_kwargs)
        return {}

    pipeline.prepare_inputs_for_generation = fake_prepare_inputs
    pipeline.forward_call = lambda **_kwargs: {"diffusion_prediction": torch.ones(1, 1)}
    pipeline._update_model_kwargs_for_generation = lambda _output, model_kwargs: model_kwargs

    pipeline._denoise_step_group([state])

    torch.testing.assert_close(captured["timestep_r"], torch.tensor([state.current_timestep - 0.125]))


@pytest.mark.parametrize(
    ("cfg_distilled", "use_meanflow", "special_tokens"),
    [(False, False, 1), (True, False, 2), (True, True, 3)],
)
def test_ragged_final_layer_removes_checkpoint_specific_special_tokens(
    cfg_distilled, use_meanflow, special_tokens
):
    pipeline = _pipeline()
    pipeline.hf_config.cfg_distilled = cfg_distilled
    pipeline.hf_config.use_meanflow = use_meanflow
    captured: dict[str, torch.Tensor] = {}
    pipeline.time_embed_2 = lambda timestep: timestep

    def fake_final_layer(image_output, timestep_emb, token_h, token_w):
        del timestep_emb, token_h, token_w
        captured["image_output"] = image_output
        return image_output

    pipeline.final_layer = fake_final_layer
    hidden_states = torch.arange((special_tokens + 3) * 2, dtype=torch.float32).reshape(1, special_tokens + 3, 2)

    output = pipeline.ragged_final_layer(
        hidden_states,
        image_mask=None,
        timestep=torch.tensor([1.0]),
        token_h=1,
        token_w=3,
        first_step=False,
    )

    torch.testing.assert_close(captured["image_output"], hidden_states[:, special_tokens:])
    torch.testing.assert_close(output, hidden_states[:, special_tokens:])


@pytest.mark.parametrize(
    ("request_id", "mutate_state", "error_match"),
    [
        pytest.param(
            "broken-req",
            lambda state: state.extra.pop(_STEP_MODEL_KWARGS),
            "broken-req",
            id="missing-model-kwargs",
        ),
        pytest.param(
            "bad-cfg",
            lambda state: state.extra.__setitem__(_STEP_CFG_FACTOR, 3),
            "bad-cfg",
            id="unsupported-cfg-factor",
        ),
    ],
)
def test_denoise_step_reports_invalid_group_state_with_request_id(request_id, mutate_state, error_match):
    pipeline = _pipeline()
    state = _state(request_id, 0)
    mutate_state(state)

    with pytest.raises(ValueError, match=error_match):
        pipeline.denoise_step(InputBatch.make_batch([state]))


def test_denoise_step_uses_input_batch_group_order_and_splits_back(monkeypatch):
    pipeline = _pipeline()
    monkeypatch.setattr(HunyuanImage3Pipeline, "device", property(lambda self: torch.device("cpu")))
    states = [_state("req-0", 1), _state("req-1", 3)]
    for idx, state in enumerate(states):
        prefix_len = 2 + idx * 2
        state.latents = torch.full((1, 1), float(idx))
        state.extra[_STEP_CFG_FACTOR] = 2
        state.extra[_STEP_GUIDANCE_SCALE] = 1.0
        state.extra[_STEP_INPUT_IDS] = None
        state.extra[_STEP_MODEL_KWARGS].update(
            {
                "attention_mask": torch.ones(2, 1, 2, prefix_len + 2, dtype=torch.bool),
                "full_attn_spans": [[(prefix_len, prefix_len + 2)], [(prefix_len, prefix_len + 2)]],
            }
        )
        state.extra[_STEP_PROMPT_KV] = [
            {
                "key": torch.zeros(2, prefix_len, 1, 1),
                "value": torch.zeros(2, prefix_len, 1, 1),
                "lens": torch.tensor([prefix_len, prefix_len]),
            }
        ]

    captured = {}

    def fake_restore_prompt_kv_cache(states_arg, row_state_indexes, row_branches):
        del states_arg
        captured["row_state_indexes"] = list(row_state_indexes)
        captured["row_branches"] = list(row_branches)

    def fake_prepare_inputs_for_generation(input_ids, images, timestep, **model_kwargs):
        captured["input_ids"] = input_ids
        captured["images"] = images.clone()
        captured["timestep"] = timestep.clone()
        captured["merged_attention_mask_shape"] = tuple(model_kwargs["attention_mask"].shape)
        captured["merged_full_attn_spans"] = model_kwargs["full_attn_spans"]
        return {"model_kwargs": model_kwargs}

    pipeline._restore_prompt_kv_cache = fake_restore_prompt_kv_cache
    pipeline.prepare_inputs_for_generation = fake_prepare_inputs_for_generation
    pipeline.forward_call = lambda **kwargs: {"diffusion_prediction": torch.tensor([[10.0], [20.0], [1.0], [2.0]])}
    pipeline._update_model_kwargs_for_generation = lambda model_output, model_kwargs: model_kwargs
    pipeline._pipeline = SimpleNamespace(cfg_operator=lambda cond, uncond, scale, step: cond + uncond)

    batch = InputBatch.make_batch(states)
    out = pipeline.denoise_step(batch)

    assert captured["row_state_indexes"] == [0, 1, 0, 1]
    assert captured["row_branches"] == [0, 0, 1, 1]
    assert captured["input_ids"] is None
    assert tuple(captured["images"].shape) == (4, 1)
    assert captured["timestep"].tolist() == [0.5, 0.0, 0.5, 0.0]
    assert captured["merged_attention_mask_shape"] == (4, 1, 2, 6)
    assert captured["merged_full_attn_spans"] == [[(4, 6)], [(4, 6)], [(4, 6)], [(4, 6)]]
    torch.testing.assert_close(out, torch.tensor([[11.0], [22.0]]))
    assert states[0].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (2, 1, 2, 4)
    assert states[1].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (2, 1, 2, 6)
    assert states[0].extra[_STEP_MODEL_KWARGS]["full_attn_spans"] == [[(2, 4)], [(2, 4)]]
    assert states[1].extra[_STEP_MODEL_KWARGS]["full_attn_spans"] == [[(4, 6)], [(4, 6)]]
