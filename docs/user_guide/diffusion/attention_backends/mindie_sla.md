# MindIE SparseLinearAttention

`MINDIE_SLA` runs the MindIE-SD `SparseLinearAttention` operator in the
HunyuanImage-3.0-Instruct-Distil diffusion path. It does not replace the
autoregressive Hunyuan attention path and it does not implement another SLA
kernel.

The backend consumes the compact recovery artifact exported by the
HunyuanImage3-SLA training repository:

```text
adapter.safetensors
adapter_config.json
SHA256SUMS
```

At startup it validates the artifact architecture, format version, SHA256,
tensor names, shapes, finite values, tensor count, parameter count, and runtime
SLA settings. Every TP rank loads the same small `proj_l` parameters. Existing
Hunyuan TP continues to shard Q heads and shard or replicate KV heads; the
backend expands GQA KV only when the model has not already expanded it.

## Configuration

Use per-role selection so autoregressive attention remains dense:

```yaml
diffusion_attention_config:
  default:
    backend: TORCH_SDPA
  per_role:
    hunyuan.ar:
      backend: TORCH_SDPA
    hunyuan.diffusion:
      backend: MINDIE_SLA
      mindie_sla:
        adapter_path: ${oc.env:HUNYUAN_SLA_ADAPTER}
        topk: 0.125
        blkq: 64
        blkk: 128
        use_bf16: true
        mask_policy: hybrid
```

The repository includes complete Dense and SLA deployment files:

```text
vllm_omni/deploy/hunyuan_image_3_distil_dense.yaml
vllm_omni/deploy/hunyuan_image_3_distil_sla.yaml
```

Set the adapter directory before starting the SLA deployment:

```bash
export HUNYUAN_SLA_ADAPTER=/path/to/results/adapters/sla-step-200
vllm serve /path/to/HunyuanImage-3.0-Instruct-Distil \
  --omni \
  --trust-remote-code \
  --deploy-config vllm_omni/deploy/hunyuan_image_3_distil_sla.yaml \
  --enforce-eager
```

## Mask handling

Hunyuan uses mixed causal/full attention. With `mask_policy: hybrid`, only the
timestep/guidance prefix is evaluated by Dense SDPA. The generated-image query
suffix identified by `full_attn_spans` is evaluated by SLA against the complete
prompt plus image KV sequence. The backend requires one image span ending at
the valid KV length. It raises an error when that contract is not satisfied;
it never silently discards a mask.

`mask_policy: error` rejects every masked call. `mask_policy: dense_fallback`
is an explicit diagnostic mode that evaluates the full call with SDPA and logs
that choice.

## First-version limits

- Ascend NPU only, BF16 recommended.
- Hunyuan diffusion role only; AR remains dense.
- `max_num_seqs=1`.
- `sequence_parallel_size=1`, Ulysses/Ring/AllGather degrees all equal to 1.
- `cfg_parallel_size=1`; Distil runs one guidance-conditioned branch.
- Dense legacy KV ownership; Scheduler paged KV is not supported by this backend.
- TP8 is the first validation target. TP16 requires a separate full 8-step run.

The Distil checkpoint defaults to 8 denoising steps and guidance scale 2.5.
Explicit request values still override those defaults.
