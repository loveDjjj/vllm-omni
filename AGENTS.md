# HunyuanImage3 Instruct-Distil + SLA Development Rules

This branch adapts vLLM-Omni for `HunyuanImage-3.0-Instruct-Distil` and the
MindIE-SD `SparseLinearAttention` module on Ascend NPU.

Read `docs/design/hunyuan_image3_distil_sla.md` before changing code. Its phase
gates and acceptance criteria are requirements, not suggestions.

## Scope

- Keep the existing vLLM-Omni HunyuanImage3 implementation. Do not rewrite the
  model or pipeline.
- Reuse `mindiesd.layers.SparseLinearAttention`. Do not implement or copy an SLA
  kernel.
- Replace only image-diffusion attention. AR/text attention must remain Dense.
- Preserve existing Base and Instruct behavior. Distil behavior must be selected
  from checkpoint configuration, not from a repository-wide default.
- First support Ascend NPU, BF16, batch size 1, 1024x1024 and 8 denoising steps.

## Engineering Rules

- Pin and report the tested versions of vLLM, vLLM-Ascend, torch, torch-npu,
  CANN and MindIE-SD.
- Add tests with every behavioral change. CPU tests may use a fake SLA module,
  but final forward tests must run the real MindIE-SD module on NPU.
- Do not silently discard an attention mask. Either preserve its semantics,
  choose an explicitly documented Dense fallback, or fail with a clear error.
- Do not silently ignore missing or unexpected SLA adapter weights.
- Keep AR and diffusion attention roles independently configurable.
- Avoid unrelated formatting, API, registry or model refactors.
- Use the repository's existing lint, typing and test conventions.

## Required Validation Order

1. Dense Instruct-Distil correctness against official HunyuanImage-3.0.
2. SLA backend shape and weight-loading tests.
3. SLA first denoising step.
4. SLA later denoising step with prompt KV reuse (`Lq != Lk`).
5. Complete 8-step image generation.
6. TP=8 on Ascend; TP=16 only after TP=8 passes.

Do not report the work complete unless the checklist in the design document has
real command output and artifacts attached.
