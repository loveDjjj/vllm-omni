#!/usr/bin/env bash
# Configurable inputs (environment variables):
#   PYTHON_BIN        Python executable. Default: python
#   VLLM_OMNI_ROOT    vLLM-Omni checkout. Default: repository root
#   DIFFUSERS_VERSION Required Diffusers version. Default: 0.38.0
#   CACHE_DIT_VERSION Required Cache-DiT version. Default: 1.3.0
#   PIP_INDEX_URL     Optional pip index URL inherited by pip.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_OMNI_ROOT="${VLLM_OMNI_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DIFFUSERS_VERSION="${DIFFUSERS_VERSION:-0.38.0}"
CACHE_DIT_VERSION="${CACHE_DIT_VERSION:-1.3.0}"

"${PYTHON_BIN}" -m pip install --upgrade --no-deps \
  "diffusers==${DIFFUSERS_VERSION}" \
  "cache-dit==${CACHE_DIT_VERSION}"
"${PYTHON_BIN}" -m pip install -e "${VLLM_OMNI_ROOT}" --no-deps

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata

import diffusers

assert hasattr(diffusers, "ContextParallelConfig"), diffusers.__version__
assert hasattr(diffusers, "ParallelConfig"), diffusers.__version__

from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    HunyuanImage3Model,
)

print("diffusers", diffusers.__version__)
print("cache-dit", importlib.metadata.version("cache-dit"))
print("HunyuanImage3 import OK:", HunyuanImage3Model.__name__)
PY
