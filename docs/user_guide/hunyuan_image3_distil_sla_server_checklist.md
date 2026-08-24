# HunyuanImage3 Distil SLA 服务器验收命令

本文档由实现者持续追加实际可运行命令。不得只保留伪命令。

## 当前开发状态（2026-08-24）

```text
Phase A Dense Instruct-Distil code: IMPLEMENTED_LOCAL_ONLY
Phase B AR/diffusion role: IMPLEMENTED_LOCAL_ONLY
Phase C MINDIE_SLA backend: IMPLEMENTED_LOCAL_ONLY
Phase D hybrid attention mask policy: IMPLEMENTED_LOCAL_ONLY
Phase E SLA adapter validation/loading: IMPLEMENTED_LOCAL_ONLY
Phase F TP8/EP8 runtime: NOT_TESTED_ON_NPU
```

本地工作站为 Apple Silicon macOS，不具备 Ascend NPU。按照设计文档的阶段门禁，
Phase A-E 的代码、静态检查、CPU 测试和隔离 smoke test 已完成；Dense/SLA 8-step、
官方输出对齐和 TP8/EP8 必须在 Ascend 服务器执行，不能用本地模拟结果替代。

## 1. 环境记录

```bash
python - <<'PY'
import importlib.metadata

for name in (
    "vllm", "vllm-omni", "vllm-ascend", "torch", "torch-npu",
    "safetensors", "diffusers", "cache-dit",
):
    try:
        print(name, importlib.metadata.version(name))
    except Exception as exc:
        print(name, "NOT_INSTALLED", exc)
PY

npu-smi info
```

记录 CANN：

```bash
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
```

## 2. 路径检查

```bash
test -f /mnt/share/r50063443/HunyuanImage-3.0-Instruct-Distil/config.json
export HUNYUAN_SLA_ADAPTER=/mnt/share/r50063443/HunyuanImage3-SLA/results/adapters/sla-step-200
test -f "$HUNYUAN_SLA_ADAPTER/adapter.safetensors"
test -f "$HUNYUAN_SLA_ADAPTER/adapter_config.json"
sha256sum -c "$HUNYUAN_SLA_ADAPTER/SHA256SUMS"
```

验证部署 YAML 能解析环境变量，而不是把路径保留为字面量：

```bash
python - <<'PY'
import os
from omegaconf import OmegaConf

config = OmegaConf.load("vllm_omni/deploy/hunyuan_image_3_distil_sla.yaml")
resolved = OmegaConf.to_container(config, resolve=True)
adapter = resolved["stages"][1]["diffusion_attention_config"]["per_role"] \
    ["hunyuan.diffusion"]["mindie_sla"]["adapter_path"]
assert adapter == os.environ["HUNYUAN_SLA_ADAPTER"], (adapter, os.environ["HUNYUAN_SLA_ADAPTER"])
print("deploy config OK:", adapter)
PY
```

## 3. 安装开发分支

```bash
cd /mnt/share/r50063443/vllm-omni
git fetch origin hunyuan-image3-distil-sla-v0.26
git switch hunyuan-image3-distil-sla-v0.26
git pull --ff-only origin hunyuan-image3-distil-sla-v0.26
git status --short --branch

# 不能只执行 `pip install -e . --no-deps`。v0.27 容器或旧环境可能残留不兼容的
# diffusers/cache-dit 组合。此脚本只校准这两个包，不修改 vLLM、vLLM-Ascend、
# torch、torch_npu 或 CANN。
bash scripts/install_hunyuan_image3_distil_sla_npu.sh

# 必须使用含 SparseLinearAttention NPU op 的已验证 MindIE-SD 版本。
python -m pip install -e /mnt/share/r50063443/HunyuanImage3-SLA/upstream/MindIE-SD --no-deps
```

必须看到：

```text
diffusers 0.38.0
cache-dit 1.3.0
HunyuanImage3 import OK: HunyuanImage3Model
```

如果出现 `cannot import name 'ContextParallelConfig' from 'diffusers'`，说明实际运行
环境没有完成上述依赖校准。`cache-dit==1.3.0` 会导入 Diffusers 的并行配置类型，
而 v0.26 分支固定使用 `diffusers==0.38.0`。重新运行安装脚本后再启动服务。

如果基础 checkpoint 严格校验报告
`sla.proj_l.weight/bias were not initialized from checkpoint`，不要把 SLA adapter 合并进
基础模型。SLA 参数来自独立的 `adapter.safetensors`，不是基础 checkpoint 的组成部分。
更新本分支后，启动日志必须先出现 adapter 校验成功，并随后出现：

```text
Validated HunyuanImage3 SLA adapter ... tensors=64 ...
Strict checkpoint validation excludes 64 parameters supplied by external artifacts.
```

第一条缺失表示 `HUNYUAN_SLA_ADAPTER`、adapter 文件或配置不正确；第一条存在但第二条
缺失表示运行的不是包含外部参数加载契约的最新源码。

确认命令加载的是当前源码：

```bash
python - <<'PY'
import importlib.metadata
import vllm, vllm_omni

assert importlib.metadata.version("vllm").startswith("0.26."), importlib.metadata.version("vllm")
print(vllm_omni.__file__)
print("vllm", vllm.__version__)
PY
```

本分支基于官方 `vllm-omni v0.26.0`，必须搭配
`quay.io/atlas-ci/vllm-ascend:v0.26.0-a3` 或等价的 vLLM/vLLM-Ascend 0.26 A3 环境。
不要在该分支混用 vLLM 0.27。

## 4. 单元测试

Phase A 准确命令：

```bash
conda run -n oneday python -m pytest -q \
  tests/diffusion/attention/test_attention_config.py \
  tests/diffusion/attention/test_mindie_sla.py \
  tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_step_execution.py
```

2026-08-24 本地输出：pytest collection 被 `ModuleNotFoundError: No module named
'vllm'` 阻断。该 macOS 环境不能作为 vLLM/NPU 验收环境。已完成的本地检查：

```bash
conda run -n oneday python -m compileall -q \
  vllm_omni/diffusion/attention/backends/mindie_sla.py \
  vllm_omni/diffusion/models/hunyuan_image3
# exit 0

conda run -n oneday python -m ruff check \
  vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py \
  vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py \
  vllm_omni/diffusion/attention/backends/mindie_sla.py \
  tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_step_execution.py
# All checks passed!

conda run -n oneday python -m ruff format --check \
  vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py \
  vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py \
  vllm_omni/diffusion/attention/backends/mindie_sla.py \
  tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_step_execution.py
# 所有目标文件应通过
```

本地环境实际版本：

```text
torch 2.13.0
safetensors 0.8.0
aenum 3.1.16
ruff 0.14.10
vllm NOT_INSTALLED
vllm-omni NOT_INSTALLED
vllm-ascend NOT_INSTALLED
torch-npu NOT_INSTALLED
```

## 5. Dense Distil 基线

2026-08-24 从本地尝试连接 `HuaWei_npu1` 和 `HuaWei_npu2`，两者均在 8 秒
连接超时内返回 `ssh: connect to host ... port 22: Operation timed out`。因此下面的
Dense 命令尚未执行，不能勾选 Phase A 验收。

服务端命令必须显式记录 deploy config、TP/EP 和模型路径。示例占位：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
mkdir -p logs/hunyuan_image3_distil_dense

vllm serve /mnt/share/r50063443/HunyuanImage-3.0-Instruct-Distil \
  --omni \
  --deploy-config vllm_omni/deploy/hunyuan_image_3_distil_dense.yaml \
  --enforce-eager 2>&1 | tee logs/hunyuan_image3_distil_dense/server.log
```

请求必须包含：

```text
steps=8
guidance_scale=2.5
height=1024
width=1024
seed=42
```

## 6. SLA 服务

示例占位：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HUNYUAN_SLA_ADAPTER=/mnt/share/r50063443/HunyuanImage3-SLA/results/adapters/sla-step-200
mkdir -p logs/hunyuan_image3_distil_sla

vllm serve /mnt/share/r50063443/HunyuanImage-3.0-Instruct-Distil \
  --omni \
  --deploy-config vllm_omni/deploy/hunyuan_image_3_distil_sla.yaml \
  --trust-remote-code \
  --enforce-eager 2>&1 | tee logs/hunyuan_image3_distil_sla/server.log
```

服务启动后发送固定参数请求：

```bash
mkdir -p results/hunyuan_image3_distil_sla
curl -sS http://127.0.0.1:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/mnt/share/r50063443/HunyuanImage-3.0-Instruct-Distil",
    "prompt": "A classroom full of students with laptop computers.",
    "size": "1024x1024",
    "num_inference_steps": 8,
    "guidance_scale": 2.5,
    "seed": 42,
    "response_format": "b64_json"
  }' > results/hunyuan_image3_distil_sla/response.json

python - <<'PY'
import base64, json
from pathlib import Path

response = json.loads(Path("results/hunyuan_image3_distil_sla/response.json").read_text())
Path("results/hunyuan_image3_distil_sla/output.png").write_bytes(base64.b64decode(response["data"][0]["b64_json"]))
PY
```

首版限制必须保持：`max_num_seqs=1`、`sequence_parallel_size=1`、`cfg_parallel_size=1`。
Hunyuan 的 Q/KV 头由现有 TP 路径处理；SLA backend 只在本地头数仍不相等时执行一次
GQA repeat。受因果 mask 约束的时间步/引导前缀使用 Dense SDPA，图像 full-attention
suffix 使用 SLA。任何不满足 suffix 契约的 mask 会直接报错，不会静默忽略。

## 7. 资源与日志

另一个终端持续记录：

```bash
mkdir -p logs/hunyuan_image3_distil_sla
while true; do
  date -Iseconds
  npu-smi info
  sleep 5
done > logs/hunyuan_image3_distil_sla/npu-smi.log 2>&1 &
echo $! > logs/hunyuan_image3_distil_sla/npu-smi.pid
```

至少保存：

```text
环境版本
git commit
deploy config
adapter SHA256
请求 JSON
服务日志
输出图片
每步耗时
峰值 HBM
Dense fallback 次数
```

## 8. 必须填写的结果

```text
Dense 8-step: NOT_TESTED
Dense official alignment: BLOCKED_SERVER_SSH_TIMEOUT_2026-08-24
SLA first step: NOT_TESTED
SLA later step with KV reuse: NOT_TESTED
SLA complete 8-step: NOT_TESTED
TP8/EP8: NOT_TESTED
TP16/EP16: NOT_TESTED
Base/Instruct regression: NOT_TESTED
```
