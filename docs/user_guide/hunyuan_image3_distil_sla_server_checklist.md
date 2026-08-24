# HunyuanImage3 Distil SLA 服务器验收命令

本文档由实现者持续追加实际可运行命令。不得只保留伪命令。

## 1. 环境记录

```bash
python - <<'PY'
import importlib.metadata

for name in ("vllm", "vllm-omni", "vllm-ascend", "torch", "torch-npu", "safetensors"):
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
test -f /path/to/sla-step/adapter.safetensors
test -f /path/to/sla-step/adapter_config.json
```

## 3. 安装开发分支

```bash
cd /mnt/share/r50063443/vllm-omni
git status --short --branch
python -m pip install -e . --no-deps
```

确认命令加载的是当前源码：

```bash
python - <<'PY'
import vllm_omni
print(vllm_omni.__file__)
PY
```

## 4. 单元测试

实现后把准确 test node ID 补充到这里：

```bash
pytest -q tests/ -k 'hunyuan and distil'
pytest -q tests/ -k 'mindie_sla'
```

## 5. Dense Distil 基线

服务端命令必须显式记录 deploy config、TP/EP 和模型路径。示例占位：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

vllm serve /mnt/share/r50063443/HunyuanImage-3.0-Instruct-Distil \
  --omni \
  --deploy-config vllm_omni/deploy/hunyuan_image_3_distil_dense.yaml \
  --enforce-eager
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

vllm serve /mnt/share/r50063443/HunyuanImage-3.0-Instruct-Distil \
  --omni \
  --deploy-config vllm_omni/deploy/hunyuan_image_3_distil_sla.yaml \
  --enforce-eager
```

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
SLA first step: NOT_TESTED
SLA later step with KV reuse: NOT_TESTED
SLA complete 8-step: NOT_TESTED
TP8/EP8: NOT_TESTED
TP16/EP16: NOT_TESTED
Base/Instruct regression: NOT_TESTED
```
