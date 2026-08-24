# HunyuanImage3 Instruct-Distil 与 SLA 适配任务书

## 1. 项目定位

本仓库基于官方 vLLM-Omni，目标是在 Ascend NPU 上提供：

1. `HunyuanImage-3.0-Instruct-Distil` 的正确 8 步推理。
2. HunyuanImage3 diffusion attention 的 MindIE-SD
   `SparseLinearAttention` 后端。
3. 从 HunyuanImage3-SLA recovery fine-tuning 项目加载每层 `proj_l`
   训练权重。
4. AR 保持 Dense Attention，只有 diffusion 阶段使用 SLA。
5. 支持 vLLM-Omni 原有服务接口和 Ascend 多卡并行。

这不是重新训练或重写 HunyuanImage3，也不是重新实现 SLA kernel。

基准版本：

```text
vLLM-Omni commit: 072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65
development branch: hunyuan-image3-distil-sla
target model: /mnt/share/r50063443/HunyuanImage-3.0-Instruct-Distil
target hardware: 16 x Ascend 910C A3, 64 GiB HBM per device
target dtype: BF16
target resolution: 1024 x 1024
target denoising steps: 8
```

## 2. 已知事实与不能采用的假设

### 2.1 vLLM-Omni 当前能力

当前主线已经包含 HunyuanImage3 pipeline、TP-aware QKV projection、MoE、
prompt KV reuse、diffusion attention backend 选择以及 Ascend NPU 路径。
官方 supported-models 表列出了 Base 和 Instruct，但没有列出
Instruct-Distil。因此不能只依赖模型名称自动认为 Distil 已受支持。

关键源码：

```text
vllm_omni/diffusion/models/hunyuan_image3/
  hunyuan_image3_transformer.py
  pipeline_hunyuan_image3.py
  request_layout.py
  hunyuan_image3_tokenizer.py

vllm_omni/diffusion/attention/
  layer.py
  selector.py
  backends/abstract.py
  backends/registry.py
  backends/sdpa.py
  backends/flash_attn.py

vllm_omni/platforms/npu/platform.py
```

### 2.2 Distil 不只是 `steps=8`

官方 HunyuanImage-3.0 的 Distil 路径至少包含：

- `config.cfg_distilled == true`。
- 生成图像布局中增加 guidance token。
- 构造并加载 `guidance_emb`。
- 每个去噪 step 传入 `1000.0 * guidance_scale` 的 guidance tensor。
- 使用单模型分支，不执行标准 positive/negative true CFG 双分支。
- 使用 8 个去噪步骤。

当前 vLLM-Omni Hunyuan pipeline 默认仍是 50 步、`guidance_scale=5.0`，并且
根据 `guidance_scale > 1` 构造双分支 CFG。因此“只把 50 改成 8”不是完整适配。

Distil 第一版默认参数：

```text
num_inference_steps = 8
guidance_scale = 2.5
cfg_factor = 1
guidance = tensor([2500.0], dtype=bf16)
add_guidance_token = true
```

这些默认值只对 `cfg_distilled` checkpoint 生效，不得改变 Base/Instruct 默认行为。

### 2.3 当前 recovery 训练的边界

训练项目 `loveDjjj/HunyuanImage3-SLA` 已完成 Dense teacher 到 SLA student 的
recovery training，当前只训练 32 层 SLA 的 `proj_l.weight` 和 `proj_l.bias`：

```text
32 layers x 2 tensors = 64 tensors
total trainable parameters = 528384
```

训练适配器使用：

```text
Q heads = 32
KV heads = 8
repeat_kv: 8 -> 32
SLA input layout = [B, H, S, D]
head_dim = 128
topk = 0.125
BLKQ = 64
BLKK = 128
dtype = BF16
```

但 recovery training 使用 `attention_mask=None`，并主要覆盖完整序列的
`first_step=True`。vLLM-Omni 后续去噪会复用 prompt KV，可能形成 `Lq != Lk`。
所以必须单独验证后续 step，不能用 one-step training 成功代替推理验收。

## 3. 目标架构

```text
HTTP / offline request
        |
        v
HunyuanImage3 Instruct-Distil pipeline
        |
        +-- AR / reasoning attention
        |       role = hunyuan.ar
        |       backend = existing Dense backend
        |
        +-- image diffusion attention
                role = hunyuan.diffusion
                backend = MINDIE_SLA
                Q: local TP heads
                K/V: local GQA heads -> repeat_kv -> local Q heads
                layout: B,S,H,D -> B,H,S,D -> SLA -> B,S,H,D
```

不得用全局 backend 开关把 AR attention 一起替换为 SLA。

## 4. 阶段化开发任务

### Phase A：Dense Instruct-Distil 基线

目标是在不引入 SLA 的情况下先证明 Distil 语义正确。

需要修改：

1. 从 model config 读取 `cfg_distilled`，不要依赖路径名包含 `Distil`。
2. checkpoint 为 Distil 时加载 `guidance_emb`，移除当前相关 skip 规则。
3. `build_image_info()` 为 Distil 设置 `add_guidance_token=True`。
4. tokenizer/layout 返回有效 `guidance_scatter_index`。
5. step state 固定 `cfg_factor=1`，不生成 negative CFG branch。
6. 每步构造 `guidance = 1000.0 * guidance_scale`。
7. Distil 默认 8 步、guidance scale 2.5，同时允许请求显式覆盖。
8. Base/Instruct 继续走原有 true CFG 路径。

验收：相同 checkpoint、prompt、seed、分辨率和参数下，与官方 HunyuanImage-3.0
Dense 输出比较。至少保存最终图片、latent 统计量和每一步 prediction 统计量。

### Phase B：区分 AR 与 diffusion attention role

在 `HunYuanAttention` 中有两条执行路径：

```python
self.attn       # gen_text / AR
self.image_attn # gen_image / diffusion
```

为它们设置不同 role：

```python
self.attn = Attention(..., role="hunyuan.ar")

self.image_attn.attn = Attention(
    ...,
    role="hunyuan.diffusion",
)
```

保留现有 prefix、TP、paged KV 和 cache metadata。增加测试，证明 per-role 配置只改变
`hunyuan.diffusion`。

### Phase C：新增 MINDIE_SLA backend

建议文件：

```text
vllm_omni/diffusion/attention/backends/mindie_sla.py
```

并在 `DiffusionAttentionBackendEnum` 增加：

```python
MINDIE_SLA = (
    "vllm_omni.diffusion.attention.backends.mindie_sla."
    "MindIESLABackend"
)
```

后端最小接口骨架：

```python
class MindIESLABackend(AttentionBackend):
    supported_platforms = ("npu",)

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return False

    @staticmethod
    def get_name() -> str:
        return "MINDIE_SLA"

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_impl_cls():
        return MindIESLAImpl


class MindIESLAImpl(AttentionImpl):
    def __init__(
        self,
        num_heads,
        head_size,
        softmax_scale,
        causal=False,
        num_kv_heads=None,
        prefix="",
        backend_kwargs=None,
        **kwargs,
    ):
        from mindiesd.layers import SparseLinearAttention

        opts = backend_kwargs or {}
        self.prefix = prefix
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.sla = SparseLinearAttention(
            head_dim=head_size,
            topk=float(opts.get("topk", 0.125)),
            BLKQ=int(opts.get("blkq", 64)),
            BLKK=int(opts.get("blkk", 128)),
            use_bf16=bool(opts.get("use_bf16", True)),
        )

    def forward_npu(self, query, key, value, attn_metadata=None):
        # vLLM-Omni input: [B, S, H, D]
        if query.shape[2] % key.shape[2] != 0:
            raise ValueError("Q heads must be divisible by KV heads")
        repeat = query.shape[2] // key.shape[2]
        if repeat != 1:
            key = key.repeat_interleave(repeat, dim=2)
            value = value.repeat_interleave(repeat, dim=2)
        q, k, v = (x.transpose(1, 2).contiguous() for x in (query, key, value))
        out = self.sla(q, k, v)
        return out.transpose(1, 2).contiguous()
```

这只是接口参考，不可直接视为完成实现。实现者必须处理：

- vLLM Hunyuan 当前可能已经在 `ImageKVCacheManager` 中 repeat KV，避免重复扩展。
- TP8/TP16 下的 local head 数量。
- first step 与 prompt KV reuse 的不同 Q/K 长度。
- attention mask 与 padding 的语义。
- SP/joint attention metadata。
- SLA module 参数注册和权重加载时机。

### Phase D：attention mask 策略

MindIE `SparseLinearAttention` 不接受任意 attention mask。禁止无日志地丢弃 mask。

第一版必须在以下方案中明确选择并测试一种：

1. **推荐的最小验证路径**：只对无需 mask 的 image-query 区域调用 SLA，文本/prefix
   query 保持 Dense，然后拼接输出。
2. 对 first step 关闭 prompt KV optimization，使用与 recovery training 一致的完整无 mask
   序列；该方案只用于验证，性能可能较差。
3. 对不满足 SLA 合约的 shape/mask 显式 Dense fallback，并记录 fallback 计数。

不得直接删除混合 text-causal/image-full mask 后声称与 Dense 等价。

### Phase E：SLA adapter 权重格式

vLLM 不直接读取 DeepSpeed ZeRO-3 的 16 份训练 checkpoint。输入应先由训练仓库导出为
紧凑 safetensors：

```text
adapter.safetensors
adapter_config.json
```

建议 key：

```text
layers.0.sla.proj_l.weight
layers.0.sla.proj_l.bias
...
layers.31.sla.proj_l.weight
layers.31.sla.proj_l.bias
```

配置示例：

```json
{
  "format_version": 1,
  "base_model": "HunyuanImage-3.0-Instruct-Distil",
  "num_layers": 32,
  "head_dim": 128,
  "topk": 0.125,
  "blkq": 64,
  "blkk": 128,
  "dtype": "bfloat16",
  "trained_parameters": ["proj_l.weight", "proj_l.bias"]
}
```

加载规范：

- adapter 路径必须通过 deploy config/CLI 显式提供，不写死服务器路径。
- 校验 32 层、64 tensors 全部存在且 shape 正确。
- 缺失、多余、NaN/Inf、错误 dtype/shape 立即失败。
- `proj_l` 只有约 528K 参数，TP rank 上复制，不进行 tensor shard。
- rank 0 打印 adapter SHA256、层数和加载参数数目，其余 rank 不重复刷屏。

### Phase F：多卡策略

优先配置：

```text
NPU 0-7:  AR stage, TP=8
NPU 8-15: diffusion stage, TP=8, EP=8
SP=1
CFG parallel=1 (Distil 是单分支)
batch size=1
enforce eager=true
```

Hunyuan Q=32、KV=8。当前 TP 实现允许 TP 大于 KV head 数时复制 KV，因此 TP16
理论可行，但不是第一阶段验收配置。先通过 TP8，再测试 diffusion TP16/EP16。

SLA 首版不要同时引入 SP。只有在 SLA 对 joint query/key/value、prompt KV 和跨 rank
序列布局都有测试后，才能声明 SP 支持。

## 5. 配置目标

增加单独 deploy config，例如：

```text
vllm_omni/deploy/hunyuan_image_3_distil_sla.yaml
```

attention 配置表达以下语义：

```json
{
  "default": {"backend": "TORCH_SDPA"},
  "per_role": {
    "hunyuan.ar": {"backend": "TORCH_SDPA"},
    "hunyuan.diffusion": {
      "backend": "MINDIE_SLA",
      "topk": 0.125,
      "blkq": 64,
      "blkk": 128,
      "use_bf16": true,
      "adapter_path": "/path/to/adapter.safetensors"
    }
  }
}
```

字段名必须服从当前 `AttentionConfig`/`AttentionSpec` 结构；不要另建一套重复的环境变量
解析系统。

## 6. 测试矩阵

### CPU/无 NPU 测试

- Distil config 识别。
- Distil layout 含 guidance token。
- Distil `cfg_factor == 1`。
- Base/Instruct 仍按原逻辑生成 true CFG branches。
- AR 与 diffusion role backend 选择。
- adapter 64 tensors 的严格校验。
- GQA repeat 和 layout 转换 shape。
- 非 NPU 选择 `MINDIE_SLA` 时给出清晰错误。

### 单 NPU 测试

- MindIE-SD import 和 kernel smoke test。
- SLA `B=1, Hq=32, Hkv=8, S=4123, D=128`。
- SLA 不同 Q/K 长度测试。
- first-step diffusion forward。
- later-step prompt KV reuse forward。
- adapter 加载前后输出确实发生变化。
- 输出、sparsity、latents 全部 finite。

### 8/16 NPU 测试

- Dense Distil TP8 完整 8 步。
- SLA Distil TP8 完整 8 步。
- 连续生成至少 10 张，验证 cache 不串请求。
- 相同 seed 重复请求具有确定性或记录可接受误差。
- TP16 仅作为后续项目，不阻塞 TP8 第一版。

## 7. 完成标准

以下全部满足才可以报告完成：

- [ ] Dense Distil 8 步完整生成成功。
- [ ] Dense Distil 与官方实现的对齐报告已保存。
- [ ] AR attention 未被 SLA 替换。
- [ ] 32 层 diffusion attention 全部加载 SLA adapter。
- [ ] first-step 和 later-step SLA shape 均测试成功。
- [ ] 完整 SLA 8 步生成成功，输出图片非空。
- [ ] 所有 SLA 输出和 latent finite。
- [ ] TP8/EP8 服务启动成功。
- [ ] HTTP 或 offline 请求成功返回图片。
- [ ] 日志包含 backend、adapter SHA256、并行配置、耗时与峰值 HBM。
- [ ] Base/Instruct 回归测试通过。
- [ ] 文档给出可直接执行的安装、启动、请求和排错命令。

不得把“模型加载成功”“单个 kernel forward 成功”或“训练 one-step 成功”写成完整推理成功。

## 8. 非目标

- 不修改 MindIE-SD SLA kernel。
- 不重新实现 SparseLinearAttention。
- 不训练或微调 HunyuanImage3。
- 不在本仓库存放基础模型、训练 checkpoint 或数据集。
- 第一版不做量化、SP、paged SLA KV、continuous batching 性能优化。
- 不为了 SLA 破坏 vLLM-Omni 其他 diffusion 模型的 backend 行为。

## 9. 参考资料

- vLLM-Omni：<https://github.com/vllm-project/vllm-omni>
- 支持模型：<https://github.com/vllm-project/vllm-omni/blob/main/docs/models/supported_models.md>
- Diffusion attention backend：<https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/attention_backends.md>
- HunyuanImage3 recipe：<https://github.com/vllm-project/vllm-omni/blob/main/recipes/Tencent/HunyuanImage-3.0-Instruct.md>
- Custom backend RFC：<https://github.com/vllm-project/vllm-omni/issues/3715>
- HunyuanImage-3.0：<https://github.com/Tencent-Hunyuan/HunyuanImage-3.0>
- Instruct-Distil 模型：<https://modelscope.ai/models/Tencent-Hunyuan/HunyuanImage-3.0-Instruct-Distil>
- MindIE-SD：<https://gitcode.com/Ascend/MindIE-SD>
- recovery training：<https://github.com/loveDjjj/HunyuanImage3-SLA>

## 10. 交付报告格式

最终报告必须列出：

1. 修改文件和每个文件的职责。
2. Distil 与普通 Instruct 的行为差异。
3. Hunyuan diffusion attention 的准确替换位置。
4. SLA adapter key 到 vLLM module 的映射。
5. Dense 对齐结果。
6. first-step/later-step/8-step 的测试结果。
7. TP、EP、SP 和 CFG parallel 的实际配置。
8. 峰值 HBM、单步耗时和完整生成耗时。
9. 已知限制和 Dense fallback 是否发生。
10. 最终服务启动命令与请求命令。
