# Qwen3.5 golden reference — 供自研 forward 对照用

给"自己写 forward"准备的**可对照标准答案**。纯 CPU、无需下载大模型、无需 GPU。

## 一句话结论（来自实测）

**逐张量对比是主仪器；token 轨迹是端到端体检，不是调试工具。**

我把 `A_log`（SSM 的衰减参数）扰动 1%，实测：

| ssm_gain | 逐张量相对变化 | **greedy token 变化** |
|---|---|---|
| 1（官方初始化） | 5.6e-04 | **0 / 16** |
| 30 | 1.3e-03 | **0 / 16** |
| 100 | 3.2e-03 | **0 / 16** |
| **300** | 1.5e+00 | **4 / 16** |

**在真实权重尺度下，递归路径只占残差的 0.113%**，所以"改 1% 权重 → token 一个不变"是必然的。
`ssm_gain=300` 能让 token 轨迹变敏感，但那时递归路径占残差比 >1，已经**不真实**了。

→ **默认 `golden_tiny`（`ssm_gain=1.0`，官方初始化）用于逐张量对照** —— 逐张量在任何 gain 下都敏感（5.6e-04 就能反映 1% 权重扰动）。
→ **另生成一份 `golden_sensitive`（`--ssm-gain 300`）专门用于 token 轨迹验收**，因为它会让 4/16 个 token 发生变化。

两份都用同一套权重导出格式，Rust 侧代码不变。

## 快速开始

```bash
# 1. 生成金标准（约 1 秒）
python golden/gen_golden.py --out golden_tiny --tokens 16

# 2. 生成一个 token 轨迹敏感的版本（用于端到端验收）
python golden/gen_golden.py --out golden_sensitive --ssm-gain 300 --tokens 16

# 3. 验证金标准本身是否可信（必须做）
python golden/validate_golden.py --bundle golden_tiny
```

## 实测验证结果

`validate_golden.py` 会检查三件事。**一个"不可复现"或"无区分度"的金标准比没有更糟** ——
它会产生自信而毫无意义的判定。

```
1) reproducibility
   [PASS] all 260 files byte-identical across runs
2) well-formedness
   [PASS] 259 tensors checked
     note: 21 weight tensors are all zero; 21 are RMSNorm weights
           (expected: Qwen3_5RMSNorm uses x*(1+w) with zero init)
3) discrimination
   [PASS] eps=0 is bit-identical (deterministic)
   [PASS] eps=1e-2 moves per-tensor values by up to 5.127e-06
   [NOTE] eps=1e-2 changed 0/16 greedy tokens: the token trace alone
          cannot validate the recurrent path here
```

以及生成时的自检 —— **delta rule 的两种形态必须吻合**，这先验证了参考实现本身：

```
delta B1_H2_T6_K16_V16: recurrent vs chunked  out 2.235e-08  state 5.960e-08
delta B1_H2_T1_K16_V16: recurrent vs chunked  out 1.118e-08  state 5.960e-08
delta B2_H3_T5_K8_V8 : recurrent vs chunked  out 2.980e-08  state 1.192e-07
```

## ⚠️ 两个会静默杀死网络的坑（已从源码确认）

### 1. 归一化有两种**不同**约定，写反不会报错

```python
class Qwen3_5RMSNorm:
    self.weight = nn.Parameter(torch.zeros(dim))     # ← 零初始化
    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())   # ← (1 + w)
```

| Norm | 公式 | 权重初始化 | 实测值 |
|---|---|---|---|
| `Qwen3_5RMSNorm`（`input_layernorm` / `post_attention_layernorm` / `q_norm` / `k_norm`） | `x * (1 + w)` | **零** | 全 0 ✓ |
| `Qwen3_5RMSNormGated`（`linear_attn.norm`） | `w * x`，再乘 `silu(gate)` | 一 | 全 1 ✓ |

**写成 `x * w` 会让每个 norm 输出归零**，模型照样跑完，只是全是垃圾。
源码里还有一条注释：`Llama does x.to(float16) * w whilst Qwen3_5 is (x * w).to(float16)`。

→ 这也解释了校验器最初误报的"21 个全零权重"：那是**正确的**，是我的检查太天真。现在它只作为 note 报告。

### 2. `q_proj` 输出是 2 倍，一半是门控

```python
query_states, gate = torch.chunk(
    self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
...
attn_output = attn_output * torch.sigmoid(gate)
```

而且 **`config.json` 写的是 `output_gate_type: "swish"`，代码用的却是 `sigmoid`** —— 读源码，别读配置。

## 产物结构

```
<out>/
  manifest.json       全部元数据：配置、张量清单、token 轨迹、自检结果
  weights/            109 个权重张量（fp32 raw LE）
  intermediates/      逐层中间张量（hook 捕获）+ 每步 logits
  units/              delta rule 独立金标准（k/v/q/g/beta + recurrent/chunked 两种输出）
```

### 格式：raw little-endian f32 + JSON manifest

**刻意不用 `.npy`** —— 消费方是 Rust，手写 npy 解析器是没必要的分歧来源。
raw + JSON 无歧义、易 diff。读法：

```rust
let raw = std::fs::read(bundle.join(&entry.file))?;
let vals: Vec<f32> = raw.chunks_exact(4)
    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    .collect();
// entry.shape 给出 reshape 目标
```

### 张量命名

模块路径里的 `.` 被替换为 `__`，所以能用文件名表达：

```
model.layers.0.linear_attn.out_proj  →  model__layers__0__linear_attn__out_proj
```

## 对照的三层用法

| 层 | 对照什么 | 敏感度 | 用途 |
|---|---|---|---|
| **单元**（`units/`） | 单独调 delta rule，喂 `q/k/v/g/beta`，比 `out` 和 `state` | 高 | **先做这个**。不碰 GGUF、不碰模型加载、不碰 CUDA |
| **逐张量**（`intermediates/`） | 每层每个算子的输出 | 高（任何 gain 下都敏感） | 定位"错在哪一层哪个算子" |
| **逐 token**（`greedy`） | `argmax` token 序列 + 每步 top-k logits | **低**（见上表） | 端到端体检，以及最终的真实模型验收 |

`manifest.greedy.steps[i]` 每步给出 `argmax_token`、`topk_tokens`、`topk_logits`、`logit_sum`、`logit_max`，
另外每步完整 logit 向量也落盘在 `intermediates/greedy_stepNN__logits.f32`。

## 极小模型配置（367,952 参数）

```python
hidden_size=64, intermediate_size=128, num_hidden_layers=8,
num_attention_heads=2, num_key_value_heads=1, head_dim=32,
layer_types = 4 层一轮（SSM,SSM,SSM,full_attn）× 2，
linear_conv_kernel_dim=4, linear_num_key_heads=2, linear_num_value_heads=4,
linear_key_head_dim=16, linear_value_head_dim=16,
vocab_size=256, rms_norm_eps=1e-6, partial_rotary_factor=0.25
```

**为什么极小模型够用**：delta rule 的数学与维度无关。用 64 维假权重验证数学，和用 5120 维真权重验证，验的是同一件事 —— 但前者快一万倍，且不需要 GPU 或下载。

`linear_num_value_heads=4` 而 `linear_num_key_heads=2`，**故意保留 ratio=2**，这样会走
`query.repeat_interleave(...)` 那条 GQA 分支（27B 的 ratio 是 3，2B 是 1 所以不走 —— 只在 2B 上开发会漏掉这条路径）。

## 参数

```
--out DIR          输出目录
--tokens N         greedy 步数（默认 16）
--seq N            prompt 长度（默认 6）
--topk N           每步记录 top-k（默认 5）
--seed N           权重初始化种子（默认 0）
--ssm-gain F       缩放 linear_attn.out_proj（默认 1.0）
```

## 已知限制

- **token 轨迹对递归路径不敏感**（已量化，见上表）。这不是 bug，是真实权重尺度下的必然结果。
- **无 cache 的 token 轨迹**。greedy 每步重跑整个 forward，刻意把**数学**与 cache 记账隔离开：
  轨迹若分歧，原因在 forward 而不在 cache。cache 语义需要单独的金标准。
- **极小模型的随机权重**。它验证的是"数学实现对不对"，不是"模型能力强不强"。
  真实模型的验收需要另一份金标准（加 `--model-dir`，但 27B fp16 需 ~52 GiB 内存）。
- **`torch.use_deterministic_algorithms` 未开启**：CPU fp32 在这些算子上实测逐次一致
  （eps=0 逐次 bit-identical 已验证），但换硬件/换 BLAS 后端时需重新确认。
