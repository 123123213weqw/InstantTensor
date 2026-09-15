# qwen35-forward — 自研 forward 的对照基建

给你自己写 Qwen3.5 forward 用的**标准答案生成器**和**比较器**。纯 CPU，不需要 GPU，不需要下载大模型。

## 为什么需要它

自研 forward 最大的风险不是写不出来，是**写错了还以为对**。没有能对的东西，你无法判断。

而且有些错**不会报错、不会崩**，只会让输出变成垃圾。本次实测抓到一个真实的例子：

```
golden (x*(1+w)) absmax = 2.97936
buggy  (x*w)     absmax = 0          ← 所有激活归零
logits: golden absmax=0.541   buggy absmax=0
```

`Qwen3_5RMSNorm` 用的是 **`x * (1 + w)`**，且 `w` **零初始化**。写成 `x * w` 让整个网络归零，
模型照样跑完 22 个 token，只是全是垃圾。

## 目录

```
golden/              金标准生成器 + 校验器（Python）
  gen_golden.py       生成 bundle
  validate_golden.py  验证 bundle 本身可信（可复现 / 完好 / 有区分度）
  README.md           详细文档：格式、两个静默陷阱、区分数表
bundle/              Rust 读取器 + 比较器
  bundlecmp summary|show|compare|selftest
golden_tiny/         已入库的金标准 bundle（gain=1.0，逐张量对照用）
golden_sensitive/    已入库的金标准 bundle（gain=300，token 轨迹对照用）
```

### 已入库的两份 bundle

| | `golden_tiny` | `golden_sensitive` |
|---|---|---|
| `ssm_gain` | 1.0（官方初始化） | 300 |
| 张量数 | 259 | 259 |
| greedy 步数 | 16 | 16 |
| 用途 | **逐张量对照**（任何 gain 下都敏感） | **token 轨迹对照**（gain=1 时轨迹对递归路径几乎不敏感） |
| 大小 | 2.3 MB / 260 文件 | 2.3 MB / 260 文件 |

为什么需要两份，见下文"用数字说清那个取舍"。

**它们是可以再生成的派生物**，随仓库提交只是为了让对比器开箱可用。
改了 `gen_golden.py` 就必须重新生成并跑 `validate_golden.py`。

## 快速开始

**两份金标准 bundle 已随仓库提交，开箱即用，不需要 Python 环境：**

```bash
# 直接构建比较器并使用已入库的 bundle
cargo build --release
./target/release/bundlecmp summary golden_tiny
./target/release/bundlecmp selftest golden_tiny
```

想让你的实现和标准答案对照，把中间结果按同样布局写成 bundle 再跑：

```bash
./target/release/bundlecmp compare golden_tiny <yours>
```

若要重新生成（需要 `transformers` + `torch`，约 1 秒）：

```bash
python golden/gen_golden.py --out golden_tiny --tokens 16
python golden/gen_golden.py --out golden_sensitive --ssm-gain 300 --tokens 16

# 验证金标准本身可信 —— 改了生成器就必须做
python golden/validate_golden.py --bundle golden_tiny
```

## 对照的三层

顺序很重要：**从最便宜的开始，一次只引入一个未知量。**

| 层 | 目录 | 对照什么 | 敏感度 | 用途 |
|---|---|---|---|---|
| **单元** | `units/` | 单独调 delta rule，喂 `q/k/v/g/beta`，比 `out` / `state` | 高 | **先做这个**。不碰 GGUF、不碰加载、不碰 CUDA |
| **逐张量** | `intermediates/` | 每层每个算子的输出 | **高** | 定位"错在哪层哪个算子" |
| **逐 token** | `manifest.greedy` | argmax 序列 + 每步 top-k logits | **低** | 端到端体检 |

## 用数字说清那个取舍

把 SSM 的衰减参数 `A_log` 扰动 1%，实测：

| ssm_gain | 逐张量相对变化 | **greedy token 变化** |
|---|---|---|
| 1（官方初始化） | 5.6e-04 | **0 / 16** |
| 30 | 1.3e-03 | **0 / 16** |
| 100 | 3.2e-03 | **0 / 16** |
| **300** | 1.5e+00 | **4 / 16** |

**真实权重下递归路径只占残差的 0.113%**，所以"改 1% 权重 → token 一个不变"是必然的。
要到 `ssm_gain=300` 才让 token 轨迹敏感，但那时递归路径已经压过残差，不真实了。

→ 所以：**`golden_tiny`（gain=1）验逐张量；`golden_sensitive`（gain=300）验 token 轨迹。**

## 实现者的契约

写出同样布局的 bundle，然后 `bundlecmp compare <golden> <yours>`。

```
<bundle>/manifest.json
<bundle>/weights/<name>.f32          原始 little-endian f32，C 连续
<bundle>/intermediates/<name>.f32
<bundle>/units/<name>.f32
```

命名规则：**模块路径里的 `.` 换成 `__`**

```
model.layers.0.linear_attn.out_proj  →  model__layers__0__linear_attn__out_proj
```

**不用 `.npy` 是刻意的** —— 消费方是 Rust，手写 npy 解析器是没必要的分歧来源。

## 比较器验证过它能抓到 bug

| 测试 | 结果 |
|---|---|
| `selftest`（与自己比） | **PASS** —— 259 张量 0 非零，证明比较器自反 |
| 自己 vs 自己 | **PASS** —— 259 张量全部 bit-identical |
| tiny vs sensitive | **FAIL** —— 指出首个分歧 token 6，`rel=2.990e2`（正好等于 gain−1=299，数学自洽） |
| **`x*w` 而非 `x*(1+w)` 的 bug** | **FAIL** —— 104/123 张量不同，`rel=1.000`，精确定位 `input_layernorm` |

### 一个被这个测试暴露的漏洞（已修）

第一次跑 `x*w` 的 bug 时，**token 轨迹竟显示"完全一致"** —— 因为被比较的是
**候选 manifest 里抄来的 token id**，而候选的 logits 全是 0。

**真实漏洞：比较器信任候选自己记录的 token id。** 一个实现可以写对 id 而 logits 全错。

修法：现在比较器会**从候选自己的 `greedy_stepNN__logits` 反推 argmax**，与它记录的 id 对账：

```
!! CANDIDATE token ids disagree with its own logits at 1 of 16 steps:
     step 0: manifest says 68, its logits say 0
   (recorded ids are not trustworthy; the logits are the ground truth)
```

## 两个会静默杀死网络的陷阱

### 1. 归一化有**两种**约定，写反不报错

| Norm | 公式 | 权重初始化 | 实测值 |
|---|---|---|---|
| `Qwen3_5RMSNorm`（`input_layernorm` / `post_attention_layernorm` / `q_norm` / `k_norm`） | `x * (1 + w)` | **零** | 全 0 ✓ |
| `Qwen3_5RMSNormGated`（`linear_attn.norm`） | `w * x`，再乘 `silu(gate)` | 一 | 全 1 ✓ |

源码注释：`Llama does x.to(float16) * w whilst Qwen3_5 is (x * w).to(float16)`

### 2. `q_proj` 输出是 2 倍，一半是门控

```python
query_states, gate = torch.chunk(
    self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
...
attn_output = attn_output * torch.sigmoid(gate)
```

**`config.json` 写 `output_gate_type: "swish"`，代码用 `sigmoid`** —— 读源码，别读配置。

## 金标准的可信度（已实测）

```
[PASS] 260 个文件跨运行逐字节一致
[PASS] 259 张量完好（无 NaN/Inf，无全零激活）
[PASS] eps=0 逐次 bit-identical
[PASS] delta rule 自检：recurrent vs chunked 吻合到 1e-8 / 1e-7
```

delta rule 自检先验证了**参考实现本身**：

```
delta B1_H2_T6_K16_V16: recurrent vs chunked  out 2.235e-08  state 5.960e-08
delta B1_H2_T1_K16_V16: recurrent vs chunked  out 1.118e-08  state 5.960e-08
delta B2_H3_T5_K8_V8 : recurrent vs chunked  out 2.980e-08  state 1.192e-07
```

## 已知限制

- **token 轨迹对递归路径不敏感**（已量化，见上表）。不是 bug，是真实权重尺度下的必然。
- **无 cache**。greedy 每步重跑整个 forward，刻意把**数学**与 cache 记账隔离：轨迹若分歧，原因在 forward 而不在 cache。cache 语义需单独金标准。
- **随机权重的极小模型**（367,952 参数）。它验证"数学实现对不对"，不是"模型能力强不强"。
- **`linear_num_value_heads=4` / `linear_num_key_heads=2`，故意让 ratio=2**，从而走 `query.repeat_interleave(...)` 那条 GQA 分支（27B 的 ratio=3 走，2B 的 ratio=1 不走 —— 只在 2B 上开发会漏掉）。
- 真实模型验收需要另一份金标准（`--model-dir`），但 27B fp16 需 ~52 GiB 内存。
