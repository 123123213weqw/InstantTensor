# rust-qwen-engine — 加载层（第一章）

用 Rust 重写 safetensors 权重加载层。**目标不是让硬盘更快，而是消灭 Python 启动开销。**

## 为什么要做

实测拆解（7.2b 冷拉起，Python 基线）：

| 段 | 耗时 | 占比 | 成因 |
|---|---|---|---|
| `import torch` | 0.665s | 13.5% | Python 解释器 + C 扩展导入 |
| `load_repo` | 1.871s | 37.9% | Python 模块导入机 |
| 建 447 个量化模块 | 0.666s | 13.5% | Python 对象构造 |
| **IT 读入** | 1.682s | 34.1% | 磁盘 I/O，**已达 98% 盘顶** |
| 合计 | **4.933s** | | |

固定成本 `F ≈ 2.8s` 全部来自 Python 运行时；I/O 那 34.1% 已经吃到盘顶（4.25 / 4.33 GB/s）。
**Rust 能砍掉 3.202s（64.9%），对 I/O 零贡献。**

```
现在:  F 2.80s + W 2.74s = 5.54s  →  1.58x
Rust:  F 0.40s + W 2.74s = 3.14s  →  ~2.8x
```

验收口径：**7.2b 冷拉起 5.54s → 3.2s**（最终数字在 4080 上测；开发在 V100 上做）。

## 架构

```
crates/stloader/         库
  header.rs   safetensors 头解析：8B LE 长度 + JSON；多分片 + index.json
  plan.rs     张量字节区间 → 块对齐读请求（合并相邻张量）
  reader.rs   io_uring + O_DIRECT，深度受限、buffer 池复用
  cache.rs    posix_fadvise 驱逐 + mincore 验证驻留（冷缓存可信度）
  aligned.rs  显式对齐的缓冲区分配
crates/stbench/          CLI：list / read / verify
tools/                   Python 参考工具（GGUF 检查、对齐分析）
```

### 关键设计：为什么需要合并读请求

safetensors 里张量是紧排的，但**头部长度只保证 8 字节对齐**，所以张量起始偏移几乎不可能是 4096 对齐的。`O_DIRECT` 要求 offset 和 length 都按块对齐，于是每个张量都要向外取整。

实测（`Qwen3.8-27B`，18 分片，1199 张量，51.747 GiB）：

```
internal gaps = 0B    head = 0B    tail = 0B      ← 完全紧排，无空洞
unaligned offset: 1199/1199 (100.0%)              ← header 长度不是 4096 倍数
unaligned size:    533/1199 (44.5%)
requests: naive=1199  →  coalesced=18             ← 每分片 1 次
read amplification = 1.0000x                      ← 零放大
```

**合并后零浪费。** 这是 `plan.rs` 存在的理由；不合并会退化成 1199 次请求。

## 已验证

| 项 | 方法 | 结果 |
|---|---|---|
| 头解析正确性 | Rust 输出 vs Python 独立实现 | **完全一致**（1199 张量、51.747 GiB、18 请求、1.0000x、BF16×1199、index.json 1199 条 0 缺失） |
| 头解析速度 | 18 分片全部头 | **2.6 ms** |
| O_DIRECT 数据正确性 | 40 个最小张量，direct vs buffered 逐字节比对 | **0 mismatch** |
| index.json 交叉校验 | `weight_map` vs 分片实际张量 | 1199 条，**0 缺失** |

选取最小张量做比对是有意的：它们是**最不可能块对齐**的那些，正好压测取整逻辑。

## 用法

```bash
cargo build --release

# 张量清单 + 对齐统计 + 读请求计划
stbench list <model_dir>

# 冷缓存吞吐（--cold 会先 posix_fadvise 驱逐并报告 mincore 驻留）
stbench read <model_dir> --depth 32 --chunk-mb 1 --cold
stbench read <model_dir> --depth 32 --chunk-mb 1 --cold --buffered

# 对照组：每张量一次请求（暴露对齐惩罚）
stbench read <model_dir> --cold --per-tensor

# 正确性：O_DIRECT vs 缓冲读逐字节比对
stbench verify <model_dir> --tensors 40
```

## 环境要求

- Linux 内核 ≥ 5.6（io_uring），推荐 ≥ 5.15；需 `kernel.io_uring_disabled = 0`
- 支持 `O_DIRECT` 的文件系统

## 实测结果

开发/验证机：Linux / kernel 6.8 / rustc 1.95 / 单 NVMe。
测试模型：`Qwen3.8-27B`（18 分片，1199 张量，51.747 GiB，BF16）。
每次运行前 `posix_fadvise(DONTNEED)` + `sync`，并用 **mincore 确认 `residency = 0.000`** 才开始计时。

### 吞吐

| 配置 | 耗时 | GB/s | 请求数 |
|---|---|---|---|
| `dd` 单流 O_DIRECT（3.2 GiB 参照） | 3.148s | **1.10** | — |
| Rust depth=8, chunk=1MiB | 31.139s | 1.78 | 53004 |
| **Rust depth=32, chunk=1MiB** | **25.569s** | **2.17** | 53004 |
| Rust depth=64, chunk=1MiB | 25.522s | 2.18 | 53004 |
| Rust depth=32, chunk=8MiB | 25.501s | 2.18 | 6634 |
| Rust depth=32, chunk=1MiB, **缓冲读** | 36.622s | **1.52** | 53004 |
| Rust **每张量一请求**（对照组） | 25.786s | 2.16 | 54081（放大 1.0001x） |

结论：

1. **O_DIRECT 比缓冲读快 43%**（2.17 vs 1.52 GB/s）→ 冷加载必须走直连，与预期一致。
2. **深度在 32 饱和**（32→64 仅 +0.5%）、**chunk 大小完全平**（1MiB vs 8MiB +0.5%）→ 参数地形是平的，与冷缓存方法论的结论一致。
3. **2.17 GB/s 已超过该单盘 2.0 GB/s 的既有天花板**；`dd` 单流只有 1.10 GB/s，说明并发带来约 2x。
4. 峰值 RSS 0.03~0.07 GiB（depth × chunk 的 buffer 池），无内存膨胀。

### 启动开销（本章的核心收益）

| | 实测 |
|---|---|
| Rust 进程启动 + 18 分片全头解析 | **wall 0.00–0.01 s**，maxRSS **2–2.5 MB** |
| 头部解析本身 | 2.5–9.6 ms |
| （对照）Python `import torch` + `load_repo` | **2.80 s** |

**Rust 把固定开销 F 从 2.80s 压到 ~0.01s 量级。** 投影到 7.2b 冷拉起：

```
现在:  F 2.80s + W 2.74s = 5.54s  →  1.58x
Rust:  F 0.01s + W 2.74s = 2.75s  →  ~3.2x
```

### 跨模型验证

| 模型 | 分片 | 张量 | 大小 | 请求 | 放大 | dtype |
|---|---|---|---|---|---|---|
| `Qwen3.8-27B` | 18 | 1199 | 51.747 GiB | 18 | 1.0000x | BF16×1199 |
| `DeepSeek-R1-Distill-Qwen-7B` | 2 | 339 | 14.185 GiB | 2 | 1.0000x | BF16×339 |
| `rwkv7-g1g-1.5b-hf` | 6 | 795 | 2.845 GiB | 6 | 1.0000x | F16×795 |

三者 `index.json` 条目数与分片实际张量数**完全一致，0 缺失**。
DeepSeek-7B 冷读：**2.21 GB/s**。

### 一个诚实的修正

原本预期"合并相邻张量"是性能关键。实测显示：**这三个模型的张量是完美紧排的**（`internal gaps = 0B`），
且读请求本来就要按 chunk 切分，所以**合并与不合并的吞吐几乎一样**（2.16 vs 2.17 GB/s）。
合并的价值在于把规划层的请求数从 1199 降到 18，而不是吞吐。

若文件里有**大空洞或大量碎片**，合并才会变成吞吐杠杆 —— 当前测试集里没有这种文件。

## 尚未做的（下一章）

加载层只做到「把字节以盘顶速度读出来并校验」，还没有：

1. **交付给消费者** —— 权重如何落到显存、按什么布局交给自研 Qwen 计算代码（pinned buffer + `cudaMemcpyAsync` 流水，或直接落显存）
2. **量化构造** —— 若走 r1mm8/w8row 那条路，构造开销（0.666s）要重新计入
3. **4080 上的端到端验收** —— `5.54s → 3.2s` 的目标口径在 4080（4.33 GB/s 盘）上，本机是 V100（2.2 GB/s），比例不同

## 鲁棒性

### 自审发现并修掉的 13 个问题

第一版只验证了"合法文件能读对"，没有验证"非法文件会怎样"。自审后发现以下问题：

| # | 问题 | 后果 | 修法 |
|---|---|---|---|
| 1 | `plan.rs` 的 `a1 - a0` 可能**下溢** | release 关闭溢出检查 → 绕回成 u64 巨值 → **巨额分配/OOM** | `a1 <= a0` 时跳过，永不发出 |
| 2 | `last.offset + last.len + max_gap` 溢出 | 合并判断绕回，可能错误合并或拒绝 | 改用 `saturating_add` |
| 3 | `data_start + self.end` 未检查溢出 | 与 1 同类 | 改为解析期强制不变量 |
| 4 | 未校验 `end <= buffer_size` | 畸形头会导致读到文件外的偏移 | 解析期拒绝 |
| 5 | 未校验 `offsets` 与 `shape × dtype` 一致 | 合法 JSON 但描述错误字节范围的头部会被静默接受 | 解析期交叉校验 |
| 6 | `ceil_to` 可溢出 | 极大输入绕回 | 改 `saturating_mul` |
| 7 | `ReadConfig.keep_open` 是**死字段** | 谎报能力 | 删除 |
| 8 | `read_files` 对不匹配的 plan 会**数组越界 panic** | 崩而非报错 | 改为 `InvalidInput` 错误 |
| 9 | `submit_and_wait` 把 `EINTR` 当致命错误 | 被信号打断就假报失败 | 重试 |
| 10 | **零元素张量会发出一次虚假读请求**（读到 header 区填充） | 无谓 I/O | 跳过 `nbytes()==0` |
| 11 | `shape` 解析把 `u64::MAX` 当成"非整数" | 错误信息误导 | 区分两种情况 |
| 12 | 完全没有测试 | — | 见下 |
| 13 | `verify` 只覆盖 ≤64 KiB 张量 | 占绝大多数字节的大张量从未被校验 | 增加大张量覆盖 |

**第 10 条是我自己的测试抓出来的**，不是审出来的 —— 这正好说明测试的价值。

### 测试

```bash
cargo test --release        # 3 个测试，含 28 个合成用例
stbench selftest            # 28 个合成鲁棒性用例，逐条打印
```

`stbench selftest` 会**现场生成**畸形 safetensors 并断言行为，不需要任何真实模型：

```
[PASS] hdr_zero                          拒绝：header length is zero
[PASS] hdr_overrun                       拒绝：header length 4096 overruns file of 10 bytes
[PASS] hdr_absurd                        拒绝：header length 1099511627776 exceeds 268435456
[PASS] hdr_short                         拒绝：file shorter than 8-byte header length
[PASS] json_bad                          拒绝：bad JSON header
[PASS] json_array                        拒绝：header is not a JSON object
[PASS] entry_not_obj                     拒绝：entry a is not an object
[PASS] meta_not_obj                      拒绝：__metadata__ is not an object
[PASS] dtype_unknown                     拒绝：unknown dtype F7
[PASS] missing_shape                     拒绝：missing shape
[PASS] offsets_not_pair                  拒绝：data_offsets is not a pair
[PASS] end_before_start                  拒绝：end 10 < start 100
[PASS] end_beyond_buffer                 拒绝：ends at 1024 beyond buffer of 8 bytes
[PASS] shape_dtype_mismatch              拒绝：declares 20 bytes but shape [10] x F32 needs 40
[PASS] shape_noninteger                  拒绝：shape has a non-integer dimension "x"
[PASS] offset_u64max                     拒绝：ends at 18446744073709551615 beyond buffer of 16
[PASS] shape_overflow                    拒绝：shape product overflows u64
[PASS] off_by_one                        拒绝：ends at 17 beyond buffer of 16 bytes
[PASS] valid: two contiguous tensors     tensors=2 tensor_bytes=8192 ranges=1
[PASS] valid: zero-element tensor        tensors=1 tensor_bytes=0 ranges=0
[PASS] valid: scalar tensor shape []     tensor_bytes=4
[PASS] gap: merge respects max_gap       ranges tight=2 loose=1
[PASS] arithmetic: ceil_to saturates, floor_to safe, no panic
[PASS] arithmetic: shape_numel overflow -> None
[PASS] api: out-of-range shard index rejected
[PASS] api: zero-length range rejected
[PASS] api: empty plan is a no-op
[PASS] plan: no zero-length, in-range, non-overlapping

28 cases, 28 passed, 0 failed
```

另有一个**已知字节模式的往返测试**：构造一个 300 字节填充 + 8192 字节确定性 pattern 的文件，
确保张量偏移**故意不对齐**，然后走 `O_DIRECT` 读回并逐字节比对 —— 端到端验证对齐切片逻辑。

### 真实文件损坏测试

在真实的 512 MiB 分片（`rwkv7-g1g-1.5b-hf`）上做变体：

| 变体 | 结果 |
|---|---|
| payload 截断一半 | `entry lm_head.weight ends at 268435456 beyond buffer of 268435344 bytes` |
| **payload 只少最后 1 字节** | `entry model.embeddings.weight ends at 536870912 beyond buffer of 536870911 bytes` |
| 只保留 header | `ends at 268435456 beyond buffer of 0 bytes` |
| 完好（对照） | 通过，`tensors=2 planned: 1 requests amplification 1.0000x` |
| 截断文件上执行 `read` | 同样在解析期报错，不进读路径 |
| 全随机字节 | `header length 9636161184222581920 exceeds 268435456` |
| 空文件 | `file shorter than 8-byte header length` |
| 目录无 .safetensors | `no .safetensors files` |

**全部干净报错，无 panic、无 OOM、无巨量分配。** 1 字节的截断也能精确检出。

### 一个必须记录的设计事实

plan 会向上取整到**超出文件尾**，靠 EOF 短读收尾（基准里那 18 个 `short_reads` 就是 18 个分片）。
这是 `O_DIRECT` 下的唯一正确做法：**把长度夹到 `file_size` 会破坏块对齐，内核直接返回 `EINVAL`。**

### 内建 fuzzer（`stbench fuzz`）

手写用例只能覆盖**作者想到的情况**。本 crate 自带一个变异 fuzzer，不依赖任何外部工具（不需要 nightly、libFuzzer 或 sanitizer）：

```bash
cargo build --profile fuzz
stbench fuzz --iters 1000000 --seed 1 --read
```

它做两件事，缺一不可：

1. **捕获 panic** —— 整个解析/规划/读取流程跑在 `catch_unwind` 里，越界或 unwrap 会变成一条记录而不是杀掉进程。
2. **真正的安全判据** —— 只防崩溃是不够的。**如果解析成功**，则规划出的范围必须满足：
   - 每个范围两端都块对齐、非空；
   - 每个范围都在 `[0, ceil(file_size, block)]` 内 → **不可能读到文件外**；
   - 范围有序且不重叠；
   - **每个张量的绝对字节区间被完全包含在某个范围内** → 不可能漏读；
   - **每个范围至少覆盖某个张量的一些字节** → 不可能空转读。

最后一条是"效率"判据，前四条是"安全"判据。只有安全判据会漏掉一类 bug（见下）。

#### 关键：验证判据本身有没有牙齿

"fuzz 结果是 clean" 有两种解释：代码没问题，或者**判据是瞎的**。所以故意注入 bug 看它是否报警：

| 注入的 bug | fuzzer 是否抓到 | 诊断信息 |
|---|---|---|
| 去掉 `.min(file_ceil)` 夹取 | **没抓到（正确）** | 解析期已保证 `abs_e <= file_size`，故 `ceil_to(abs_e) <= file_ceil` 恒成立 —— 这个夹取是**冗余的**，不是 bug |
| `a0` 用 `ceil_to` 而非 `floor_to` | **抓到** | `tensor a [127,4223) not contained in range [4096,8192)` |
| 同上（另一种表现） | **抓到** | `tensor w [96,104) has no covering range` |
| 去掉零元素张量跳过 | **改进前漏掉 → 改进后第 3 次迭代抓到** | `range [0, 4096) covers no tensor bytes (wasted read)` |

#### 一个真实的教训：fuzzer 和手写用例覆盖不同的东西

注入"去掉零元素跳过"这个 bug 后：

| | 结果 |
|---|---|
| **手写 `selftest`** | **抓到** — `[FAIL] valid: zero-element tensor  tensors=1 tensor_bytes=0 ranges=1` |
| **fuzzer（改进前）** | **漏掉** — 随机生成的头部**几乎不会**产生"合法文件 + 退化张量（零元素 / scalar / 未对齐偏移）"这种组合 |

修法不是调判据，而是**给 fuzzer 加一条策略：直接回放种子语料**（不做变异）。改进后同一 bug 在第 3 次迭代被抓到。

所以：**28 个手写用例不是 fuzzer 的替代品，两者是互补的。** 手写用例编码"我知道的语义边界"，fuzzer 负责"我没想到的结构畸形"。

#### 大规模运行结果

| 版本 | 规模 | 解析成功 | 正确拒绝 | 执行读 | 结果 |
|---|---|---|---|---|---|
| 改进前 | 3 × 400,000 = 120 万次（含读） | 157k | 104 万 | 157k | clean |
| **改进后** | **8 × 250,000 = 200 万次（含读）** | **625,729** | **1,374,271** | **625,729** | **clean** |

改进版的 8 个 seed（11/22/33/44/55/66/77/88）逐个独立运行，全部
`RESULT: clean (no panics, no invariant violations)`，吞吐约 3,200 次/秒。
另有 `dtype table consistent: true`（dtype 名称与枚举双向 round-trip 一致）。

`panic = "abort"` 与 `catch_unwind` 冲突，所以 fuzzer 必须用 `fuzz` profile（`inherits = "release"` + `panic = "unwind"`）。这不是可选项 —— 用错 profile 会让"捕获到的 panic"变成进程终止，fuzzer 形同虚设。

### 仍然没做的（诚实清单）

- **自带 fuzzer 无覆盖引导**。它是随机变异，不像 `cargo-fuzz` / libFuzzer 那样朝新代码路径定向进化，
  所以**发现深藏 bug 的能力弱于 libFuzzer**；运行时间是分钟级，不是长时间持续 fuzzing。
- 有 28 个手写用例 + 8 个真实文件变体 + 200 万次随机变异，但**都不是全量字节比对**
- **没有覆盖全部字节**：`verify` 只比对了 48 个最小张量 + 4 个最大张量的前 4 MiB，不是 51.7 GiB 全量
- **单线程单 ring**：没做多 ring / 多线程，也没做 NUMA 或 CPU 亲和
- **`--gap` 语义未在真实带空洞文件上验证**（三个测试模型的张量都是零间隙的）
- **未校验分片集合与 `index.json` 的文件名一致**，只校验了张量名集合
- **无 mmap 路径**可比对（`--buffered` 是 `read(2)`，不是 mmap）
- **无并发加载测试**（两个进程同时加载同一模型）

## 相关文档

- `../qwen35-forward/` — Qwen3.5 forward 的金标准生成器与比较器（本仓库另一目录）
- `docs/loader-internals.md` — Siphon 自身（Python/C++）的加载器内部原理
- `docs/benchmark.md` — Siphon 的原始 H200 基准

### 冷缓存方法论

本文所有吞吐数字都在**冷缓存**下测得：先 `posix_fadvise(DONTNEED)` + `sync`，
再用 `mincore` 采样确认驻留率为 `0.000` 才开测（`stbench read --cold` 会自动做这两步并打印）。
不做这一步的数字没有意义 —— 页缓存会把它变成内存带宽测试。
