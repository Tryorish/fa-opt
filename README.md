# BW1000 / gfx936 FlashAttention

将 `flash_attn_v2_r64c32d64_kernel` 迁移为独立 HIP 实现。依赖 **DTK 编译器和 DTK 的 HIP PyTorch**，不依赖 TileLang。支持 FP16、连续 `[B,H,N,D]` Q 和 `[B,H,M,D]` K/V，D=64 或 128、非 causal、无 dropout、前向计算。数据指针需 16 字节对齐（普通 PyTorch 分配满足；有偏移的 view 可能不满足）。N/M 支持尾块；默认 scale 为 `1/sqrt(D)`。D128 已完成首轮四组分块对照；B1/H32/N4096 当前最快 Q128/C32 为 2031.34 μs，仍慢于 Flash SDPA。

D64 默认采用已在 BW1000 验证的 **Q128/C64、矩阵 V、完整 KV 专用路径、原生 exp2、QK 前预取**，关闭条件重缩放、异步 K/V、延后分母归约和延后 scale。不会根据形状自动切换。复现最快实测结果请使用下列优化配置；各轮对照记录见文末。

## 已测领先 Flash SDPA 的配置

BW1000/gfx936、PyTorch 2.7.1、HIP 6.3.26045，B1/H16、D64、N=M、FP16、非 causal、scale=0.125，Graph 中位数计时。第十九轮开启异步 K/V、延后分母归约和延后 scale 后，按形状选取 Q 分块的成绩：

| N | Q/C 分块 | 本 kernel μs | Flash SDPA μs | 加速比 |
| ---: | --- | ---: | ---: | ---: |
| 1024 | 128/64 | 57.31 | 68.91 | 1.202x |
| 2048 | 256/64 | 188.99 | 193.75 | 1.025x |
| 4096 | 128/64 | 648.08 | 659.34 | 1.017x |

四组对照各 19 项测试全部通过，包括 scale、尾块、共享内存标签、输出 guard 和当前 stream；均无 scratch/spill。这里是本轮指定形状和配置的结果，不表示一个固定 Q 分块或默认配置在所有形状都领先。N2048/4096 的优势较小，性能受环境与输入规模影响。

复现上述配置（每次保留独立结果目录）：

```bash
bash bench_best.sh
```

脚本固定 B1/H16、上述 Q/C 选择和原来的计时参数，强制 Flash 后端；每个形状仍由 `bench.py` 先校验输出再计时。JSON 保存到 `build/best/run-*/`，不会静默回退到 Math SDPA。

在自己的 Python 程序中使用对应优化：

```bash
export FA_BW_CXXFLAGS="-DFA_BW_Q_TILE=128 -DFA_BW_ASYNC_V=1 -DFA_BW_ASYNC_K=1 -DFA_BW_DEFER_DENOM=1 -DFA_BW_DELAY_SCALE=1"
# N=2048 的上述成绩使用 Q_TILE=256；改变编译选项后启动新的 Python 进程。
```

然后调用下文的 `attention(q, k, v)`。这些开关只优化满足条件的完整 KV 调用；零/负 scale 和 KV 尾块使用原通用路径。

## B1 / H32 / N4096 / D128

首轮 BW1000 实测，B=1、H=32、N=M=4096、D=128、FP16、非 causal、scale=`1/sqrt(128)`，Graph 计时，启用 async K/V、延后分母归约与延后 scale：

| Q | K | 本 kernel μs | Flash SDPA μs | 加速比 |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 32 | 3755.25 | 1521.67 | 0.405x |
| 64 | 64 | 3745.27 | 1521.07 | 0.406x |
| 128 | 32 | 2031.34 | 1521.38 | 0.749x |
| 128 | 64 | 2471.82 | 1521.49 | 0.616x |

目前 Q128/K32 最好，耗时比配对 SDPA 多 33.5%；自身耗时还需降低约 25.1% 才能追平。Q64→Q128 在 K32 下快约 1.85 倍；Q128 下 K64 比 K32 慢约 21.7%。这支持继续检查 Q 方向的复用收益，但单靠时间不能确认是带宽、LDS 容量还是寄存器调度限制。

下一轮固定 K32，对比 Q128/Q192/Q256（已有 kernel 参数，无新增计算路径）：

```bash
bash compare_d128_qtiles.sh
```

三组均占 32 KiB LDS，该规模 block 数分别为 1024/704/512；Q 与输出寄存器负担随 Q 增大。脚本每组先强制编译一份 D128 诊断产物，输出 VGPR/scratch/spill 和完整块汇编摘要，再运行正确性测试和强制 Flash SDPA 计时。结果、编译信息、测试输出分别保留在 `build/d128_qtiles/run-*/`。请返回最终计时表及 `q*_resources.log` 中的编译器资源信息；不需要下载二进制产物。Q192/Q256 的 D128 性能尚未实测，可能因 spill 退步。

D128 与 D64 共用计算逻辑和 QK 列置换；Python 按输入末维选择独立编译产物，因此同一进程可调用两种维度。默认 scale 为 `1/sqrt(128)`，约 `0.08838835`。不需要手工设置 `FA_BW_HEAD_DIM`，Python 构建会按输入覆盖该宏。

D128 默认 Q64/C32，占 32 KiB LDS。对照脚本测试 Q64/Q128 × C32/C64，启用已在 D64 测过的 async K/V、defer denominator、delay scale；每组先运行 D128 正确性与指令标签测试，再对比强制 Flash SDPA：

```bash
bash compare_d128.sh
```

默认正是 B=1、H=32、N=M=4096、D=128。结果逐组打印并保存到独立的 `build/d128/run-*/`，末尾打印可直接粘贴的汇总。脚本报错时停止，不跳过失败测试。首轮这四组结果见上表；默认配置仍保持 Q64/C32，使用最快实测配置需要显式设置 Q128 并开启上述优化。

只验证默认配置并测一次：

```bash
unset FA_BW_CXXFLAGS
python -m unittest test_head_dim -v
python bench.py --batch 1 --heads 32 --lengths 4096 --head-dim 128 --backend flash --json results_d128.json
```

首次 D128 实机反馈中，Q64/C32 的 attention 数值用例（包括后 64 个通道和 D64/D128 混用）通过；三项独立探针报错，脚本未进入计时。日志中的 V 期望行步长仍为 64、C32 的标签总数仍为 8192；K 的错误也与旧 D64 探针读取新 D128 布局完全吻合。请同步 **`test_matrix_v.hip`、`test_async_k.hip` 和 `test_flash_attn.py`**，不只更新 kernel/头文件。当前入口会在运行前检查探针版本；新版探针打印 `D=128`，C32 的 V 标签总数应为 16384、K 两个 tile 的标签总数应为 8192。同步后重新运行 `bash compare_d128.sh`，保留全部校验。

随后同步探针后，Q64/C32 的 18 项测试全部通过，包括 K32/K64 拷贝探针和全部四个 V panel；计时又被远端旧 `bench.py` 的 `unrecognized arguments: --head-dim 128` 阻止。请同时更新 **`bench.py` 和 `compare_d128.sh`**。新版 benchmark 的 tensor shape、默认 scale、LDS 元数据和 FLOPs 都随 D 改变；比较脚本会先检查 `--head-dim` 参数是否存在，再运行测试。这次中断尚未产生计时；后续完整运行的结果见本节首轮实测表。

D128/C64 的 K/V 双缓冲占 64 KiB LDS；D128/C128 需要 128 KiB，目前编译时拒绝。Q/K 的归约覆盖 8 个 16-channel 片段，V 矩阵读取覆盖 4 个 32-channel panel；测试专门检查后 64 个输入/输出通道、K/V 搬运、最高 V buffer 的 guard、尾块及 D64/D128 混用。D64 的既有成绩不能外推为此规模的性能。

单独采集 D128 workload 时可用 `profile_workload.py --batch 1 --heads 32 --length 4096 --head-dim 128 --impl custom`（或 `--impl sdpa`）。现有 `profile_hipprof.sh` 的预设仍是 D64，不要将其报告误当 D128。

## 在 BW1000 上运行

激活 DTK / PyTorch 环境，确保 `hipcc` 或 `aicc` 在 PATH 上。整个目录复制到 BW1000 即可。

```bash
cd /path/to/fa_bw
# 如编译器不在 PATH：export HIPCC=/实际DTK路径/bin/hipcc
# 使用当前默认配置前，清除以前 export 的实验参数。
unset FA_BW_CXXFLAGS
python flash_attn_bw.py
python -m unittest test_flash_attn -v
python bench.py --json results.json
```

当前 `test_flash_attn` 有 19 项测试。默认配置跳过未开启的 async K/V 两项探针；同时开启 `FA_BW_ASYNC_K=1` 和 `FA_BW_ASYNC_V=1` 时全部执行，包括 C32/C64/C128 拷贝标签检查。这些探针及 Q64/Q128 的 C128 attention 用例已通过 BW1000 验证。显式设置 `FA_BW_MATRIX_V=0` 时也会跳过矩阵 V 标签测试。

首次构建或源码、`shared_layout.h`、`matrix_v.h`、`async_v.h`、`async_k.h`、编译参数变化时重新编译，产物放在 `build/`。更新时请复制整个目录，包含新增头文件。默认优先使用 `aicc`，其次 `hipcc`；`HIPCC` 可指定编译器。额外参数可通过 `FA_BW_CXXFLAGS` 设置。默认不启用全局 fast-math，以保留 softmax 的无穷值掩码语义。

```bash
# 指定规模，并强制 SDPA Flash 后端；后端不可用时直接报错。
python bench.py --batch 4 --heads 8 --lengths 512 1024 2048 4096 --backend flash --json flash.json
# 交叉注意力和尾块
python bench.py --lengths 65 513 --kv-length 1001
# Graph 不受当前运行时支持时使用普通事件计时
python bench.py --timing event
```

`--backend auto` 使用 PyTorch 正常选择的 SDPA 后端，输出 profiler 观察到的 SDPA 算子名；不要仅凭接口名认为运行的是 FlashAttention。`flash/efficient/math` 可分别强制选择，不能使用时不会静默回退。

计时前对全部输出与 SDPA 比较，并对抽样 query 行与独立 FP32 reference 比较；不通过就停止。默认对双方都预热，通过 HIP graph 重放、GPU event 计时，取多轮中位数。Graph 模式排除 ctypes/Python 调用开销和每次分配的 CPU 开销；自定义 kernel 使用预分配输出。`event` 模式包含 Python 供给速度造成的 GPU 空闲，不能当作纯 kernel 延迟。TFLOP/s 采用两次 GEMM 共 `4*B*H*N*M*D` FLOPs；speedup=`SDPA时间/本kernel时间`，大于 1 才表示本 kernel 更快。

Python 调用：

```python
from flash_attn_bw import attention
o = attention(q, k, v)                 # 同当前 PyTorch stream
o = attention(q, k, v, out=o)          # 可复用输出，但不能与 Q/K/V 共用 storage
```

## 列置换为何能省掉 P 的 layout 转换

令 lane `t`，`i=t%16`，`g=t//16`，寄存器序号 `r=0..3`。gfx936 的单条 16×16×16 MMAC 使用：

| fragment | lane t 的第 r 个元素 |
|---|---|
| A | `A[i, 4*g+r]` |
| B | `B[4*g+r, i]` |
| C | `C[i, g+4*r]` |

在每个 16-key 块内，把原始 QK 的第 `j` 列放到 `perm(j)=4*(j%4)+j//4` 列。这个置换是自身的逆，因此读取物理列 `j` 时取原始 key `perm(j)` 即可，无需先生成一份重排 K。

于是 C 的第 r 个寄存器对应原始 key：

```text
perm(g + 4*r) = 4*g + r
```

softmax 保持寄存器归属不变，转成 FP16 后直接作为下一次 MMAC 的 A 输入。**V 的逻辑 key 顺序不需要置换**。同一 query 行位于相隔 16 个 lane 的四个线程上，行归约使用 xor 16 和 xor 32。最终 O 按普通 C 布局 `g+4*r` 写回，不能套用 P 的解释。

## 实现选择与验证范围

- D64 默认采用 128×64×64 分块，D128 默认采用 64×32×128 分块；可编译设置 `FA_BW_Q_TILE=64/128/192/256` 和 `FA_BW_KV_TILE=32/64/128`，C128 要求矩阵 V。每 block 四个 wave64，共 256 线程。
- Q 常驻寄存器。D64 的 K/V 双缓冲分别占 16/32/64 KiB LDS（KV tile=32/64/128），D128 的 C32/C64 分别占 32/64 KiB，复用于 Q 和输出暂存。
- K/V 全局加载采用每线程 16 字节，下一块先读进寄存器，在本块 PV 后写入另一 LDS buffer。实际 load/compute 重叠程度取决于编译器调度；这里没有照搬 CUDA `cp.async`。
- 默认 V 保持逻辑行主序，以 16 字节向量写入 LDS，矩阵读取指令生成两个 B fragment。物理地址使用 `row*D + (col ^ ((row&1)*32))`，Q/K 则按四元素组做 XOR swizzle。
- QK、PV、online softmax 的 m/l 和输出累加使用 FP32，P 转成 FP16，最终只做一次输出归一化。与原 CUDA 的 FP16 累加并非逐位一致。
- softmax 的指数计算默认调用 `__builtin_amdgcn_exp2f`。减去行最大值后，输入在精确算术下非正，FMA 舍入可能留下正残差。可编译设置 `FA_BW_NATIVE_EXP2=0` 恢复第二版的 `exp2f` 路径；该选项不启用全局 fast-math。
- 默认 `FA_BW_MATRIX_V=1` 使用 `__builtin_hcu_ds_read_m32x16_f16`；参考依据是普通 LDS 的 `ds_read_format_tensor_b_linear`，不依赖 MLS atom 特化。原 D64 六组分块均在 gfx936 上通过全部 attention 测试和 16,384 项标签校验；设置为 0 可恢复旧的转置 V 路径。
- 默认 `FA_BW_FULL_KV=1`：M 整除 KV_TILE 时选择完整块 kernel，否则保留带边界检查的 kernel。D64 的两条路径均通过 GPU 测试。

参考实现依据：`tilelang-hygon/src/tl_templates/hcu/gemm.h` 的 MMAC builtin、`tilelang/hcu/intrinsics/hcu_mmac_layout.py` 的 A/B/C lane 映射。

本地环境没有 DTK 编译器或可用 BW1000。CPU 测试验证列置换、寄存器重解释、online softmax 数学关系，以及实际 `shared_layout.h` 的索引双射、向量连续性和 32-bank 地址分布；**不验证设备 intrinsic、实际指令、同步或性能**。LDS 地址测试需要主机 `c++` 编译器，没有时会跳过。GPU 测试覆盖尾块、batch/head 偏移、交叉注意力、scale、较大 logits、非默认 stream 和输入校验，必须在 BW1000 上执行。

## 第一版 BW1000 实测与后续修正

以下第一至第十一轮记录各轮测试时的配置和默认值；当前默认以上文为准。旧比较脚本已显式固定各自的历史配置，避免默认值更新改变对照条件。

用户在 BW1000、PyTorch 2.7.1、HIP 6.3.26045 上反馈：第一版编译成功，所有 kernel 用例均通过独立 FP32 reference；但 `scale=0` 和 `scale=-0.25` 与默认 SDPA 比较时出现 NaN。当前测试对这两类 scale 显式使用 SDPA Math，仍保留独立 FP32 校验、有限值检查和原始误差阈值。正 scale 继续使用默认 SDPA，性能脚本仍默认比较 Flash SDPA；没有更换性能基线。

第一版 `B=1,H=16,D=64`、Graph 模式的用户实测（**不是当前改版的成绩**）：

| N=M | 本 kernel (μs) | Flash SDPA (μs) | 加速比 |
|---:|---:|---:|---:|
| 256 | 16.46 | 23.70 | 1.440x |
| 512 | 41.19 | 31.99 | 0.777x |
| 1024 | 130.18 | 69.02 | 0.530x |
| 2048 | 423.77 | 193.75 | 0.457x |
| 4096 | 1497.08 | 659.24 | 0.440x |

第二版只调整 V 的 LDS swizzle。旧地址映射在模拟的 V scalar 写入分组中每 bank 最多访问 8 个不同 dword，PV 向量读取中为 2 个；新映射两者均为 1。用户反馈第二版 7 项测试全部通过，实际测得：

| N=M | 第二版 kernel (μs) | Flash SDPA (μs) | 加速比 |
|---:|---:|---:|---:|
| 256 | 14.83 | 23.16 | 1.562x |
| 512 | 34.53 | 31.37 | 0.908x |
| 1024 | 107.28 | 68.89 | 0.642x |
| 2048 | 339.43 | 193.02 | 0.569x |
| 4096 | 1186.11 | 659.21 | 0.556x |

## 第三版：原生 exp2 对照实验

参考 `tilelang-hygon/examples/hcu/fa_v1.py` 的说明，其 FA 显式打开将 `exp2f` 转换为硬件指数指令的编译选项。我们的第二版编译没有启用此选项，因此第三版直接在 softmax 中调用原生 builtin，避免依赖编译器是否对通用数学函数做近似替换。分块、共享内存、MMAC 和 SDPA 基线均保持第二版设置。

新增测试覆盖 4097 个 key（tile=32 时为 129 个 tile），每个 tile 都缓慢提高行最大值，并逆序重跑以检查 online rescaling 的累计误差；原来的尾块和非正 scale 测试继续保留。原生 builtin 对 `-inf` 的处理也通过首块初始化、尾块和单 key 用例间接验证。用户已反馈第三版 8 项测试全部通过。

同步整个目录后，建议在同一环境分别运行（每条命令启动新进程；编译选项变化会自动重编译）：

```bash
(
# 固定第三版的 Q64/C32、转置 V、通用 KV 配置。
export FA_BW_CXXFLAGS="-DFA_BW_Q_TILE=64 -DFA_BW_KV_TILE=32 -DFA_BW_MATRIX_V=0 -DFA_BW_FULL_KV=0"
# 默认 FA_BW_NATIVE_EXP2=1
python -m unittest test_flash_attn -v
python bench.py --json results_v3_native.json

# 控制组：恢复第二版的通用 exp2f 调用
FA_BW_CXXFLAGS="$FA_BW_CXXFLAGS -DFA_BW_NATIVE_EXP2=0" python bench.py --json results_v3_libm.json
)
```

JSON 和控制台新增 `exp2` 字段，读取已加载库的编译选择（`native` 或 `libm`），不是根据当前环境变量猜测。它记录调用路径，不等同于反汇编结果：编译器仍可能自行优化 `libm` 路径。若原生版本未提速，就不能把长序列瓶颈归因于通用指数函数。

第三版同机对照实测（用户提供）：

| N=M | native (μs) | libm (μs) | native 对应 Flash SDPA (μs) |
|---:|---:|---:|---:|
| 256 | 14.71 | 14.84 | 23.20 |
| 512 | 30.85 | 34.29 | 31.50 |
| 1024 | 93.10 | 107.28 | 68.83 |
| 2048 | 292.95 | 339.50 | 193.63 |
| 4096 | 1015.19 | 1186.19 | 659.37 |

## 第四版：KV tile=32 / 64 对照

在原生 exp2 的基础上增加 `FA_BW_KV_TILE=64` 编译选项，默认仍为 32。每条 MMAC 的 16×16 列置换保持不变；一次处理四个 score fragment，替代原来的两个。对于 M 为 64 的倍数，tile=64 的 online max/sum 更新、输出重缩放和循环内 block barrier 次数减半；总 QK/PV 运算量、概率指数计算量不变。代价是更大的 score/prefetch 寄存器数组和两倍 LDS，占用率可能降低。

CPU 检查已覆盖两种配置的 P 寄存器重解释、尾块、LDS 双射和向量连续性；两者的 V 地址分布检查都达到每 bank 最多一个不同 dword。新增 GPU 规模覆盖 63/64/65 和 129 个 key。用户已反馈两种配置各 8 项测试全部通过；tile=64 在已测规模上均更慢，因此默认继续采用 tile=32。

同步整个目录后运行：

```bash
bash compare_tiles.sh
# 或缩短第一轮规模
bash compare_tiles.sh --lengths 256 1024 4096
```

脚本固定 Q64、转置 V、通用 KV 路径和原生 exp2，依次对 tile=32/64 运行完整测试和基准；任一测试失败就停止。输出为 `results_v4_c32.json` 和 `results_v4_c64.json`，含编译得到的 `kv_tile`、`lds_bytes` 字段。脚本显式设置历史配置所需编译宏，会替换调用环境的 `FA_BW_CXXFLAGS`。

第四版实测延迟（μs，用户提供）：

| N=M | tile=32 | tile=64 |
|---:|---:|---:|
| 256 | 14.34 | 15.52 |
| 512 | 30.92 | 33.33 |
| 1024 | 92.88 | 117.78 |
| 2048 | 292.36 | 386.68 |
| 4096 | 1015.96 | 1360.13 |

## 编译器资源与汇编诊断

tile=64 虽然减少 barrier 和在线更新次数，但实际退化。单凭计时还无法区分寄存器溢出、驻留 wave 数减少或调度变化；下一步收集编译证据：

```bash
python diagnose.py
```

该脚本使用当前 DTK 编译器，分别编译原生 exp2 的 tile=32/64，添加 `-save-temps=obj` 和 `-Rpass-analysis=kernel-resource-usage`。每次运行都创建独立目录保存编译日志、汇编和其他中间产物；不会加载生成的库或运行 GPU。控制台报告也保存在 `build/diagnostics/summary.txt`。

摘要包含编译器资源报告、AMDHSA kernel descriptor（VGPR/SGPR、LDS、private segment 等）和目标 kernel 的部分静态指令计数。完整 opcode 计数保存在本次运行的 `summary.json` 中。静态计数不能代表动态执行次数、等待时间或实际占用率；`private_segment` 非零也不能单独证明 spill，需结合编译报告和访存指令判断。若无法识别 kernel 汇编，脚本会明确提示并保留所有文件，不把缺失数据当成零。

本地摘要解析测试和脚本语法检查已通过；诊断编译需在 BW1000 的 DTK 环境运行。编译器版本字符串可能带仓库 URL 的认证信息，诊断输出会将 URL userinfo 脱敏。

用户提供的资源报告确认（编译器报告值）：

| 配置 | VGPR | SGPR | LDS/块 | waves/SIMD | Scratch | SGPR/VGPR spill |
|---|---:|---:|---:|---:|---:|---:|
| C32 | 96 | 29 | 16 KiB | 2 | 0 | 0 / 0 |
| C64 | 135 | 31 | 32 KiB | 1 | 0 | 0 / 0 |

两者均无 spill，C64 的编译器 occupancy 由 2 降为 1，是资源占用导致其退化的重要证据。这些数字是编译器上限估计，不是 profiler 测得的动态驻留率；不能单凭汇编中的静态 waitcnt 数量推算等待耗时。

## 第五版：缩短 K/V 预取寄存器的存活时间

增加 `FA_BW_LATE_PREFETCH=1` 候选：在 softmax 生成 P 之后才加载下一块 K/V，随后执行 PV，并像原版一样在 PV 后写入另一 LDS buffer。两种配置均保持 C32、原生 exp2、16 KiB LDS、相同的 MMAC 数量和 block barrier。默认 `FA_BW_LATE_PREFETCH=0` 保持已验证的 QK 前预取。

候选用消费 P 的空 inline asm 加 memory clobber 约束编译器，防止下一块全局加载被提升回 softmax 之前；这不是硬件同步，也不增加 GPU barrier。它减少预取数据与 QK/softmax 临时值的生命周期重叠，但预取可覆盖的计算时间也缩短。实际 VGPR、occupancy 和耗时需要同时比较；没有预设候选一定更快。

```bash
bash compare_prefetch.sh
```

脚本固定 C32/原生 exp2，依次编译 early/late 两种位置，使用 `-Rpass-analysis=kernel-resource-usage` 打印 VGPR、scratch、spill、occupancy，再跑完整正确性测试和性能对比。输出为 `results_v5_prefetch0.json` 和 `results_v5_prefetch1.json`；JSON 的 `prefetch` 字段来自编译好的库。脚本会设置本次实验的 `FA_BW_CXXFLAGS`，支持追加 benchmark 参数，例如 `bash compare_prefetch.sh --lengths 256 1024 4096`。

用户已验证两种预取位置各 8 项测试全部通过。延后预取将 VGPR 从 96 降到 91，但两者仍为 2 waves/SIMD、无 spill；N=4096 的 Graph 延迟从 990.78 μs 增至 1049.56 μs。因此保留 `FA_BW_LATE_PREFETCH=0` 为默认，未采用延后预取。

第五版实测（μs，用户提供）：

| N=M | 原预取 | 延后预取 | 原预取对应 Flash SDPA |
|---:|---:|---:|---:|
| 256 | 13.86 | 15.96 | 23.14 |
| 512 | 30.24 | 33.60 | 31.34 |
| 1024 | 91.13 | 96.54 | 68.78 |
| 2048 | 285.61 | 305.85 | 192.87 |
| 4096 | 990.78 | 1049.56 | 659.09 |

## 使用 DTK 26.04 hipprof 采集

根据目标机器 `hipprof -h` 确认的接口，新增固定 workload 和采集脚本。默认 B=1、H=16、N=M=4096、D=64，分别运行 C32 原预取、C32 延后预取、强制 Flash SDPA：

```bash
bash profile_hipprof.sh
# 可分别采集，或指定另一种长度
bash profile_hipprof.sh trace 4096
bash profile_hipprof.sh pmc 4096
```

每个自定义版本先在 profiler 外编译并运行正确性测试；失败立即停止。脚本固定 C32、原生 exp2，只改变预取位置，设置本次实验的编译参数。目标 workload 使用 CPU 随机输入再拷贝到 GPU，避免 GPU RNG/转换 kernel 混入；不包含参考计算、torch.profiler 或 Graph。Flash SDPA 继续强制使用 Flash 后端，不可用时不会回退。

- Trace 命令采用 `hipprof --hip-trace --output-type 0`：每组 5 次 warmup + 20 次重复调用，**warmup 也在 trace 中**。输入复制和初始化 API 仍可能出现；应查看目标 kernel，并区分前 5 次注意力调用。SDPA 一次调用是否包含多个 device kernel，以实际 trace 为准。
- PMC 命令采用 `hipprof --pmc --pmc-type 3`：每组 0 次 warmup + 3 次调用，输出该版本实际支持的计数器。默认不设 `--kernel-name`，避免未经确认的名字匹配规则漏掉目标 kernel。
- Eager trace 中的 Python 调用间隙不是 kernel 内部停顿。PMC 可能重放/串行化执行，插桩时间不能直接替代 `bench.py` 的 Graph 计时来计算最终加速比。

每次采集创建 `build/hipprof/run-XXXXXX/`，下分 `before_qk/`、`after_softmax/`、`sdpa/` 三个目录，保存日志及 hipprof 生成的报告。新版脚本在采集结束后自动生成同名 `.tar.gz`，可直接提供完整采集文件。分析时先查看各目录的 **PMC CSV** 和 **Trace 统计 CSV/日志**；需要核对调用间隙时再查看完整 JSON timeline。若目标 PMC 字段包含占用率、算术管线利用率、访存流量或 LDS 相关指标，再据此分析，不假定所有架构都有同一组计数器。

也可直接运行单组（先用相同编译参数通过测试）：

```bash
FA_BW_CXXFLAGS="-DFA_BW_Q_TILE=64 -DFA_BW_KV_TILE=32 -DFA_BW_MATRIX_V=0 -DFA_BW_FULL_KV=0 -DFA_BW_LATE_PREFETCH=1" hipprof --pmc --pmc-type 3 -o late_pmc \
    python profile_workload.py --impl custom --length 4096 --warmup 0 --repeats 3
```

用户已完成一次三组采集：Trace 中各有 25 次目标 kernel，设备端均值分别为原预取 996.426 μs、延后预取 1060.816 μs、Flash SDPA 663.533 μs，与未插桩 Graph 计时接近。SDPA 显示为 `flash_fwd_kernel_16x64_prefetch` 系列，完整模板名在控制台被截断。启动阶段的 hipMalloc、输入拷贝耗时不能解释稳态 kernel 差距。

随后收到 `run-8ynTvp` 的完整 PMC CSV 文字预览。下表的指令数是每次 dispatch 的计数，三次采样一致；时间来自之前未插桩的 Graph benchmark：

| 指标，B1/H16/N=M4096 | 原预取 | 延后预取 | Flash SDPA |
| --- | ---: | ---: | ---: |
| 时间（μs） | 990.78 | 1049.56 | 659.09 |
| block 数（grd / wgr） | 1024 | 1024 | 256 |
| arch_vgpr | 96 | 92 | 232 |
| LDS 字节 | 16384 | 16384 | 16384 |
| scratch 字节 | 0 | 0 | 0 |
| SQ_INSTS_LDS | 15,843,328 | 15,843,328 | 1,581,056 |
| SQ_INSTS_VALU | 78,213,120 | 78,213,120 | 30,524,416 |
| SQ_INSTS_VMEM_RD | 1,064,960 | 1,064,960 | 401,408 |
| SQ_LDS_BANK_CONFLICT | 0 | 0 | 0 |

当前 LDS 指令数约为 SDPA 的 10 倍，报告中的 bank conflict 为零。延后预取的 `SQ_WAIT_INST_LDS` 均值从约 2839 万降至 1644 万，但运行更慢；这个聚合计数不能直接当作耗时比例。SDPA 使用更多寄存器仍然更快，下一步应检查数据复用，不能单靠压低 VGPR 数来选方案。VMEM 指令数也不等同于传输字节或带宽。

完整 SDPA 名称包含 `Flash_fwd_kernel_16x64_prefetch_traits_dim64<64,256,64,4,...>`。结合 65536 个 query row 只启动 256 个 block，这支持其每 block 处理 256 行 Q、每轮 64 个 key 的判断；尚未检查其源码，不能由名字中的 `16x64` 推断每 block 只有 16 行 Q。

### 不方便下载报告时，直接粘贴文字

将新增的 `dump_hipprof.py` 放到 BW1000 的 `fa_bw` 目录后运行：

```bash
python dump_hipprof.py build/hipprof/run-8ynTvp
```

它只依赖 Python 标准库，列出报告文件，并输出 PMC CSV 的前几条记录、SQLite DB 的表名/列名和样例值。数据库使用只读连接；不重新采集，也不假定表结构或指标单位。将终端输出直接粘贴回来即可。

默认每张表预览 3 行、最多抽样 8 张表，优先抽样名字包含 pmc/counter/metric/kernel/dispatch/ops 的表，所有表的列名都会列出。如果有截断提示，可增加 `--rows 20 --max-chars 10000`，或用 `--table '实际表名'` 定位需要的表。BLOB 只显示长度；空表、读取失败或缺失数据会明确标注，不会当作零利用率。本地已用真实 CSV/SQLite 样例检查引号、数值、空表和数据库内容不变。

## 第六版：增加 Q 行数，复用 K/V fragment

新增编译选项 `FA_BW_Q_TILE=64/128/256`，默认仍为 64。固定 256 线程，每个 wave 分别保存 1/2/4 组独立的 16 行 Q、输出累加器和 softmax 状态。每次从 LDS 读取的 K fragment 用于多组 QK，每个 V fragment 用于多组 PV；原有 P 寄存器直接复用的列置换不变。

Q 和输出都按 64 行分批经过同一份共享暂存区，每次复用前完成 block 同步。因此 C32 下 LDS 始终为 **16 KiB**，Q256 不会扩大到 32 KiB。代价是更多寄存器和首尾暂存同步，短序列的 block 数也减少；是否提速必须结合实际寄存器、spill 和计时判断。

同步更新后的代码到 BW1000 的 `fa_bw` 目录，然后执行：

```bash
bash compare_query_tiles.sh
# 或先只比较三个规模
bash compare_query_tiles.sh --lengths 256 1024 4096
```

脚本依次跑 Q64/Q128/Q256，固定 C32、原生 exp2 和 QK 前预取，打印编译资源信息，完整正确性测试通过后才计时。SDPA 默认强制 Flash 后端，输出 `results_v6_q64.json`、`results_v6_q128.json`、`results_v6_q256.json`。它会设置本次实验的 `FA_BW_CXXFLAGS`；JSON 中新增的 `q_tile` 字段读取实际编译库。

新增 GPU 测试覆盖 127/128/129、255/256/257 行 Q、多 batch/head、各 query 组不同的 logits 分布，以及预填 NaN 和前后保护区的输出 buffer。4097-key 的在线 softmax 测试也扩展至 257 行 Q，每行使用不同的 logits 斜率，覆盖多个 query part 的反复重缩放。用户已反馈三档 C32 配置各 9 项测试全部通过。

第六版实测，B1/H16、C32、原生 exp2、QK 前预取、Graph 计时，单位 μs：

| N=M | Q64 | Q128 | Q256 | Flash SDPA（三组范围） |
| ---: | ---: | ---: | ---: | ---: |
| 256 | **13.68** | 19.22 | 30.25 | 23.01–23.81 |
| 512 | **30.11** | 31.10 | 48.40 | 31.23–32.10 |
| 1024 | 90.55 | **80.18** | 86.09 | 68.66–68.98 |
| 2048 | 284.80 | 262.34 | **251.56** | 192.89–193.70 |
| 4096 | 988.01 | **846.50** | 888.01 | 659.00–660.11 |

| Q tile | VGPR | SGPR | waves/SIMD | LDS（字节） | scratch / spill |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 97 | 32 | 2 | 16384 | 0 / 0 |
| 128 | 140 | 46 | 1 | 16384 | 0 / 0 |
| 256 | 211 | 50 | 1 | 16384 | 0 / 0 |

Q128 在 N=4096 比 Q64 减少约 14.3% 耗时，但相对同组 Flash SDPA 仍为 0.779x。增加 query 复用有收益，但不能一直增大：Q256 在 N=4096 又比 Q128 慢。三组都没有 spill，不能把这种退化归因于寄存器溢出。Q64 仍适合已测短序列；目前不添加跨 batch/head/交叉注意力规模的自动选择规则，默认继续为 Q64/C32。

## 第七轮：Q 行复用与 C64 联合对比

此前 Q64 从 C32 改 C64，编译器报告的占用率从 2 降到 1 wave/SIMD，性能退化。现在 Q128/Q256 在 C32 下已经是 1 wave/SIMD，因此值得单独检查它们采用 C64 的效果：对于 M 为 64 的倍数，在线 max/sum 更新、输出重缩放和循环 barrier 次数减半。C64 的 LDS 仍为 32 KiB，寄存器也会增加，尤其 Q256 需要检查是否出现 spill；不能预先断言会更快。

这轮使用内核已有的编译选项，新增脚本依次比较 Q128/C32、Q128/C64、Q256/C32、Q256/C64。每组都先打印编译资源并通过完整正确性测试，再与强制 Flash SDPA 比较：

```bash
bash compare_reuse_tiles.sh --lengths 1024 2048 4096
```

输出为 `results_v7_q128_c32.json` 等四个文件。第七轮只添加了脚本，没有修改 HIP kernel 或 Python 接口。用户已反馈四组各 9 项测试全部通过，实测如下（μs）：

| N=M | Q128/C32 | Q128/C64 | Q256/C32 | Q256/C64 | Flash SDPA（四组范围） |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 81.02 | **75.21** | 85.97 | 85.24 | 68.86–69.05 |
| 2048 | 263.05 | 266.47 | 251.59 | **247.98** | 192.80–193.16 |
| 4096 | **846.80** | 878.44 | 887.82 | 914.27 | 659.49–659.97 |

Q128/C64 为 188 VGPR、32 KiB LDS、1 wave/SIMD、无 spill。Q256/C64 达到 256 VGPR，产生 **17 个 VGPR spill、72 字节 scratch/lane**。所有组合都正确，但 C64 没有改善 N=4096 的差距。Q128/C32 保持为下一轮长序列实验的对照组；默认配置仍为 Q64/C32。

## 第八轮：V 使用矩阵 LDS 读取

此前检查 builtin 时仅查看 MLS atom 路径，漏掉了参考仓库的普通 LDS 路径。`src/tl_templates/hcu/mls/tilelang_ds_read_format.hpp` 中的 `ds_read_format_tensor_b_linear` 使用 `__builtin_hcu_ds_read_m32x16_f16`：一个 wave 读取逻辑 `[16 key,32 channel]` panel，每个请求 lane 取 8 个连续 half，指令将数据分发为两个 MMAC B fragment。

对于 lane `t`、fragment `f=0/1`、元素 `r=0..3`，生产 helper 期望返回：

```text
request(t): V[key_tile*16 + t/4, panel*32 + (t%4)*8 : +8]
result[4*f+r]: V[key_tile*16 + 4*(t/16)+r, panel*32 + 16*f+t%16]
```

新增 `FA_BW_MATRIX_V=1` 候选：V 保持逻辑行主序，每线程用一个 16 字节向量写入 LDS，替换原来转置时的 8 次 half 写入。读取时，一次 builtin 提供两个 B fragment。QK、P 寄存器复用、FP32 累加、在线 softmax 和双缓冲同步均保持原逻辑。C32 仍占 16 KiB LDS，默认 `FA_BW_MATRIX_V=0` 保留已验证路径。

V 的新物理地址为 `row*D + (col ^ ((row&1)*32))`，将奇数 key 行的两个 32-channel panel 交换；生产与读取使用同一地址函数，16 字节向量保持连续且对齐。CPU 测试验证双射、向量连续性以及参考矩阵读取 phase 的地址分布。参考 phase/layout 测试面向 gfx938/gfx946，不能代替 gfx936 验证。

因此增加独立设备测试 `test_matrix_v.hip`，使用生产 `load_v_matrix`，逐位核对 16,384 个 lane/register 标签，覆盖 4 个 key tile、2 个 channel panel 和 4 个 wave。测试失败会报告首批错误坐标；完整 attention 数值测试也必须通过。

同步整个更新目录到 BW1000 后运行：

```bash
bash compare_matrix_v.sh --lengths 1024 2048 4096
```

脚本固定 Q128/C32、原生 exp2、QK 前预取，比较旧转置路径和矩阵读取路径，打印编译资源，并在完整测试之后计时。旧路径运行 10 项测试，其中标签测试跳过；新路径应 10 项全部通过，并打印 `Matrix V tags: 16384/16384 match`。两组 JSON 为 `results_v8_matrix0.json` / `results_v8_matrix1.json`，新增 `v_load` 字段来自实际编译库。

第八轮首轮反馈：矩阵 V 配置在 gfx936 编译成功，完整 attention 数值测试通过；VGPR 从 140 降到 128，编译器报告占用率从 1 升到 2 waves/SIMD，LDS 仍为 16 KiB，无 scratch/spill。但标签探针只匹配 2048/16384 项，每个 lane 的 8 个值均重复第一个，脚本因此正确停止，**没有矩阵 V 的性能结果**。

本地 Clang 最小 C++ 复现确认探针的 `__builtin_bit_cast(uint16_t, value[r])` 会重复转换向量的第一个元素；这是探针提取方式的问题。改成整个 128-bit 向量的位转换和写回后，本地复现中的所有元素均正确。仅修改 `test_matrix_v.hip`，attention 内核、生产装载 helper、所有期望标签和失败条件不变。gfx936 的 16,384 项标签校验仍须重跑通过，之后才能收集性能。

这一修正只需复制 `test_matrix_v.hip`，可直接重跑未完成的矩阵路径：

```bash
export FA_BW_CXXFLAGS="-DFA_BW_Q_TILE=128 -DFA_BW_KV_TILE=32 -DFA_BW_MATRIX_V=1 -DFA_BW_FULL_KV=0 -DFA_BW_NATIVE_EXP2=1 -DFA_BW_LATE_PREFETCH=0 -Rpass-analysis=kernel-resource-usage"
python -m unittest test_flash_attn -v && python bench.py --backend flash --lengths 1024 2048 4096 --json results_v8_matrix1.json
```

修正后的 BW1000 结果：**10 项测试全部通过，`Matrix V tags: 16384/16384 match`**。Q128/C32、矩阵 V、原生 exp2、QK 前预取的实测如下（μs）；旧 V 对照来自此前同组采集：

| N=M | 转置 V | 矩阵 V | Flash SDPA（矩阵 V 运行时） | 矩阵 V / SDPA 加速比 |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 80.36 | 74.94 | 68.74 | 0.917x |
| 2048 | 262.49 | 246.49 | 192.87 | 0.782x |
| 4096 | 846.58 | 796.87 | 659.36 | 0.827x |

N=4096 的耗时减少约 5.9%，但仍比 SDPA 多约 20.9%。保留该路径用于后续长序列优化；这组结果只覆盖 Q128/C32，尚未据此改变 Q64 的默认路径或加入通用形状选择规则。

### 重新采集当前 Q128 的 PMC 和汇编

旧 PMC 使用 Q64/转置 V，不能直接解释当前 Q128/矩阵 V 的剩余差距。新增 `matrix` 对照模式，固定 Q128/C32，分别采集转置 V、矩阵 V 和强制 Flash SDPA：

```bash
bash profile_hipprof.sh all 4096 matrix
```

更新 `profile_hipprof.sh` 和 `dump_hipprof.py` 后即可执行。脚本在 profiler 外验证两个自定义版本；`all` 模式还调用现有 `diagnose.py --tiles 32`，保存实际汇编并打印资源和静态 opcode 计数。之后采集三组 trace/PMC，并自动打印精简 CSV。输出目录为 `build/hipprof/run-XXXXXX/{transposed,matrix,sdpa}/`；直接粘贴终端输出，无需下载文件。

精简结果保留 kernel 名、grid/workgroup、LDS/scratch/寄存器和所有 `SQ_*` 字段，不将指令次数当成字节数或利用率，也不使用 PMC 插桩计时代替 benchmark。现有两参数命令仍默认运行旧的 Q64 预取位置对照；第三参数必须指定 `matrix` 才是本次比较。

也可对已有报告使用 `dump_hipprof.py --columns '^(KernelName|SQ_.*)$'` 筛选 CSV 列。筛选以第一条 CSV 记录为表头，列数异常时显示原始记录，无匹配列时显示表头；不影响 SQLite 输出。

当前 Q128 采集已返回。Trace 中每组各 25 次调用，均值为转置 V **853.828 μs**、矩阵 V **801.721 μs**、Flash SDPA **663.744 μs**。PMC 中下列每次 dispatch 的指令计数在三次采样中一致：

| 指标 | 转置 V | 矩阵 V | Flash SDPA |
| --- | ---: | ---: | ---: |
| SQ_INSTS_LDS | 9,027,584 | 6,144,000 | 1,581,056 |
| SQ_INSTS_VALU | 60,217,344 | 57,251,840 | 30,524,416 |
| SQ_INSTS_VMEM_RD | 540,672 | 540,672 | 401,408 |
| SQ_LDS_BANK_CONFLICT | 0 | 0 | 0 |
| scratch | 0 | 0 | 0 |

矩阵 V 使 LDS 指令减少约 31.9%，实际提速小于这个比例。其 LDS 计数恰好符合 `2048 waves * (128 iterations * 23 + 56)`：每轮 8 次 K 读取、4 次 V 矩阵读取、3 次 K/V 暂存写入、8 次 softmax shuffle。shuffle 共 2,097,152 条；这些也是 LDS 指令，不能把整个计数都视为共享数组读写。尚未确认 gfx936 可用的更快跨行归约指令，不直接套用其他架构的 permlane 指令。

## 第九轮：完整 KV 块专用路径

新增 `FA_BW_FULL_KV=1` 候选。当 `M % KV_TILE == 0` 时，host 选择 `FullKV=true` 模板 kernel，去掉每线程 K/V 加载的尾块判断及 score 掩码；`M` 有尾块时自动调用 `FullKV=false`，保持原有掩码行为。Q 和输出的边界检查继续保留，非默认 stream、softmax、P 布局、数值精度和预取逻辑保持原样。

默认 `FA_BW_FULL_KV=0` 只使用通用路径。新选项的目标是测量重复边界处理的实际成本；当前计数并不能证明这一改动一定提速。benchmark/profile 的 `full_kv_specialization` 字段记录库是否启用该分派功能，具体调用只有在 M 整除时才走专用 kernel。

同步整个目录到 BW1000 后执行：

```bash
bash compare_full_kv.sh --lengths 1024 2048 4096
```

脚本固定 Q128/C32/矩阵 V/原生 exp2/原预取，对比关闭和启用专用路径。测试扩展了 `M=31/32/33` 附近的 Q 尾块用例，并让非正 scale、数值稳定性、非默认 stream 都同时覆盖完整和不完整 KV 块。每组完整测试通过后才计时，输出 `results_v9_fullkv0.json` / `results_v9_fullkv1.json`。

启用时还运行 `diagnose.py --tiles 32`，现在按符号分别报告两个模板 kernel 的资源和静态 opcode，包含 `v_cmp*`、`v_cndmask*`，避免只展示其中一条路径。汇编 JSON 的每个文件条目改为 kernel 摘要列表。

用户已反馈关闭/启用两组各 **10 项测试全部通过**。专用路径 VGPR 从 128 降至 124，SGPR 从 46 降至 34，占用率维持 2 waves/SIMD，LDS 16 KiB，无 scratch/spill。静态 `v_cndmask` 从 18 条降至 2 条，K/V 与 MMAC 指令数不变。实测（μs）：

| N=M | 通用路径 | 完整 KV 路径 | Flash SDPA（专用路径运行时） | 加速比 |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 75.06 | 68.12 | 68.84 | 1.011x |
| 2048 | 246.64 | 221.41 | 192.97 | 0.872x |
| 4096 | 797.15 | 712.71 | 659.14 | 0.925x |

N=4096 耗时减少约 10.6%，与 SDPA 的差距缩至约 8.1%。N=1024 这一轮接近 SDPA，不据 1% 的差值宣称稳定领先。后续实验使用 Q128/C32、矩阵 V、完整 KV 专用路径作为对照。

## 第十轮：最大值未变时跳过重缩放

在线 softmax 的最大值没有增加时，`exp2(old_max-new_max)=1`。原路径仍计算这个指数，并给输出的每个 FP32 累加器乘以 1。新增 `FA_BW_CONDITIONAL_RESCALE=1` 候选，仅在 `new_max > old_max` 时计算 correction 并重缩放输出；否则只给分母加上本块 `tile_sum`。最大值变化时保留原来的 `denominator * correction + tile_sum` 表达式，P 和输出的精度不变。

所有 shuffle 和 MMAC 均在该分支之外。分支可以在同一个 wave 内按行分歧，可能引入调度成本，不能预先保证提速。默认 `FA_BW_CONDITIONAL_RESCALE=0` 保留每块重缩放。

4097-key 测试改用正负交替的 query 斜率：同一 wave 内一半行每块更新最大值，另一半行只在第一块更新；倒序 K/V 后交换角色。该测试对照独立 FP32 reference，检查分歧分支、多个 query part 和尾块处理。

同步整个更新目录到 BW1000，运行：

```bash
bash compare_rescale.sh --lengths 1024 2048 4096
```

两组均固定 Q128/C32、矩阵 V、完整 KV 专用路径、原生 exp2 和 QK 前预取。每组通过全部测试后计时，打印资源信息并保存 `results_v10_rescale0.json` / `results_v10_rescale1.json`；`conditional_rescale` 字段读取实际编译库。

用户反馈两组各 10 项测试全部通过，16,384 项标签全部匹配。实测（μs）：

| N=M | 每块重缩放（0） | 条件重缩放（1） | Flash SDPA（对照组） |
| ---: | ---: | ---: | ---: |
| 1024 | 68.84 | 72.69 | 68.78 |
| 2048 | 222.75 | 227.75 | 192.79 |
| 4096 | 716.95 | 716.70 | 659.40 |

条件重缩放没有可确认的收益：N=4096 差值仅 0.25 μs，N=1024/2048 分别慢约 5.6%/2.2%。完整 KV kernel 的 VGPR 从 123 增至 126；两组都是 2 waves/SIMD、16 KiB LDS、无 scratch/spill，因此不能归因于占用率下降。保留 `FA_BW_CONDITIONAL_RESCALE=0`。

## 第十一轮：在矩阵 V 与完整 KV 路径上重新比较分块

此前分块比较使用转置 V 和通用 KV 路径；矩阵 V 与完整 KV 专用路径已改变寄存器用量，旧的分块排名不能直接套用。新增 `compare_tuned_tiles.sh`，统一比较 Q64/128/256 × C32/64 六组。固定矩阵 V、完整 KV、原生 exp2、QK 前预取，关闭条件重缩放。这轮仅增加比较脚本，kernel 与默认配置保持不变。

只需将新脚本复制到 BW1000 的现有目录，然后运行：

```bash
bash compare_tuned_tiles.sh
```

默认覆盖 N=M=256/512/1024/2048/4096，补齐短序列的证据。可传 `--lengths`、`--batch`、`--heads` 等 benchmark 参数；脚本固定 Flash 后端并管理 JSON 路径。每组先运行全部正确性测试，通过后才计时；初次编译会打印资源报告，尤其关注 Q256/C64 是否仍有 spill。

每次运行创建独立的 `build/tuned_tiles/run-XXXXXX/`，保存六个 `results_v11_q{rows}_c{keys}.json`。全部成功后自动打印同一形状下的六组耗时，以及当次最低耗时配置对应的 SDPA 耗时和加速比。脚本只汇总本次文件；任何测试或 benchmark 失败即停止。直接粘贴编译资源、计时和最终汇总即可，无需下载报告。接近 1% 的差异需要复测，不根据单轮 B1/H16 结果推导通用分派策略。

六组 BW1000 实测已返回：每组 10 项测试全部通过，标签各 16,384/16,384 匹配。B1/H16、N=M、Graph 计时如下（μs）：

| N | Q64/C32 | Q64/C64 | Q128/C32 | Q128/C64 | Q256/C32 | Q256/C64 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 11.84 | **10.71** | 16.73 | 15.02 | 28.56 | 25.31 |
| 512 | 25.14 | **23.05** | 26.56 | 23.41 | 45.46 | 38.91 |
| 1024 | 73.24 | 76.99 | 69.08 | **61.56** | 80.71 | 67.66 |
| 2048 | 228.81 | 245.47 | 222.93 | 214.76 | 228.08 | **201.27** |
| 4096 | 790.18 | 864.89 | 717.19 | **704.09** | 798.27 | 732.97 |

完整 KV kernel 的编译资源如下；六组均无 scratch/spill，通用尾块 kernel 也均无 scratch/spill：

| 配置 | VGPR | waves/SIMD | LDS KiB |
| --- | ---: | ---: | ---: |
| Q64/C32 | 82 | 3 | 16 |
| Q64/C64 | 108 | 2 | 32 |
| Q128/C32 | 123 | 2 | 16 |
| Q128/C64 | 153 | 1 | 32 |
| Q256/C32 | 196 | 1 | 16 |
| Q256/C64 | 246 | 1 | 32 |

矩阵 V 和完整 KV 路径使 Q256/C64 消除了此前的 spill，但它仍不是所有规模最快的配置。Q128/C64 虽然占用率低于 Q128/C32，却在本轮所有规模更快；不能只根据占用率选择分块。

各规模当次最快配置与其同组 Flash SDPA：

| N | 配置 | 本 kernel μs | Flash SDPA μs | 加速比 |
| ---: | --- | ---: | ---: | ---: |
| 256 | Q64/C64 | 10.71 | 23.22 | 2.167x |
| 512 | Q64/C64 | 23.05 | 31.42 | 1.363x |
| 1024 | Q128/C64 | 61.56 | 68.89 | 1.119x |
| 2048 | Q256/C64 | 201.27 | 193.65 | 0.962x |
| 4096 | Q128/C64 | 704.09 | 659.43 | 0.937x |

前三个规模已快于 Flash SDPA，2048/4096 的最佳配置仍分别多耗时约 3.9%/6.8%。这些是按规模手动选块的成绩，不是同一默认配置的成绩。选择 **Q128/C64** 作为固定默认，是因为它在本轮所有规模优于此前 Q128/C32 对照，并在 1024/4096 最快；2048 默认耗时为 214.76 μs。当前修改只更新四个默认宏，没有改变 kernel 运算、布局、同步或 Python 接口。

清除旧 `FA_BW_CXXFLAGS` 后可使用默认配置；已测的特定规模也可手动选择 Q64/C64（256/512）或 Q256/C64（2048）：

```bash
FA_BW_CXXFLAGS="-DFA_BW_Q_TILE=64" python bench.py --backend flash --lengths 256 512
FA_BW_CXXFLAGS="-DFA_BW_Q_TILE=256" python bench.py --backend flash --lengths 2048
```

以上选块依据限定在 B1/H16 和列出的自注意力规模，不添加通用形状分派。其他 batch/head/交叉注意力可用 `compare_tuned_tiles.sh` 实测选择。

### 采集当前默认配置的剩余开销

此前 PMC 使用 Q128/C32、未开启完整 KV 路径。新增 `tuned` 模式固定 Q128/C64、矩阵 V、完整 KV，与强制 Flash SDPA 比较：

```bash
bash profile_hipprof.sh all 4096 tuned
```

该模式只采集当前 kernel 和 SDPA 两组；测试通过后保存两条 KV 模板的汇编摘要，再采集 trace（各 25 次调用）和 PMC（各 3 次调用），最后自动打印精简报告。复制更新目录后运行，直接粘贴终端输出即可。旧 `prefetch` / `matrix` 模式显式关闭完整 KV，保留原来的对照配置；省略第三参数仍使用旧 `prefetch` 模式。

当前配置的 GPU 正确性及 benchmark 来自上述用户实测；`tuned` trace 后续已采集成功（见下文），PMC 仍待采集。PMC 指令计数不直接等于时间或利用率，后续优化以新报告为依据。

### 处理 hipprof 没有采集到数据的情况

用户反馈 `tuned` 首次采集失败：10 项正确性测试通过，编译仍为 153 VGPR、32 KiB LDS、无 scratch/spill；但 custom 和 SDPA 的 trace 都提示 `has not valid trace data`，PMC 都提示 `merge pmc data failed`。没有任何 workload 参数或 `Completed ... calls` 输出，只有空 trace DB 和日志。该次运行没有可用的设备时间或 PMC 计数，不能据此诊断性能瓶颈。两边均失败，但仅凭这些信息还不能判断是应用启动、profiler 注入还是导出阶段的问题。

旧脚本只依赖 hipprof 的退出码；本次 hipprof 打印错误后仍返回成功，导致脚本继续采集并打包无效报告。已增加应用日志及采集结果检查：

- 使用与外部测试相同的 Python 绝对路径；每组先不带 profiler 运行一次相同 workload。
- 新增 `profile_capture.py`，在同一进程中运行原来的 `profile_workload.py`，在导入 torch 前记录启动信息，将 Python/native stdout/stderr 直接写入独立日志，并记录退出状态；不新增 GPU 操作。
- 各阶段保存 `preflight/trace/pmc.workload.log` 和对应 `.workload.status`。应用异常可看到 traceback，异常退出时缺少 status 也视为失败。
- 检查应用确实完成预期调用数、hipprof 没有报告已知错误、kernel CSV 有数据行。失败立即停止，保留并打印日志目录，不再输出成功归档提示。

只需更新 `profile_hipprof.sh` 和新增的 `profile_capture.py`，先运行 trace：

```bash
bash profile_hipprof.sh trace 4096 tuned
```

这一步会自动打印应用自身的日志。直接粘贴终端输出即可；若 preflight 都失败，先排查 workload，若它通过而 profiler 下失败，再根据捕获到的异常排查。trace 成功后可使用 `bash profile_hipprof.sh pmc 4096 tuned` 单独采集 PMC。

本地 `python -m unittest test_hipprof -v` 的 8 项回归测试覆盖 hipprof 返回 0 时应用异常、异常退出、未启动应用、两份日志中的导出报错、缺失导出、API CSV 被误当成 PMC 数据，以及成功采集的日志/归档行为。外部 GPU 和 hipprof 用受控替身模拟，shell 和捕获代码实际执行；这不代表 BW1000 上的采集故障已解决。

后续目标日志在 **preflight** 就停止：捕获入口启动、退出状态为 0，但没有 workload 参数和 `Completed 1 calls`。此时尚未调用 hipprof。完整本地 `profile_workload.py` 通过同一捕获入口运行 `--help` 能进入 `main()`；空文件或缺少入口调用的文件可复现目标现象，但目标文件内容尚未核对。先重新同步完整 `profile_workload.py`（当前 3387 字节），确认末尾保留 `if __name__ == "__main__": main()`，再重跑上述 trace 命令。不要根据该日志修改 kernel 或 profiler 配置。

本次重跑 `run-t8YSjD` 已恢复：10 项正确性测试通过，custom/SDPA 的 preflight 各完成 1 次调用，trace 各完成 25 次调用，状态和 kernel CSV 检查均通过。Q128/C64、矩阵 V、完整 KV 的 HIPOPS 设备平均耗时为 **708.583 μs**，Flash SDPA 为 **661.927 μs**，自定义 kernel 多耗时约 **7.05%**（0.9342x）。这是包含前 5 次 warmup 的 trace 均值，与此前 Graph 基准的差距相符；HIP API 表中的分配、launch 等 CPU 耗时不用于计算此加速比。原目标文件的具体缺失尚未核对，不将空文件或缺少入口认定为已证实的原因。

无需再改文件或重复 trace，现在只需采集当前配置的 PMC：

```bash
bash profile_hipprof.sh pmc 4096 tuned
```

脚本分别运行 custom 和 Flash SDPA，各 3 次被采集调用，最后自动打印精简 CSV。现有 trace 只能确认设备时间差距，不能单独确定其指令或访存瓶颈。

## 第十二轮：V 从全局内存直接异步进入 LDS

当前配置 PMC 已在 `run-cFbzCO` 成功采集，两组各 3 次调用的指令计数一致：

| 指标 | Q128/C64 矩阵 V / 完整 KV | Flash SDPA |
| --- | ---: | ---: |
| SQ_INSTS_LDS | 4,571,136 | 1,581,056 |
| SQ_INSTS_VALU | 43,280,384 | 30,524,416 |
| SQ_INSTS_VMEM_RD | 540,672 | 401,408 |
| SQ_INSTS_VMEM_WR | 16,384 | 69,632 |
| SQ_LDS_BANK_CONFLICT | 0 | 0 |
| scratch | 0 | 0 |

LDS 和 VALU 指令分别约为 SDPA 的 2.89/1.42 倍。没有 bank conflict 或 scratch 的证据，下一项尝试减少 V 搬运所需指令和中间寄存器。`SQ_WAIT_INST_LDS` 分别约 292.8–293.7 万与 130.1 万，但不将这个计数直接当成 kernel 时间占比。PMC 的分配资源字段为 VGPR156/SGPR48；编译器报告 VGPR153/SGPR35，两种口径的差异不证明代码发生了资源回退。

参考仓库 `src/tl_templates/hcu/copy.h` 和 `core/arch/amd_buffer_addressing.hpp` 明确在 gfx936 路径中使用 `buffer_load_dwordx4 ... lds`，直接把每 lane 的 16 字节从全局内存写入 LDS。新增默认关闭的 `FA_BW_ASYNC_V=1` 候选只替换完整 KV kernel 中的 V 预取；要求矩阵 V、完整 KV 专用路径和 QK 前预取同时开启。

硬件将每个 lane 的数据写入 wave LDS 基址加 `lane*16` 字节。为保持原来的 V 物理布局，对全局源列应用逆 swizzle：线程 `tid`、块内 `chunk` 读取 `V[tid/8 + chunk*32, (tid%8)*8 ^ ((row&1)*32)]` 起始的 8 个 half。该 XOR 是自身的逆，因此 LDS 中的最终排列与原路径一致，P 和 PV fragment 不变。

下一块 V 在 QK 前直接发往另一 LDS buffer，K 继续通过原来的寄存器预取和写入。每轮末尾等待全局/LDS 操作完成，然后执行原有 block barrier；上一轮 barrier 保证另一 buffer 已读完，下一轮才会读取新数据。首块同样等待并同步。M 有尾块时整个调用沿用原来的通用 kernel，不依赖硬件越界填零行为。QK、softmax、P 精度和输出累加逻辑保持不变。

新增 CPU 检查验证每个源向量与线性 LDS 目标位置一一对应。设备探针使用生产 helper，分别校验 C32 的 8,192 项与 C64 的 16,384 项标签，且全局输入带 16 字节偏移；长期 softmax 测试同时覆盖 4096-key 完整块流水线和 4097-key 尾块路径。CPU 地址验证不能替代目标指令、同步和数值测试。

同步整个更新目录（包含 `async_v.h` 和更新的测试/loader），运行：

```bash
bash compare_async_v.sh --lengths 1024 2048 4096
```

脚本固定已测的 Q128/C64，比较 `ASYNC_V=0/1`，每组先通过完整测试再与强制 Flash SDPA 计时，打印编译资源并保存 `results_v12_async_v0.json` / `results_v12_async_v1.json`。编译元数据 `async_v` 记录开关；有尾块时即使开关为 1 也使用原路径。当前仅完成本地检查，新指令在目标编译器上的编译、标签、GPU 正确性与性能均待本轮验证，默认配置继续关闭该候选。

首轮反馈：关闭异步 V 的对照组 10 项通过、1 项按预期跳过，N=1024/2048/4096 为 62.29/215.10/704.47 μs。开启组在编译阶段失败：`__builtin_amdgcn_readfirstlane` 返回 `int`，不能隐式放入 `uint32_t` 向量的初始化列表。已在 `async_v.h` 的两个地址分量处显式转换返回值；不关闭窄化检查。用本地 Clang 的真实 AMDGPU builtin 复现后，修正后的 C32/C64 模板通过语法检查；该检查使用通用 gfx90a 目标和最小 HIP 声明，不是 gfx936 DTK 编译或运行结果。

只需更新 `async_v.h`，继续未完成的候选组（其余参数采用当前 Q128/C64 默认）：

```bash
export FA_BW_CXXFLAGS="-DFA_BW_ASYNC_V=1 -Rpass-analysis=kernel-resource-usage"
python -m unittest test_flash_attn -v && python bench.py --backend flash --lengths 1024 2048 4096 --json results_v12_async_v1.json
```

头文件变化会自动触发重编译。修正窄化转换后的目标结果已返回：11 项测试全部通过，async V 的 C32/C64 标签分别为 8192/8192、16384/16384，常规矩阵 V 标签也全部匹配。完整 KV kernel 为 145 VGPR（对照 153）、35 SGPR、32 KiB LDS、1 wave/SIMD，无 scratch/spill。实测（μs）：

| N | 寄存器 V 对照 | 异步 V | Flash SDPA（异步组） |
| ---: | ---: | ---: | ---: |
| 1024 | 62.29 | 60.86 | 68.93 |
| 2048 | 215.10 | 212.07 | 192.96 |
| 4096 | 704.47 | 695.16 | 659.20 |

本轮耗时分别减少约 2.3%/1.4%/1.3%，寄存器减少但占用率未变。长序列仍未超过 SDPA；小幅差异需结合后续同配置复测判断。

该编译还报告 `inline asm clobber list contains reserved registers: m0`。进一步修正 `async_v.h`：在同一个 asm 块内用独立 SGPR 保存旧 m0，设置 LDS 地址并发出 load，然后恢复 m0。输出约束采用 `=&s`，防止保存值与尚待使用的地址/描述符输入重叠；因此无需把保留寄存器列入 clobber，仍保留 `memory`。不是关闭 warning 检查。

本地 Clang 对旧头文件生成 AMDGPU 汇编时，用 `-Werror=inline-asm` 可复现错误；新头文件在 `-O3` 下通过，C32/C64 三个 copy 序列的保存/恢复及不重叠寄存器均已核对。这是通用 gfx90a 汇编生成检查，不是 DTK/gfx936 执行。修改增加少量标量指令，**上述性能属于保存/恢复修正之前的版本**，默认继续关闭异步 V。

只更新 `async_v.h` 后，验证警告已消失并复测：

```bash
export FA_BW_CXXFLAGS="-DFA_BW_ASYNC_V=1 -Rpass-analysis=kernel-resource-usage -Werror=inline-asm"
python -m unittest test_flash_attn -v && python bench.py --backend flash --lengths 1024 2048 4096 --json results_v12_async_v1_m0.json
```

保存/恢复 m0 后的 BW1000 复测已通过：`-Werror=inline-asm` 编译成功且没有 m0 警告，11 项测试全部通过，C32/C64 async 标签与矩阵 V 标签全部匹配。完整 KV 的资源仍为 145 VGPR、35 SGPR、32 KiB LDS、1 wave/SIMD，无 scratch/spill。当前修正版成绩如下（μs）：

| N | 异步 V（保存/恢复 m0） | Flash SDPA | 加速比 |
| ---: | ---: | ---: | ---: |
| 1024 | 60.96 | 68.86 | 1.130x |
| 2048 | 213.00 | 193.57 | 0.909x |
| 4096 | 698.82 | 659.06 | 0.943x |

相对第十二轮原寄存器 V 对照，耗时分别减少约 2.1%/1.0%/0.8%；这些对照不是本次同进程重新测得，小幅差异不能据此宣称稳定领先。N=4096 仍比同组 SDPA 多耗时约 6.0%。异步 V 保留为已经通过目标编译与正确性验证的可选路径，`FA_BW_ASYNC_V` 默认继续为 0；设置为 1 可以使用该实现。此后没有修改内核，因此本表对应当前保存/恢复 m0 的代码。

## 第十三轮：K 直接异步进入 LDS，并复测 Q256

长序列的下一项候选是 `FA_BW_ASYNC_K=1`。它要求同时开启异步 V、矩阵 V、完整 KV 专用路径和 QK 前预取；两个异步开关仍默认关闭。完整块中 K/V 都在 QK 前直接发往另一 LDS buffer，省去 K 的中间寄存器和显式 LDS 写入，共用原来的等待与 block barrier。QK 列置换、P→PV 寄存器复用、softmax 和累加精度保持不变，有 KV 尾块的调用仍走原来的通用路径。

K 的现有 XOR swizzle 保持 4 个 half 连续，因此 `async_k.h` 使用 gfx936 的 `buffer_load_dwordx2 ... lds`，每 lane 搬运 8 字节。源地址应用逆 swizzle，最终 LDS 布局与原路径一致。沿用异步 V 已验证的 m0 保存/恢复方式。相对原来的每线程 16 字节 K 加载，加载指令数会增加；收益取决于减少 LDS 写入与寄存器暂存能否抵消这项开销，不能事先保证提速。

同步整个更新目录，运行：

```bash
bash compare_async_k.sh --lengths 1024 2048 4096
```

脚本比较 Q128/Q256 × ASYNC_K=0/1 四组，C64 和 ASYNC_V=1 固定。每组先通过正确性测试，再与强制 Flash SDPA 做 Graph 计时，打印寄存器/spill 资源，保存 `results_v13_q{128,256}_async_k{0,1}.json`。Q256 是此前 N=2048 最快的分块，本轮同时验证它与异步搬运的组合。直接粘贴四组测试、资源和性能输出即可。

新增目标探针分别核对 C32 的 4096 个、C64 的 8192 个 K 标签和 LDS 两侧 guard，源地址带 16 字节偏移；已有 attention 测试覆盖 4096 完整块、4097 尾块、多 head、部分 query tile 和非默认 stream。本地 7 项 CPU/layout/parser 测试通过，9 项 GPU 测试跳过；移除逆 swizzle 的临时变体会被布局测试拒绝。新头文件通过通用 AMDGPU 汇编生成检查，C32/C64 共 6 条 x2 copy 的 m0 保存寄存器均不与输入 SGPR 重叠；另用最小 HIP 声明检查了 18 组配置的 host 模板语法。这些检查均不能代替 DTK/gfx936 编译和执行，独立代码审查也未发现阻塞问题。

上述为候选实现时的本地检查，目标实测见下文。比较脚本仅验证 C64 的完整 attention 流水线；C32 只有布局与独立 copy 探针，若后续采用 C32，需额外启用该配置运行完整正确性测试。

若直接搬运收益不足，下一步应检查 QK/PV 与 LDS 读取的指令调度，尝试交错下一块 QK 与当前块 PV；这会增加活跃 score/P 寄存器，必须同时检查 spill 和数值结果。现有 PMC 显示 LDS/VALU 指令偏多，尚不足以断言时间都消耗在某一类指令上。形状选块可利用已经测得的短序列优势，但通用自动分派还需要不同 batch/head 的证据。

### 异步 K 的 BW1000 实测结果

四组均通过目标编译和正确性测试：K0 两组各 11 项通过、1 项按预期跳过，K1 两组各 12 项全部通过，K/V 标签及 LDS guard 全部匹配。固定 C64、ASYNC_V=1，耗时如下（μs）：

| N | Q128/K0 | Q128/K1 | Q256/K0 | Q256/K1 |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 61.07 | 60.46 | 66.90 | 67.28 |
| 2048 | 213.06 | 211.08 | 199.79 | 199.33 |
| 4096 | 698.93 | 690.07 | 725.95 | 724.42 |

Q128 的异步 K 将耗时降低约 1.00%/0.93%/1.27%，但 VGPR 从 145 增至 147，SGPR 从 35 增至 44。Q256 的 VGPR 从 239 降至 232，SGPR 从 40 增至 44；N1024 慢约 0.57%，N2048/4096 快约 0.23%/0.21%，不能确认稳定收益。四组均为 32 KiB LDS、1 wave/SIMD、无 scratch/spill，不能用占用率提升解释此次差值。编译器静态指令数基本不变，也不能用它代替动态计数或计时。

本轮最佳 N2048 为 Q256/K1 的 199.33 μs，同组 Flash 为 193.38 μs，仍多耗时约 3.08%；N4096 为 Q128/K1 的 690.07 μs，同组 Flash 为 658.94 μs，仍多耗时约 4.72%。异步 K 保留为目标编译与正确性已验证的可选路径，异步 K/V 默认值继续为 0。本次数据验证的是 C64 集成流水线，C32 仍只验证了独立 copy。

## 第十四轮：将 softmax 分母归约推迟到输出前

在扩大流水线之前，先去掉每轮重复的分母跨 lane 求和。`FA_BW_DEFER_DENOM=1` 让每个 lane 保留自己的 FP32 分母。一个 query row 的四个 lane 仍先归约最大值，因而每轮共享同一个重缩放系数 `alpha`：

```text
每 lane: d[lane] = alpha * d[lane] + 本 lane 的 FP32 probability 之和
输出前:  denominator = sum(d[0], d[1], d[2], d[3])
```

对各 lane 求和与共同的重缩放在实数运算下可交换，所以输出前统一归约得到相同的 online-softmax 分母。FP32 加法次序会变化，仍需保持原有误差门限验证；P 仍转换为 FP16，输出仍以 FP32 累加，QK 列置换、P→PV 复用及 K/V 流水线不变。

每个 query part 的分母归约从每轮两次 shuffle 变为输出前两次。M4096/C64 共 64 轮，因此分母部分由 128 次减到 2 次；最大值的两次 shuffle 仍在每轮执行。这是源代码中的分母归约减少量，不代表整个 kernel 同比例提速。

同步更新目录后运行：

```bash
bash compare_defer_denom.sh --lengths 1024 2048 4096
```

脚本固定 C64、异步 K/V 都为 1，比较 Q128/Q256 × DEFER_DENOM=0/1，先测试再与强制 Flash SDPA 计时，保存 `results_v14_q{128,256}_defer{0,1}.json`。新开关默认关闭；本轮实现同时作用于完整 KV 和通用尾块，后续根据 spill 结果将优化限定为完整 KV（见下文）。

新增 CPU 模型检查不均匀 lane 权重，并增加目标数值测试：各 key group 的概率质量不同、两 head 的主导 group 轮换，包含正/负/零 query、M64/128/4096/4097 和 N129。它能检出漏归约、错误的 lane 归属，以及把单个 lane 分母乘 4 的错误。已有尺度、softmax 稳定性、长序列最大值更新、尾块与 stream 测试保留。

本地 8 项 CPU/layout/parser 检查通过，10 项 GPU 测试跳过；临时把 CPU 模型最终求和改为 lane0×4 时测试失败。另以最小 HIP 声明完成 24 组 host 模板语法检查，覆盖两种分母和重缩放模式；这不是 DTK/gfx936 编译或运行。

### 延后分母归约的 BW1000 实测

四组各 14 项测试全部通过。固定异步 K/V，耗时如下（μs）：

| N | Q128/归约每轮 | Q128/归约最后 | Q256/归约每轮 | Q256/归约最后 |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 60.62 | 59.92 | 67.13 | 66.15 |
| 2048 | 211.30 | 206.57 | 199.32 | 195.97 |
| 4096 | 690.20 | 673.86 | 724.41 | 712.19 |

Q128 的耗时降低约 1.15%/2.24%/2.37%，Q256 降低约 1.46%/1.68%/1.69%。本轮最佳 N2048 为 Q256 的 195.97 μs，同组 Flash 为 192.92 μs，仍多耗时约 1.58%；N4096 为 Q128 的 673.86 μs，同组 Flash 为 660.11 μs，仍多耗时约 2.08%。N1024 的 Q128 为 59.92 μs，对应 Flash 69.48 μs。

Q128 完整 KV 的 VGPR 从 147 增至 148，尾块从 160 增至 171，均无 spill。Q256 完整 KV 从 232 增至 256，仍无 spill；尾块从 247 增至 256，并出现 10 个 VGPR spill、44 bytes/lane scratch。两种 Q 的 LDS 均为 32 KiB，occupancy 均为 1 wave/SIMD。完整块计时的收益不能掩盖尾块的资源回退，因此当前代码将延后分母归约限定到 `FullKV` 模板，尾块恢复每轮归约。元数据用 `defer_denom_scope: full_kv` 标明；完整 C64 路径仍执行本轮已测的算法。

## 第十五轮：按用户建议比较 K128 / C128

新增 `FA_BW_KV_TILE=128`，每轮处理 128 个 key，QK score 从 4 个增至 8 个 16-key fragment。K/V 双缓冲总计 64 KiB LDS，query/output 暂存继续复用这块空间。保持原来的 K swizzle、QK 列置换与 P→PV 直接复用；异步 K 的 x2 copy 增至每线程 8 次，异步 V 的 x4 copy 增至 4 次，每个描述符覆盖 16 KiB tile。C128 要求 `FA_BW_MATRIX_V=1`。

相同 M 下，KV 循环、最大值归约和输出重缩放的次数减半；同时每轮 score/P 寄存器和 LDS 占用增加，效果需要实测。根据用户最新选择先测试 C128，不增加 Q192。

同步整个更新目录后，在 BW1000 运行：

```bash
bash compare_k128.sh --lengths 1024 2048 4096
```

脚本比较 Q64/Q128 × C64/C128 四组，固定异步 K/V、延后分母归约为 1，先测试再与强制 Flash SDPA 计时。打印资源用量并保存 `results_v15_q{64,128}_c{64,128}.json`。benchmark 另记录设备 CU 数和每 block 共享内存限制（若 PyTorch 提供这些属性），方便判断分块与硬件容量的关系。默认配置仍是 C64，实验开关仍默认关闭。

新增验证覆盖 128-key 的 CPU 地址双射、矩阵 V bank 分组和 P 复用；GPU 探针扩展到 C128 的 16,384 个 K 标签与 guard、32,768 个 V 标签。V 探针使用完整 64 KiB LDS，将 tile 放在最高的 16 KiB，并检查前 48 KiB guard，验证最后一个 V buffer 的地址范围。矩阵 V 探针随当前 C 分块执行；attention 用例补齐 M127/128/129/256、C128 的零/负 scale 和非默认 stream。输入长度上限预留 128-key 步进空间，避免极大 M 的 int32 末轮溢出。

本地 8 项 CPU/layout/parser 检查通过，10 项 GPU 测试跳过；36 组最小声明 host 模板语法检查通过。C128 helper 的通用 AMDGPU 汇编生成包含 8 条 K x2、4 条 V x4 copy，m0 均保存/恢复且没有与输入 SGPR 重叠。独立审查已完成并修正长度边界问题。这些不能代替 DTK/gfx936 的编译与执行；后续目标结果如下。

### C128 的 BW1000 实测：正确，但性能回退

四组各 14 项测试全部通过，包括 C128 的拷贝标签、矩阵 fragment、最高 16 KiB V buffer 和前 48 KiB guard。64 KiB LDS 分配与启动成功，四组完整/尾块 kernel 均无 scratch/spill。设备报告 80 个 CU；`shared_memory_per_block: null` 表示 PyTorch 未提供该属性，不能由它推断硬件上限。

| N | Q64/C64 | Q64/C128 | Q128/C64 | Q128/C128 |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 73.73 | 94.10 | 59.24 | 69.80 |
| 2048 | 233.51 | 296.35 | 206.23 | 246.59 |
| 4096 | 818.35 | 1041.23 | 674.03 | 808.48 |

单位为 μs。C128 相对同 Q 的 C64，在 Q64 下慢约 27%，在 Q128 下慢约 18%～20%。完整 KV 资源如下：

| 配置 | VGPR | SGPR | LDS KiB | waves/SIMD |
| --- | ---: | ---: | ---: | ---: |
| Q64/C64 | 102 | 44 | 32 | 2 |
| Q64/C128 | 124 | 44 | 64 | 1 |
| Q128/C64 | 148 | 44 | 32 | 1 |
| Q128/C128 | 210 | 44 | 64 | 1 |

Q64 的占用率下降，但 Q128 两组都是 1 wave/SIMD，因此不能仅用占用率解释所有回退。C128 减少循环次数的好处没有抵消更大的每轮状态与资源需求；这些结果不单独证明具体指令或访存瓶颈。Q128/C64 尾块恢复为 160 VGPR、无 spill，符合恢复原分母算法后的预期。

保留 C128 作为已验证的可选配置，后续优化继续以 C64 为基础。当前 Q128/C64 的 N1024/2048/4096 分别为 59.24/206.23/674.03 μs，对应 Flash 为 68.86/192.95/659.22 μs。N4096 仍多耗时约 2.25%；N2048 之前测得 Q256/C64 更快，该配置没有包含在本轮四组中。

### 采集当前较快版本的剩余开销

此前 `tuned` PMC 使用异步 K/V 和延后分母归约都未启用的版本。新增 `optimized` 模式固定 Q128/C64、矩阵 V、完整 KV、异步 K/V、延后分母归约，关闭条件重缩放；它对应本轮 674.03 μs 的配置。旧模式显式关闭新增开关，避免默认值变化影响历史对照。

更新 `profile_hipprof.sh` 后运行：

```bash
bash profile_hipprof.sh all 4096 optimized
```

脚本先测试并保存当前配置的汇编摘要，再分别采集 custom 和强制 Flash SDPA 的 trace 与 PMC，最后自动打印精简报告。直接粘贴终端输出，无需下载归档。若只需重采计数器，可用 `pmc` 替换 `all`。kernel 和默认配置在此结果分析步骤中没有改变。

本地 9 项采集回归测试通过，覆盖应用/导出失败，以及 preflight、trace、PMC 确实收到 `optimized` 和 `tuned` 各自的构建参数；测试使用受控 workload/profiler 替身，不代表新的目标 PMC 已采集成功。

### 当前优化配置的 PMC 结果

`run-Eshmnk` 采集成功，配置确认为 Q128/C64、异步 K/V 和延后分母归约。14 项目标测试通过，完整 kernel 为 148 VGPR、44 SGPR、32 KiB LDS、1 wave/SIMD，无 scratch/spill。trace 两组各 25 次调用（含 5 次 warmup），设备均值为 custom **680.570 μs**、Flash **659.482 μs**，多耗时约 3.20%；不要把 HIP launch 的 CPU 时间当成 kernel 时间。

两组各 3 次 PMC dispatch，除等待计数外，下列指令计数在组内相同：

| 指标 | 当前 custom | Flash SDPA |
| --- | ---: | ---: |
| SQ_INSTS_LDS | 3,268,608 | 1,581,056 |
| SQ_INSTS_VALU | 41,590,784 | 30,524,416 |
| SQ_INSTS_VMEM_RD | 802,816 | 401,408 |
| SQ_INSTS_VMEM_WR | 16,384 | 69,632 |
| SQ_WAIT_INST_LDS | 622,422～631,004 | 1,298,526～1,307,272 |
| SQ_LDS_BANK_CONFLICT / scratch | 0 / 0 | 0 / 0 |

相比此前无异步搬运和延后归约的 PMC，custom 的 LDS 指令从 457 万降至 327 万，LDS 等待计数从约 293 万降至约 63 万；VMEM 读指令增加，符合 K 由 16B 寄存器加载改为 8B 直接 LDS 加载的方向。当前 VALU 仍约为 SDPA 的 1.36 倍。这些计数不是字节数或耗时占比，也不能单独证明性能由某一种指令决定。

PMC 的 `grd` 是工作项数：custom 为 131072、SDPA 为 65536，除以每 block 的 256 线程后分别是 512/256 个 block。因此总搬运计数的差异也包含 query 分块带来的重复读取差异。下一轮同时保留 Q128 和 Q256 对照，检查计算调度与寄存器压力。

## 第十六轮：每 lane 内的四路 softmax 归约

新增默认关闭的 `FA_BW_REDUCE4=1`，仅作用于完整 KV kernel。原来一个 lane 顺序归约它持有的全部分数/概率；现在每个寄存器列 `r=0..3` 独立跨 16-key fragment 累积，再用二叉树合并四项。C64 下，16 个输入形成四条长度为 4 的链，首项直接赋值，之后 3 步更新加最终 2 层合并。该结构旨在缩短源代码的依赖链，实际指令与调度由编译器决定。

最大值、概率输入与每个元素的权重保持不变；FP32 求和次序改变，仍使用原有误差门限。P 仍为 FP16 并直接供给 PV，输出和分母仍以 FP32 累积，K/V 搬运、LDS 布局、block 同步不变。四路状态可能增加寄存器压力，所以不能据依赖链长度预先保证提速。尾块保留原算法，所有默认开关保持原值。

同步更新目录后运行：

```bash
bash compare_reduce4.sh --lengths 1024 2048 4096
```

脚本比较 Q128/Q256 × REDUCE4=0/1 四组，固定 C64、异步 K/V 和延后分母归约，先通过全部测试再与强制 Flash SDPA 计时，保存 `results_v16_q{128,256}_reduce4_{0,1}.json`。重点观察 Q256 是否出现 spill，以及四路归约是否改善实际时间。`optimized` profiler 模式仍使用 REDUCE4 默认关闭的已测配置。

本地新增 CPU 模型测试覆盖各寄存器概率不同、正负 query、最大值逐块变化、C32/C64/C128 和短/长/尾块；移除第四路概率和会使测试失败。9 项 CPU/layout/parser 检查通过，10 项 GPU 测试跳过，36 组 host 模板语法检查通过；独立代码审查未发现阻塞问题。现有目标数值/标签/stream 测试保留。新的 DTK 编译资源、GPU 正确性与性能待本轮反馈。

### 四路归约的 BW1000 实测：不采用

四组各 15 项测试全部通过，但 REDUCE4 的实际耗时均增加。固定 B1/H16、C64、异步 K/V 和延后分母归约，Graph 时间如下（μs）：

| N | Q128/R0 | Q128/R1 | Q256/R0 | Q256/R1 |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 59.26 | 59.88 | 66.14 | 67.46 |
| 2048 | 206.30 | 207.77 | 195.97 | 199.42 |
| 4096 | 673.77 | 679.20 | 712.23 | 724.14 |

Q128 慢约 0.7%～1.0%，Q256 慢约 1.7%～2.0%。Q128 完整 kernel 从 148 降至 147 VGPR，静态指令从 1256 降至 1252，仍未提速。Q256 从 256 变为 255 VGPR，同时出现 2 个 VGPR spill、12 bytes/lane scratch，静态指令从 2140 增至 2151。两种 Q 的 LDS 都是 32 KiB，占用率都是 1 wave/SIMD；尾块仍为 160/247 VGPR、无 spill。不能把 Q128 的回退也归因于 spill，或用更少的 VGPR/静态指令替代计时结论。

保留 `FA_BW_REDUCE4=0`。本轮 N1024 的 Q128 为 59.26 μs，对应 Flash 68.82 μs，已快约 1.161x；N2048 的 Q256 为 195.97 μs，对应 Flash 192.91 μs，仍多耗时约 1.59%；N4096 的 Q128 为 673.77 μs，对应 Flash 659.37 μs，仍多耗时约 2.18%。这是按形状手动选择 Q 分块的成绩，并非默认配置在所有形状都领先。

## 第十七轮：Q192 / C64

增加 `FA_BW_Q_TILE=192`：每个 wave 保存三组 16 行 query，四个 wave 共 192 行。沿用现有按 64 行暂存 Q/O 的循环，LDS 仍为 32 KiB；QK 列置换、P→PV 复用和 softmax 计算不变。默认 Q128 保持不变。

当前 PMC 显示 custom 的总搬运指令多于 Flash，query 分块重复读 K/V 是其中一个因素。Q192 相对 Q128 减少 block 数，又比已达 256 VGPR 的 Q256 少保存一组 query 状态，实际 VGPR 与性能需目标编译和计时确认。B1/H16/N4096 时，block 数从 512 降至 352；但 Q192 最后一个 block 有无效 query，总计算行数从 4096 增至 4224（多 3.125%），因此不能只按搬运减少量预测加速。

同步更新目录后，在 BW1000 运行：

```bash
bash compare_q192.sh --lengths 1024 2048 4096
```

脚本比较 Q128/Q192/Q256，固定 C64、异步 K/V、延后分母归约，显式关闭 REDUCE4。每组先跑正确性测试，再与强制 Flash SDPA 做 Graph 计时，打印完整/尾块 kernel 的寄存器和 spill，保存 `results_v17_q{128,192,256}.json`。直接粘贴终端输出即可。

输出 guard 测试新增 N191/192/193、383/384/385，覆盖第三组 query、跨 block 和最终不足一个 tile 的情况；同时检查 M128 完整 KV 与 M97 尾块，输出预填 NaN 以检出漏写。DTK 编译、GPU 正确性和 Q192 性能仍待目标测试。

本地 9 项 CPU/layout/parser 检查通过，10 项 GPU 测试因缺少目标设备跳过。36 组 host 模板语法检查通过，覆盖 Q64/128/192/256、C32/64/128 以及原路径和优化开关组合；比较脚本通过 shell 语法检查。这些检查不代表 gfx936 编译或运行成功。

### Q192 的 BW1000 实测：不采用

三组各 15 项测试全部通过。Q192 完整 KV kernel 为 189 VGPR、48 SGPR，尾块为 208 VGPR、72 SGPR，均无 scratch/spill。LDS 仍为 32 KiB，占用率仍为 1 wave/SIMD。Graph 时间如下（μs）：

| N | Q128 | Q192 | Q256 |
| ---: | ---: | ---: | ---: |
| 1024 | 59.30 | 82.92 | 66.05 |
| 2048 | 206.31 | 224.14 | 195.99 |
| 4096 | 673.90 | 690.30 | 712.14 |

Q192 未超过原有最佳分块。即使不发生 spill、减少 K/V 重复读取，也不能由这两个指标推断整体提速；填充行、指令调度和并行分配的开销需要计时检验。继续以 Q128/C64 和 Q256/C64 优化。当前最佳 N2048 为 195.99 μs，对应 Flash 193.41 μs，仍多耗时约 1.33%；N4096 为 673.90 μs，对应 Flash 659.10 μs，仍多耗时约 2.25%。

## 第十八轮：按整个 wave 判断是否需要重缩放

新增默认关闭的 `FA_BW_WAVE_RESCALE=1`，仅作用于完整 KV kernel，与旧的逐行条件开关 `FA_BW_CONDITIONAL_RESCALE` 互斥。每个 query part 完成最大值归约后，由 64 个 lane 一起投票：

```text
如果 wave 中任意一行的最大值增加：整 wave 执行原来的 correction、分母和输出重缩放
如果所有行的最大值都未变：      只累加本轮分母，输出保留到 PV 累加
```

第二种情况中所有 `correction = exp2(old_max - new_max)` 都精确等于 1，所以可以省去这些 exp2 和输出乘法。区别于此前逐行分支，这次一个 wave 的线程走同一分支；任何一行发生更新都不会漏掉重缩放。首次迭代的最大值从负无穷更新，仍执行初始化所需的 correction。最大值/分母归约、P 的 FP16 转换、MMAC 与 K/V 流水线保持原实现。

投票和分支自身也有成本，Q256 的寄存器压力已经很高，是否获益只能实测。尾块继续使用原路径，默认配置保持不变。同步整个目录后运行：

```bash
bash compare_wave_rescale.sh --lengths 1024 2048 4096
```

脚本比较 Q128/Q256 × WAVE_RESCALE=0/1，固定 C64、异步 K/V 和延后分母归约，关闭 REDUCE4 与逐行条件重缩放。每组先验证再与强制 Flash SDPA 计时，打印资源信息，保存 `results_v18_q{128,256}_wave_rescale_{0,1}.json`。

新增测试覆盖所有行保持最大值、所有行更新，以及只有第 15 行更新而 lane0 的行不更新；同时包含多 query part、跨 block、M128/4096 完整块与 M4097 尾块。错误地把 wave 投票替换为只读取 lane0，在 CPU 模型中会触发 8 个数值比较失败。

本地 10 项 CPU/layout/parser 检查通过，11 项 GPU 测试跳过；48 组 host 模板语法检查覆盖新旧重缩放方式、各 Q/C 分块及分母模式，Python/shell 语法检查通过。DTK/gfx936 编译、资源用量、GPU 正确性与性能尚待目标验证。

### wave 重缩放的 BW1000 实测：不采用

四组各 17 项测试全部通过，包括只有一行最大值变化的用例，但新开关明显增加耗时。固定 C64、异步 K/V 与延后分母归约，Graph 时间如下（μs）：

| N | Q128/W0 | Q128/W1 | Q256/W0 | Q256/W1 |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 60.13 | 65.89 | 66.08 | 71.90 |
| 2048 | 206.55 | 231.52 | 195.82 | 209.49 |
| 4096 | 673.94 | 753.10 | 712.25 | 756.45 |

Q128 慢约 9.6%～12.1%，Q256 慢约 6.2%～8.8%。完整 kernel 的 VGPR 分别从 148 降至 143、从 256 降至 247，SGPR 都是 44，均无 scratch/spill；LDS 都是 32 KiB，占用率都是 1 wave/SIMD。静态指令数分别从 1256 增至 1275、从 2140 增至 2180。尾块资源不变。

保留 `FA_BW_WAVE_RESCALE=0`。减少源码中的乘法和实际 VGPR，没有转化为整体提速；本轮没有新的 PMC 或指令调度证据，不能单独认定回退全由投票、分支或某种等待造成。当前 N2048 最佳 195.82 μs，对应 Flash 192.72 μs，仍多耗时约 1.61%；N4096 最佳 673.94 μs，对应 Flash 659.38 μs，仍多耗时约 2.21%。

## 下一步：核对实际生成的指令

现有摘要省略了 max、add、mul、FMA 和分支操作码。新增 `diagnose.py --details --full-only`：打印完整 KV kernel 的所有静态操作码计数、按操作数区分的 `waitcnt` 计数，以及最后一次 buffer load、第一次 MMAC、exp 和矩阵 LDS 读取附近的指令。每处最多前后各 20 条记录，重叠片段合并，并带原 `.s` 文件行号。这些位置是便于阅读的固定锚点，不代表已识别的运行时热点。

同步更新目录后运行：

```bash
bash inspect_codegen.sh
```

脚本编译 Q128/W0、Q128/W1 和 Q256/W0，固定 C64、异步 K/V、延后分母归约和 REDUCE4=0，并自动打印精简报告。全部文件留在目标机器，可直接粘贴终端输出，无需下载。此步骤不修改 kernel 或默认选项。

重点核对实际归约是否已有 max3、乘法和 exp 的依赖关系、LDS 读取与 MMAC 的先后顺序，以及 wave 分支前后的指令。静态计数不等于动态执行次数或耗时，报告本身不会替代基准计时。找不到完整 KV kernel、有效 device assembly 或函数结尾时，详细报告模式返回失败，避免把缺失数据当作零开销。

6 项诊断解析测试通过，覆盖跨函数隔离、注释过滤、等待参数、上下文长度和缺失锚点。另使用受控编译器替身验证实际脚本的三组参数、编译失败即停止，以及只有尾块 kernel 时返回失败；Python/shell 语法检查通过。目标代码生成报告仍待 BW1000 采集。

### 三组实际代码生成结果

三组均编译成功。Q128/W0 为 148 VGPR，Q128/W1 为 143 VGPR，Q256/W0 为 256 VGPR，均无 scratch/spill。Q128/W0 已包含 `v_max3_f32`、`v_fma_f32`、`v_pk_add_f32` 和 `v_pk_mul_f32`，因此再手工实现 max3 或简单的乘减融合没有直接依据。

Q128/W0 与 W1 的关键静态计数如下：

| 指令 | W0 | W1 |
| --- | ---: | ---: |
| v_fma_f32 | 38 | 6 |
| v_pk_add_f32 | 16 | 0 |
| v_add_f32_e32 | 4 | 38 |
| v_sub_f32_e32 | 2 | 34 |
| s_waitcnt lgkmcnt(0) | 25 | 32 |
| 全部 s_waitcnt | 49 | 49 |

原版片段中 QK MMAC 与 softmax 交错，概率使用 `fma(raw_score, log2_scale, -new_max)`；V 矩阵读取也与输出重缩放交错。wave 分支版片段先执行 correction 和重缩放，首个 V 矩阵读取之后较快进入 `lgkmcnt(0)` 等待。它减少了某些动态乘法，但同时改变融合和调度；这些观察解释了为何不能只按跳过的乘法预测性能，仍不能据此量化各因素的耗时占比。

另一个可直接看到的重复计算是：原版先对每个 raw score 乘 `log2_scale` 来归约最大值，随后又在概率的 FMA 中使用 raw score 和 `log2_scale`。下一轮减少前一部分乘法。

## 第十九轮：将最大值路径的 scale 推迟到归约后

新增默认关闭的 `FA_BW_DELAY_SCALE=1`。仅对完整 KV、正 scale 的调用启用：

```text
s = scale * log2(e), s > 0
tile_max = max(raw_scores) * s
new_max  = max(old_max, tile_max)
P        = exp2(fma(raw_score, s, -new_max))
```

正缩放具有单调性，`max(s*x) = s*max(x)`。C64 每个 query part 的最大值路径从 16 次缩放变为 1 次；Q128 有两个 part，源码中每线程每轮省去 30 次标量乘法，Q256 有四个 part，省去 60 次。概率仍显式使用 FP32 FMA，然后转换为 FP16；输出/分母维持 FP32，P→PV 寄存器布局和 K/V 流水线不变。实际生成的指令数、寄存器分配和提速幅度需目标验证。

开启后，零/负 scale 和所有 KV 尾块调用都分派到原来的 `FullKV=false` 路径；因而这些调用也使用原来的 K/V 搬运和每轮分母归约。正 scale 的条件在主机分派时判断，不增加 KV 内层循环的运行时分支。元数据使用 `delay_scale_scope: full_kv_positive_scale` 标明范围。

同步整个更新目录，在 BW1000 运行：

```bash
bash compare_delay_scale.sh --lengths 1024 2048 4096
```

脚本比较 Q128/Q256 × DELAY_SCALE=0/1，固定 C64、异步 K/V、延后分母归约，关闭 wave/逐行条件重缩放及 REDUCE4。每组先通过正确性测试，再与强制 Flash SDPA 做 Graph 计时，打印编译资源并保存 `results_v19_q{128,256}_delay_scale_{0,1}.json`。

测试增加正负 raw logits、0.01/0.125/0.7/2.0 等正 scale、共同 logit 偏移、完整/尾块，以及大幅值负 scale 的回退情况。CPU 模型中漏掉最大值缩放的变体触发 24 个数值失败。本地 13 项 CPU/layout/parser 检查通过，12 项 GPU 测试跳过；48 组 host 模板语法与 Python/shell 语法检查通过。这些不是 DTK/gfx936 的编译或执行结果。

### 延后 scale 的 BW1000 实测：保留优化

四组各 19 项测试全部通过。开启延后 scale 后，Q128 与 Q256 在三个长度上都减少耗时：

| N | Q128/S0 μs | Q128/S1 μs | Q256/S0 μs | Q256/S1 μs |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 59.20 | 57.31 | 65.98 | 64.66 |
| 2048 | 206.32 | 198.92 | 195.96 | 188.99 |
| 4096 | 673.93 | 648.08 | 712.11 | 683.71 |

Q128 完整 kernel 的 VGPR 维持 148，静态指令数从 1256 降至 1234；Q256 的 VGPR 维持 256，静态指令数从 2140 降至 2094。SGPR 均为 44，LDS 均为 32 KiB，占用率均为 1 wave/SIMD，无 scratch/spill；尾块仍为 160/247 VGPR，无 spill。本次收益不是占用率提升。

最佳配置为 N1024/4096 使用 Q128，N2048 使用 Q256，对应同组 Flash 的加速比分别为 1.202x/1.017x/1.025x。最大绝对误差分别为 0.000244/0.000122/0.000244，均满足原阈值。相同 Q 下，三个最佳配置相对 S0 的耗时分别降低约 3.19%/3.84%/3.56%。

保留 `DELAY_SCALE=1` 作为推荐优化配置的一部分，与异步 K/V 和延后分母归约一起使用；不启用 REDUCE4、wave/逐行条件重缩放。`bench_best.sh` 固定这些已测组合以便复现，kernel 的默认开关保持原值，避免改变历史对照脚本的含义。本步骤只整理结果与运行入口，没有修改已测 kernel。
