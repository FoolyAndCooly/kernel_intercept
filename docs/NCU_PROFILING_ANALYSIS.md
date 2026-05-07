# NCU Profiling 分析文档 - CharGPT 推理

## 概述

本文档分析了使用 NVIDIA Nsight Compute (NCU) 对 CharGPT 模型推理过程中 CUDA kernel 的性能分析结果。

**模型配置**:
- Embedding Size: 128
- Layers: 1 (最小化配置，覆盖所有算子类型)
- Sequence Length: 64
- Batch Size: 2
- Vocab Size: 65

**Profile 文件**: `profiles/single_client_ncu_minimal.ncu-rep`

---

## NCU 指标说明

### 1. GPU Speed Of Light Throughput (GPU 性能上限分析)

| 指标 | 说明 | 单位 |
|------|------|------|
| **Duration** | Kernel 执行时间 | usecond |
| **Compute (SM) [%]** | SM 计算单元利用率，相对于理论峰值的百分比 | % |
| **Memory [%]** | 内存子系统利用率，相对于理论峰值的百分比 | % |
| **DRAM Throughput** | 全局内存(HBM/GDDR)吞吐量利用率 | % |
| **L1/TEX Cache Throughput** | L1/纹理缓存吞吐量利用率 | % |
| **L2 Cache Throughput** | L2 缓存吞吐量利用率 | % |
| **SM Frequency** | SM 运行频率 | cycle/usecond |
| **DRAM Frequency** | 内存运行频率 | cycle/nsecond |
| **Elapsed Cycles** | 总执行周期数 | cycle |
| **SM Active Cycles** | SM 活跃周期数 | cycle |

**解读**: 
- Compute (SM) % 高 → 计算密集型 kernel
- Memory % 高 → 内存密集型 kernel
- 两者都低 → kernel 可能存在优化空间或规模太小

### 2. Compute Workload Analysis (计算负载分析)

| 指标 | 说明 | 单位 |
|------|------|------|
| **Executed Ipc Active** | 每个活跃周期执行的指令数 | inst/cycle |
| **Executed Ipc Elapsed** | 每个总周期执行的指令数 | inst/cycle |
| **Issued Ipc Active** | 每个活跃周期发射的指令数 | inst/cycle |
| **Issue Slots Busy** | 指令发射槽繁忙百分比 | % |
| **SM Busy** | SM 繁忙百分比 | % |

**解读**:
- IPC 接近 4 表示高效利用 (Volta 架构每 SM 有 4 个 warp scheduler)
- SM Busy 高但 IPC 低 → 可能存在内存等待或指令依赖

### 3. Memory Workload Analysis (内存负载分析)

| 指标 | 说明 | 单位 |
|------|------|------|
| **Memory Throughput** | 实际内存吞吐量 | Gbyte/second |
| **Mem Busy** | 内存单元繁忙百分比 | % |
| **Max Bandwidth** | 达到的最大带宽百分比 | % |
| **L1/TEX Hit Rate** | L1/纹理缓存命中率 | % |
| **L2 Hit Rate** | L2 缓存命中率 | % |
| **Mem Pipes Busy** | 内存管道繁忙百分比 | % |

**解读**:
- 高 L1/L2 Hit Rate → 数据局部性好，减少 DRAM 访问
- 低 Hit Rate + 高 Memory Throughput → 内存带宽受限

### 4. Launch Statistics (启动统计)

| 指标 | 说明 | 单位 |
|------|------|------|
| **Block Size** | 每个 block 的线程数 | thread |
| **Grid Size** | Grid 中的 block 数量 | block |
| **Threads** | 总线程数 | thread |
| **Registers Per Thread** | 每个线程使用的寄存器数 | register/thread |
| **Shared Memory Per Block** | 每个 block 使用的共享内存 | byte/block |
| **Waves Per SM** | 每个 SM 的 wave 数 (Grid Size / SM数量) | - |

**解读**:
- Waves Per SM < 1 → Grid 太小，无法充分利用 GPU
- Registers Per Thread 高 → 可能限制 occupancy

### 5. Occupancy (占用率)

| 指标 | 说明 | 单位 |
|------|------|------|
| **Theoretical Occupancy** | 理论最大占用率 | % |
| **Achieved Occupancy** | 实际达到的占用率 | % |
| **Achieved Active Warps Per SM** | 每个 SM 实际活跃的 warp 数 | warp |
| **Block Limit SM** | SM 数量限制的最大 block 数 | block |
| **Block Limit Registers** | 寄存器限制的最大 block 数 | block |
| **Block Limit Shared Mem** | 共享内存限制的最大 block 数 | block |
| **Block Limit Warps** | Warp 数量限制的最大 block 数 | block |

**解读**:
- Achieved << Theoretical → 存在资源限制或 grid 太小
- 查看 Block Limit 系列指标找出限制因素

### 6. Scheduler Statistics (调度器统计)

| 指标 | 说明 | 单位 |
|------|------|------|
| **Active Warps Per Scheduler** | 每个调度器的活跃 warp 数 | warp |
| **Eligible Warps Per Scheduler** | 每个调度器可调度的 warp 数 | warp |
| **Issued Warp Per Scheduler** | 每个调度器发射的 warp 数 | - |
| **One or More Eligible** | 至少有一个可调度 warp 的周期百分比 | % |
| **No Eligible** | 没有可调度 warp 的周期百分比 | % |

**解读**:
- No Eligible 高 → warp 都在等待 (内存/同步/依赖)
- Eligible 高但 Issued 低 → 调度器瓶颈

### 7. Warp State Statistics (Warp 状态统计)

| 指标 | 说明 | 单位 |
|------|------|------|
| **Warp Cycles Per Executed Instruction** | 每条执行指令的 warp 周期数 | cycle |
| **Warp Cycles Per Issued Instruction** | 每条发射指令的 warp 周期数 | cycle |
| **Avg. Active Threads Per Warp** | 每个 warp 的平均活跃线程数 | thread |
| **Avg. Not Predicated Off Threads Per Warp** | 未被 predicate 关闭的平均线程数 | thread |

**解读**:
- Active Threads < 32 → 存在线程分歧 (divergence)
- Cycles Per Instruction 高 → 指令延迟高

### 8. Source Counters (源码计数器)

| 指标 | 说明 | 单位 |
|------|------|------|
| **Branch Instructions** | 分支指令数量 | inst |
| **Branch Instructions Ratio** | 分支指令占比 | % |
| **Branch Efficiency** | 分支效率 (无分歧的分支百分比) | % |
| **Avg. Divergent Branches** | 平均分歧分支数 | - |

**解读**:
- Branch Efficiency < 100% → 存在 warp 分歧
- Divergent Branches 高 → 控制流不规则

---

## Kernel 性能分析

### 1. elementwise_kernel_with_index (arange)

**用途**: 生成位置索引序列 (torch.arange)

| 指标 | 值 | 分析 |
|------|-----|------|
| Duration | 4.51 us | 非常短 |
| Compute (SM) | 0.01% | 极低 |
| Memory | 0.39% | 极低 |
| Achieved Occupancy | 6.23% | 极低 |
| Grid Size | 1 | 只有1个block |
| Waves Per SM | 0.00 | 无法填满GPU |

**瓶颈分析**: Grid 太小 (只有1个block)，无法利用 GPU 并行性。这是一个非常小的 kernel，主要开销在启动延迟。

---

### 2. indexSelectLargeIndex (Embedding)

**用途**: Embedding 查表操作

| 指标 | 值 | 分析 |
|------|-----|------|
| Duration | 7.78 us | 较短 |
| Compute (SM) | 11.93% | 中等 |
| Memory | 2.99% | 较低 |
| Memory Throughput | 5.50 GB/s | 中等 |
| L2 Hit Rate | 81.98% | 良好 |
| Achieved Occupancy | 39.09% | 中等 |
| Registers Per Thread | 32 | 中等 |

**瓶颈分析**: 计算和内存利用率都不高，主要受限于数据规模较小。L2 命中率高说明 embedding 表大部分在缓存中。

---

### 3. elementwise_kernel (add)

**用途**: 逐元素加法 (位置编码 + token 编码)

| 指标 | 值 | 分析 |
|------|-----|------|
| Duration | 7.04 us | 较短 |
| Compute (SM) | 4.45% | 较低 |
| Memory | 7.51% | 较低 |
| Memory Throughput | 18.86 GB/s | 中等 |
| DRAM Throughput | 7.51% | 较低 |
| L2 Hit Rate | 55.73% | 中等 |
| Achieved Occupancy | 19.73% | 较低 |

**瓶颈分析**: 典型的内存密集型操作，但数据量小导致利用率低。

---

### 4. vectorized_layer_norm_kernel (LayerNorm)

**用途**: 层归一化

| 指标 | 值 | 分析 |
|------|-----|------|
| Duration | 11.46 us | 中等 |
| Compute (SM) | 14.90% | 中等 |
| Memory | 13.07% | 中等 |
| L1/TEX Cache Throughput | 18.96% | 中等 |
| L1/TEX Hit Rate | 75.49% | 良好 |
| Achieved Occupancy | 33.37% | 中等 |
| Registers Per Thread | 38 | 较高 |

**瓶颈分析**: LayerNorm 需要计算均值和方差，涉及 reduction 操作。L1 命中率高说明数据复用良好。寄存器使用较多但未成为限制因素。

---

### 5. volta_sgemm_32x32_sliced1x4_tn (GEMM Q/K/V)

**用途**: 计算 Query/Key/Value 的线性变换

| 指标 | 值 | 分析 |
|------|-----|------|
| Duration | 17.12 us | 较长 |
| Compute (SM) | 2.74% | 较低 |
| Memory | 3.18% | 较低 |
| L1/TEX Cache Throughput | 36.61% | 较高 |
| SM Busy | 26.90% | 中等 |
| Executed IPC | 1.03 | 中等 |
| Theoretical Occupancy | 25% | 受限 |
| Achieved Occupancy | 12.49% | 较低 |
| Registers Per Thread | 86 | 非常高 |
| Grid Size | 4 | 很小 |

**瓶颈分析**: 
- 寄存器使用量极高 (86)，严重限制了理论 occupancy (仅25%)
- Grid 太小 (4 blocks)，无法充分利用 GPU
- 这是 cuBLAS 的 GEMM kernel，针对小矩阵使用了 sliced 策略

---

### 6. volta_sgemm_32x128_tn (GEMM Projection)

**用途**: 注意力输出投影和 FFN 层

| 指标 | 值 | 分析 |
|------|-----|------|
| Duration | 11.17 us | 中等 |
| Compute (SM) | 2.62% | 较低 |
| Memory | 2.62% | 较低 |
| L1/TEX Cache Throughput | 33.82% | 较高 |
| SM Busy | 30.06% | 中等 |
| Executed IPC | 1.18 | 中等 |
| L2 Hit Rate | 85.89% | 良好 |
| Registers Per Thread | 57 | 较高 |
| Grid Size | 4 | 很小 |

**瓶颈分析**: 与 Q/K/V GEMM 类似，受限于小矩阵规模和高寄存器使用。

---

### 7. vectorized_elementwise_kernel (GELU/Dropout)

**用途**: GELU 激活函数、Dropout mask 应用

| 指标 | 值 | 分析 |
|------|-----|------|
| Duration | 5.50 us | 较短 |
| Compute (SM) | 0.34% | 极低 |
| Memory | 3.28% | 较低 |
| Memory Throughput | 7.62 GB/s | 中等 |
| Executed IPC | 0.09 | 极低 |
| Achieved Occupancy | 12.34% | 较低 |
| Grid Size | 8 | 较小 |

**瓶颈分析**: 纯内存密集型操作，计算量极小。Grid 较小导致 GPU 利用率低。

---

### 8. softmax_warp_forward (Softmax)

**用途**: 注意力分数的 Softmax 归一化

| 指标 | 值 | 分析 |
|------|-----|------|
| Duration | 8.61 us | 中等 |
| Compute (SM) | 3.50% | 较低 |
| Memory | 2.64% | 较低 |
| SM Busy | 13.77% | 较低 |
| Executed IPC | 0.54 | 较低 |
| L2 Hit Rate | 62.79% | 中等 |
| Achieved Occupancy | 12.40% | 较低 |
| Grid Size | 16 | 较小 |

**瓶颈分析**: Softmax 需要 reduction 操作 (求 max 和 sum)，存在同步开销。Grid 较小限制了并行度。

---

### 9. volta_sgemm_32x128_nn (GEMM Attention)

**用途**: 注意力权重与 Value 的矩阵乘法

| 指标 | 值 | 分析 |
|------|-----|------|
| Duration | 18.56 us | 较长 |
| Compute (SM) | 2.18% | 较低 |
| Memory | 2.18% | 较低 |
| L1/TEX Cache Throughput | 49.88% | 较高 |
| SM Busy | 48.41% | 中等偏高 |
| Executed IPC | 1.70 | 较好 |
| L2 Hit Rate | 69.06% | 中等 |
| Registers Per Thread | 57 | 较高 |
| Grid Size | 2 | 极小 |

**瓶颈分析**: 
- Grid 极小 (仅2个blocks)，严重限制并行度
- SM Busy 较高但 Compute% 低，说明在等待内存
- IPC 相对较好，说明计算效率不错

---

### 10. CatArrayBatchedCopy (Concat)

**用途**: 多头注意力输出的拼接

| 指标 | 值 | 分析 |
|------|-----|------|
| Duration | 8.13 us | 中等 |
| Compute (SM) | 4.02% | 较低 |
| Memory | 10.21% | 中等 |
| Memory Throughput | 24.04 GB/s | 较好 |
| DRAM Throughput | 10.02% | 中等 |
| L1/TEX Hit Rate | 60% | 中等 |
| Achieved Occupancy | 11.52% | 较低 |

**瓶颈分析**: 纯内存拷贝操作，内存带宽利用相对较好。

---

### 11. splitKreduce_kernel (Split-K Reduce)

**用途**: Split-K GEMM 的归约操作

| 指标 | 值 | 分析 |
|------|-----|------|
| Duration | 9.44 us | 中等 |
| Compute (SM) | 13.04% | 中等 |
| Memory | 21.91% | 较高 |
| Memory Throughput | 51.34 GB/s | 较好 |
| DRAM Throughput | 21.91% | 较高 |
| Achieved Occupancy | 48.72% | 较好 |
| Block Size | 512 | 较大 |
| Grid Size | 32 | 中等 |

**瓶颈分析**: 这是所有 kernel 中内存利用率最高的，occupancy 也最好。主要是内存密集型的归约操作。

---

## 总体分析

### 性能瓶颈总结

1. **Grid 规模过小**: 大多数 kernel 的 Grid Size 都很小 (1-32)，导致无法充分利用 GPU 的并行计算能力。这是因为模型配置较小 (batch=2, seq=64, emb=128)。

2. **GEMM kernel 寄存器压力**: cuBLAS 的 GEMM kernel 使用了大量寄存器 (57-86)，限制了 occupancy。

3. **内存带宽未充分利用**: 大多数 kernel 的 DRAM Throughput 都很低 (<10%)，说明数据量太小，无法发挥 GPU 内存带宽优势。

4. **Achieved vs Theoretical Occupancy 差距大**: 几乎所有 kernel 的实际 occupancy 都远低于理论值，主要原因是 grid 太小。

### 优化建议

1. **增大 Batch Size**: 增加批量大小可以增大 Grid Size，提高 GPU 利用率。

2. **使用 Tensor Core**: 对于 GEMM 操作，使用 FP16/BF16 并启用 Tensor Core 可以大幅提升性能。

3. **Kernel Fusion**: 将多个小 kernel 融合成一个大 kernel，减少启动开销。

4. **使用 Flash Attention**: 对于注意力计算，使用 Flash Attention 可以减少内存访问并提高效率。

---

## 附录: Kernel 执行时间分布

| Kernel 类型 | Duration (us) | 占比 |
|------------|---------------|------|
| volta_sgemm_32x128_nn | 18.56 | 16.2% |
| volta_sgemm_32x32_sliced1x4_tn | 17.12 | 14.9% |
| vectorized_layer_norm_kernel | 11.46 | 10.0% |
| volta_sgemm_32x128_tn | 11.17 | 9.7% |
| splitKreduce_kernel | 9.44 | 8.2% |
| softmax_warp_forward | 8.61 | 7.5% |
| CatArrayBatchedCopy | 8.13 | 7.1% |
| indexSelectLargeIndex | 7.78 | 6.8% |
| elementwise_kernel | 7.04 | 6.1% |
| vectorized_elementwise_kernel | 5.50 | 4.8% |
| elementwise_kernel_with_index | 4.51 | 3.9% |

**结论**: GEMM 操作 (sgemm) 占据了约 40% 的执行时间，是主要的优化目标。
