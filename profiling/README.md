# NCU Profile 数据处理说明

本目录包含 NVIDIA Nsight Compute (NCU) 性能分析数据的处理脚本和输出文件。

---

## 目录结构

```
ncu_profile/
├── README.md                          # 本说明文档
├── single_client_ncu_minimal.ncu-rep  # NCU 原始报告文件
├── single_client_ncu_minimal.csv      # NCU 导出的基础 CSV
├── raw_ncu.csv                        # NCU 导出的详细指标 CSV
├── output_ncu.csv                     # 处理后的 NCU 数据
├── output_ncu_processed.csv           # 提取关键指标后的数据
├── output_ncu_sms.csv                 # 添加 SM 需求后的数据
├── output_ncu_sms_roofline.csv        # 添加 Roofline 分析后的数据
├── kernel_info.csv                    # 最终调度配置文件
├── process_ncu_fixed.py               # 处理脚本 (修复版)
├── get_num_blocks_fixed.py            # SM 需求计算脚本 (修复版)
└── roofline_analysis_fixed.py         # Roofline 分析脚本 (修复版)
```

---

## 数据处理流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NCU 数据处理流水线                                    │
└─────────────────────────────────────────────────────────────────────────────┘

  .ncu-rep (NCU原始报告)
       │
       │  ncu --csv export
       ▼
┌──────────────────┐     ┌──────────────────┐
│  output_ncu.csv  │     │   raw_ncu.csv    │  ← 需要两个 CSV 文件
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         │  process_ncu_fixed.py  │
         ▼                        │
┌──────────────────────────┐      │
│ output_ncu_processed.csv │      │
└────────┬─────────────────┘      │
         │                        │
         │  get_num_blocks_fixed.py
         ▼                        │
┌──────────────────────┐          │
│  output_ncu_sms.csv  │          │
└────────┬─────────────┘          │
         │                        │
         │  roofline_analysis_fixed.py
         ▼                        │
┌────────────────────────────┐    │
│ output_ncu_sms_roofline.csv│◄───┘
└────────┬───────────────────┘
         │
         │  generate_file.py
         ▼
┌─────────────────────────────┐
│     kernel_info.csv         │  ← 最终调度配置文件
└─────────────────────────────┘
```

---

## 所需输入数据

### 1. NCU 原始报告 (.ncu-rep)

使用 NCU 采集 GPU kernel 性能数据：

```bash
ncu --set full -o output_ncu your_model.py
```

### 2. 导出 CSV 文件

从 .ncu-rep 文件导出两个 CSV：

```bash
# 基础指标 CSV
ncu --csv output_ncu.ncu-rep > output_ncu.csv

# 详细指标 CSV (用于 Roofline 分析)
ncu --csv --page raw output_ncu.ncu-rep > raw_ncu.csv
```

---

## 处理脚本说明

### 1. process_ncu_fixed.py

**功能**: 从原始 NCU CSV 提取关键指标

**输入**: `output_ncu.csv`
**输出**: `output_ncu_processed.csv`

**提取的指标**:
| 指标 | 含义 |
|------|------|
| Duration | Kernel 执行时间 (ns) |
| Block Size | 每个 block 的线程数 |
| Grid Size | Kernel 启动的 block 数 |
| Compute (SM) [%] | SM 计算利用率 |
| DRAM Throughput | 显存带宽利用率 |
| Registers Per Thread | 每线程寄存器数 |
| Static Shared Memory Per Block | 每 block 静态共享内存 |

**运行命令**:
```bash
python process_ncu_fixed.py --results_dir .
```

### 2. get_num_blocks_fixed.py

**功能**: 根据 GPU 硬件限制计算每个 kernel 需要的 SM 数量

**输入**: `output_ncu_processed.csv`
**输出**: `output_ncu_sms.csv`

**硬件参数** (Tesla T4 默认值):
| 参数 | 默认值 | 含义 |
|------|--------|------|
| max_threads_sm | 1024 | 每个 SM 最大活跃线程数 |
| max_blocks_sm | 16 | 每个 SM 最大活跃 block 数 |
| max_shmem_sm | 65536 | 每个 SM 最大共享内存 (bytes) |
| max_regs_sm | 65536 | 每个 SM 最大寄存器数 |

**运行命令**:
```bash
python get_num_blocks_fixed.py --results_dir .
```

### 3. roofline_analysis_fixed.py

**功能**: 基于 Roofline 模型判断 kernel 是计算密集型还是访存密集型

**输入**:
- `output_ncu_sms.csv`
- `raw_ncu.csv`

**输出**: `output_ncu_sms_roofline.csv`

**算术强度 (AI) 计算**:
```
AI = FLOPS / Bytes

其中:
  FLOPS = (fadd + fmul + ffma * 2) * cycles / 1000
  Bytes = dram_bytes_per_second

分类规则:
  AI > 9.72  → 计算密集型 (Profile = 1)
  AI <= 9.72 → 访存密集型 (Profile = 0)
  无法计算   → 未知型 (Profile = -1)
```

**重要**: 脚本已修复单位换算问题，正确处理 NCU 数据单位。

**运行命令**:
```bash
python roofline_analysis_fixed.py --results_dir . --ai_threshold 9.72
```

### 4. generate_file.py

**功能**: 生成最终的调度配置文件

**输入**: `output_ncu_sms_roofline.csv`
**输出**: `kernel_info.csv`

**运行命令**:
```bash
python ../profiling/postprocessing/generate_file.py \
    --input_file_name output_ncu_sms_roofline.csv \
    --output_file_name kernel_info.csv \
    --model_type transformer
```

---

## 完整处理流程

```bash
# 1. 导出 CSV (如果还没有)
ncu --csv single_client_ncu_minimal.ncu-rep > output_ncu.csv
ncu --csv --page raw single_client_ncu_minimal.ncu-rep > raw_ncu.csv

# 2. 运行处理流水线
python process_ncu_fixed.py --results_dir .
python get_num_blocks_fixed.py --results_dir .
python roofline_analysis_fixed.py --results_dir .
python ../profiling/postprocessing/generate_file.py \
    --input_file_name output_ncu_sms_roofline.csv \
    --output_file_name kernel_info.csv \
    --model_type transformer
```

---

## 输出文件格式

### kernel_info.csv

最终调度配置文件，供 Orion 调度器使用：

| 字段 | 含义 |
|------|------|
| Name | Kernel 名称 |
| Profile | 1=计算密集, 0=访存密集, -1=未知 |
| Memory_footprint | 内存占用 (预留字段) |
| SM_usage | 需要的 SM 数量 |
| Duration | 执行时间 (ns) |

---

## 常见问题

### Q: 为什么所有 kernel 都被分类为计算密集型？

**A**: 可能是单位换算问题。确保使用 `roofline_analysis_fixed.py` 而不是原始脚本。
原始脚本的 AI 计算结果是真实值的 1000 倍。

### Q: 如何调整以获得更多访存密集型 kernel？

**A**:
- 减小 `batch_size` 或 `sequence_length`
- 减小 `embedding_size`
- 或调整 `--ai_threshold` 参数

### Q: raw_ncu.csv 的第二行是什么？

**A**: 第二行是单位行（如 "inst/cycle", "Gbyte/second"），脚本会自动跳过。

---

## 当前数据统计

基于 `kernel_info.csv` 的统计结果：

| 分类 | 数量 | 占比 |
|------|------|------|
| 计算密集型 (Profile=1) | 103 | 32.3% |
| 访存密集型 (Profile=0) | 175 | 54.9% |
| 未知型 (Profile=-1) | 41 | 12.8% |
| **总计** | **319** | 100% |
