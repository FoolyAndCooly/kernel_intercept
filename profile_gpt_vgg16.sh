#!/bin/bash
# profile_gpt_vgg16.sh
# 统计 GPT 和 VGG16 各算子的 SM 占用 + 单独运行时每个算子耗时
# 使用 NCU 对两个服务分别做单次前向 profiling，然后经过已有 pipeline 生成 kernel_info.csv
#
# 用法:
#   bash profile_gpt_vgg16.sh [--ncu /path/to/ncu] [--outdir /path/to/output]
#
# 输出目录结构:
#   <outdir>/
#     gpt/
#       ncu_raw.ncu-rep          # NCU 原始报告
#       output_ncu.csv           # NCU 导出
#       raw_ncu.csv              # NCU raw page 导出
#       output_ncu_processed.csv # 解析后的 metrics
#       output_ncu_sms.csv       # 加入 SM_needed 列
#       output_ncu_sms_roofline.csv  # 加入 Roofline 分类
#       kernel_info.csv          # 最终结果（Name, Profile, SM_usage, Duration）
#     vgg16/
#       ... (同上)

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
NCU="${NCU:-/usr/local/NVIDIA-Nsight-Compute-2025.4/ncu}"
OUTDIR="${OUTDIR:-$ROOT/profiles/profile_$(date +%Y%m%d_%H%M%S)}"
AI_THRESHOLD=9.72

while [[ $# -gt 0 ]]; do
    case $1 in
        --ncu)    NCU="$2";    shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── 检查 NCU ──────────────────────────────────────────────────────────────
if [[ ! -x "$NCU" ]]; then
    echo "[ERROR] ncu not found at $NCU"
    echo "  请用 --ncu 指定路径，例如: bash $0 --ncu /usr/local/cuda/bin/ncu"
    exit 1
fi

echo "========================================"
echo "  GPU Kernel Profiler: GPT & VGG16"
echo "  NCU    : $NCU"
echo "  OutDir : $OUTDIR"
echo "========================================"

# ── 读取 GPU 硬件参数 ───────────────────────────────────────────────────
echo "[0/6] Detecting GPU hardware..."
read NUM_SMS MAX_THREADS_SM MAX_REGS_SM MAX_SHMEM_SM GPU_NAME < <(python3 -c "
import torch
p = torch.cuda.get_device_properties(0)
print(p.multi_processor_count, p.max_threads_per_multi_processor,
      p.regs_per_multiprocessor, p.shared_memory_per_multiprocessor,
      p.name.replace(' ','_'))
")
MAX_BLOCKS_SM=$(python3 -c "
import torch
major = torch.cuda.get_device_properties(0).major
print(32 if major >= 8 else 16)
")
echo "  GPU      : $GPU_NAME"
echo "  SMs      : $NUM_SMS"
echo "  threads/SM: $MAX_THREADS_SM  blocks/SM: $MAX_BLOCKS_SM"
echo "  regs/SM  : $MAX_REGS_SM  shmem/SM: $MAX_SHMEM_SM"

# ======================================================================
# 通用函数：对单个模型跑完整 pipeline
# profile_model <model_tag> <python_script_inline>
# ======================================================================
profile_model() {
    local TAG="$1"
    local WORK="$OUTDIR/$TAG"
    mkdir -p "$WORK"

    echo ""
    echo "========================================"
    echo "  Profiling: $TAG"
    echo "  WorkDir : $WORK"
    echo "========================================"

    # ── Step 1: NCU profiling ─────────────────────────────────────────
    echo "[$TAG][1/5] Running NCU (--set full)..."
    local PY_SCRIPT="$WORK/run_model.py"

    if [[ "$TAG" == "gpt" ]]; then
        cat > "$PY_SCRIPT" <<PYEOF
import sys, os
sys.path.insert(0, "${ROOT}/python")

import torch
import importlib.util

# 动态 import GPT.py
spec = importlib.util.spec_from_file_location(
    "GPT",
    "${ROOT}/python/GPT.py"
)
gpt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpt_module)

# exec_module 之后覆盖模块级变量，再构建模型
# 调小配置，让大部分 kernel 的 SM 占用 < 114
gpt_module.emb_size     = 768   # 从 2048 减到 768
gpt_module.head_size    = 64    # 从 512 减到 64
gpt_module.n_layer      = 1
gpt_module.sequence_len = 128   # 从 1024 减到 128

CharGPT = gpt_module.CharGPT
vocab_size = 10000
model = CharGPT(vocab_size).to("cuda").eval()
x = torch.randint(0, vocab_size, (2, 128), device="cuda")  # batch_size 从 16 减到 2

with torch.no_grad():
    for _ in range(3): model(x)  # warmup
torch.cuda.synchronize()

# 正式 profiling pass
with torch.no_grad():
    model(x)
torch.cuda.synchronize()
print("GPT profile done")
PYEOF
    elif [[ "$TAG" == "vgg16" ]]; then
        cat > "$PY_SCRIPT" <<'PYEOF'
import sys, os
import torch
import torch.nn as nn

# 尝试用 torchvision，否则用内置简化版
try:
    from torchvision.models import vgg16
    model = vgg16(weights=None).to("cuda").eval()
    print("Using torchvision VGG16")
except ImportError:
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
        nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.AdaptiveAvgPool2d((7, 7)),
        nn.Flatten(),
        nn.Linear(256 * 7 * 7, 4096), nn.ReLU(),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Linear(4096, 1000),
    ).to("cuda").eval()
    print("Using built-in VGG16 approximation")

# 调小配置：batch_size=2, 112x112（从 8, 224x224 减小）
x = torch.randn(2, 3, 112, 112, device="cuda")

with torch.no_grad():
    for _ in range(3): model(x)  # warmup
torch.cuda.synchronize()

# 正式 profiling pass
with torch.no_grad():
    model(x)
torch.cuda.synchronize()
print("VGG16 profile done")
PYEOF
    fi

    "$NCU" --set full \
           --target-processes all \
           --csv \
           -o "$WORK/ncu_raw" \
           python3 "$PY_SCRIPT"
    echo "[$TAG][1/5] NCU profiling done"

    # ── Step 2: Export CSVs ───────────────────────────────────────────
    echo "[$TAG][2/5] Exporting NCU CSVs..."
    "$NCU" -i "$WORK/ncu_raw.ncu-rep" --csv            > "$WORK/output_ncu.csv"
    "$NCU" -i "$WORK/ncu_raw.ncu-rep" --csv --page raw > "$WORK/raw_ncu.csv"

    # ── Step 3: Extract metrics ───────────────────────────────────────
    echo "[$TAG][3/5] Extracting kernel metrics..."
    python3 "$ROOT/ncu_profile/process_ncu_fixed.py" --results_dir "$WORK"

    # ── Step 4: Compute SM usage ──────────────────────────────────────
    echo "[$TAG][4/5] Computing SM usage..."
    python3 "$ROOT/ncu_profile/get_num_blocks_fixed.py" \
        --results_dir    "$WORK" \
        --max_threads_sm "$MAX_THREADS_SM" \
        --max_blocks_sm  "$MAX_BLOCKS_SM" \
        --max_shmem_sm   "$MAX_SHMEM_SM" \
        --max_regs_sm    "$MAX_REGS_SM" \
        --num_sms        "$NUM_SMS"

    # ── Step 5: Roofline + generate kernel_info ───────────────────────
    echo "[$TAG][5/5] Roofline analysis + generating kernel_info.csv..."
    python3 "$ROOT/ncu_profile/roofline_analysis_fixed.py" \
        --results_dir "$WORK" \
        --ai_threshold "$AI_THRESHOLD"

    # 模型类型：GPT 是 transformer，VGG16 是 vision
    local MODEL_TYPE="transformer"
    if [[ "$TAG" == "vgg16" ]]; then
        MODEL_TYPE="vision"
    fi

    python3 "$ROOT/profiling/generate_file.py" \
        --input_file_name  "$WORK/output_ncu_sms_roofline.csv" \
        --output_file_name "$WORK/kernel_info.csv" \
        --model_type       "$MODEL_TYPE"

    # ── Summary ───────────────────────────────────────────────────────
    echo ""
    echo "[$TAG] === Summary ==="
    python3 - "$WORK/kernel_info.csv" "$NUM_SMS" <<'PYEOF'
import sys
path, num_sms = sys.argv[1], int(sys.argv[2])

# 手动解析 CSV（从右边解析最后 4 列，避免 kernel 名称中的逗号干扰）
rows = []
with open(path) as f:
    lines = f.readlines()
    if len(lines) <= 1:
        print("  (No kernels found in output)")
        sys.exit(0)

    for line in lines[1:]:  # 跳过表头
        line = line.strip()
        if not line:
            continue

        # 从右边找最后 4 个逗号
        comma_positions = [i for i, c in enumerate(line) if c == ',']
        if len(comma_positions) < 4:
            continue

        n = len(comma_positions)
        pos_profile = comma_positions[n - 4]
        pos_mem = comma_positions[n - 3]
        pos_sm = comma_positions[n - 2]
        pos_dur = comma_positions[n - 1]

        name = line[:pos_profile]
        profile = line[pos_profile + 1:pos_mem]
        sm_str = line[pos_sm + 1:pos_dur]
        dur_str = line[pos_dur + 1:]

        try:
            rows.append({
                'name': name,
                'profile': profile,
                'sm': int(float(sm_str)),
                'dur': float(dur_str)
            })
        except ValueError:
            continue

if not rows:
    print("  (No valid kernels found)")
    sys.exit(0)

sms = [r['sm'] for r in rows]
durs = [r['dur'] for r in rows]
profiles = [r['profile'] for r in rows]

print(f"  Total kernels    : {len(rows)}")
print(f"  Compute-bound(1) : {profiles.count('1')}")
print(f"  Memory-bound (0) : {profiles.count('0')}")
print(f"  Unknown     (-1) : {profiles.count('-1')}")
print(f"  SM_usage range   : {min(sms)} – {max(sms)}  (GPU has {num_sms} SMs)")
print(f"  Duration range   : {min(durs):.0f} – {max(durs):.0f} ns")
print("")
print(f"  {'Kernel Name':<60} {'SM':>4} {'Dur(ns)':>10} {'Prof':>5}")
print(f"  {'-'*60} {'-'*4} {'-'*10} {'-'*5}")
for r in sorted(rows, key=lambda x: -x['sm'])[:20]:  # 只显示前 20 个
    tag = {'-1':'???', '0':'MEM', '1':'COM'}.get(r['profile'], '???')
    short = r['name'][:58] if len(r['name']) > 60 else r['name']
    print(f"  {short:<60} {r['sm']:>4} {r['dur']:>10.1f} {tag:>5}")
PYEOF
    echo ""
    echo "[$TAG] kernel_info.csv => $WORK/kernel_info.csv"
}

# ── 逐个 profile ──────────────────────────────────────────────────────────
profile_model "gpt"
profile_model "vgg16"

# ── 对比汇总 ──────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Comparison Summary"
echo "========================================"
python3 - "$OUTDIR" "$NUM_SMS" <<'PYEOF'
import sys, os
outdir, num_sms = sys.argv[1], int(sys.argv[2])

def parse_kernel_info(path):
    rows = []
    with open(path) as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            line = line.rstrip('\n')
            if not line:
                continue
            # 从右边解析最后 4 个字段
            commas = [i for i, c in enumerate(line) if c == ',']
            if len(commas) < 4:
                continue
            n = len(commas)
            name = line[:commas[n-4]]
            profile_str = line[commas[n-4]+1:commas[n-3]]
            sm_str = line[commas[n-2]+1:commas[n-1]]
            dur_str = line[commas[n-1]+1:]
            try:
                rows.append({
                    'name': name,
                    'profile': profile_str.strip(),
                    'sm': int(float(sm_str)),
                    'dur': float(dur_str)
                })
            except ValueError:
                continue
    return rows

for tag in ["gpt", "vgg16"]:
    path = os.path.join(outdir, tag, "kernel_info.csv")
    if not os.path.exists(path):
        print(f"[{tag}] kernel_info.csv not found, skipping")
        continue
    rows = parse_kernel_info(path)
    if not rows:
        print(f"[{tag}] No kernels")
        continue
    sms  = [r['sm'] for r in rows]
    durs = [r['dur'] for r in rows]
    profs = [r['profile'] for r in rows]
    total_dur = sum(durs)
    print(f"\n=== {tag.upper()} ===")
    print(f"  Kernels : {len(rows)}")
    print(f"  compute-bound: {profs.count('1')}  memory-bound: {profs.count('0')}  unknown: {profs.count('-1')}")
    print(f"  Total Duration : {total_dur/1e6:.2f} ms  (one forward pass)")
    print(f"  Avg SM_usage   : {sum(sms)/len(sms):.1f}  Max: {max(sms)}  (GPU: {num_sms} SMs)")
    print(f"  Top-5 by SM_usage:")
    for r in sorted(rows, key=lambda x: -x['sm'])[:5]:
        print(f"    SM={r['sm']:>6}  {r['dur']/1e3:>8.2f}us  {r['name'][:55]}")
    print(f"  Top-5 by Duration:")
    for r in sorted(rows, key=lambda x: -x['dur'])[:5]:
        print(f"    {r['dur']/1e3:>8.2f}us  SM={r['sm']:>6}  {r['name'][:55]}")
PYEOF

echo ""
echo "========================================"
echo "  All done. Results in: $OUTDIR"
echo "    gpt/kernel_info.csv"
echo "    vgg16/kernel_info.csv"
echo "========================================"
