import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, required=True,
                        help='path to directory containing the profiling files')
parser.add_argument('--ai_threshold', type=float, default=9.72,
                        help='arithmetic intensity that separates compute from memory bound kernels')
args = parser.parse_args()

# Read raw NCU data (one row per kernel)
# Skip row 1 which contains units (e.g., "inst/cycle", "Gbyte/second")
df_raw = pd.read_csv(f'{args.results_dir}/raw_ncu.csv', skiprows=[1])
df_basic = pd.read_csv(f'{args.results_dir}/output_ncu_sms.csv')

print(f"Raw NCU kernels: {len(df_raw)}, Basic kernels: {len(df_basic)}")

# Column names for the metrics we need
fadd_col = 'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed'
fmul_col = 'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed'
ffma_col = 'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed'
cycles_col = 'smsp__cycles_elapsed.avg.per_second'
bytes_col = 'dram__bytes.sum.per_second'

def parse_float(val):
    """Parse float value, handling various formats"""
    if pd.isna(val):
        return 0.0
    if isinstance(val, str):
        val = val.replace(',', '').replace("'", '')
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

ai_list = []
roofline_prof = []  # 1: comp, 0: mem, -1: invalid

comp_bound = 0
mem_bound = 0
rest = 0

# Get DRAM throughput and Compute SM from basic file
dram_throughput = df_basic['DRAM_Throughput(%)'].tolist()
comp_throughput = df_basic['Compute(SM)(%)'].tolist()

for index, row in df_raw.iterrows():
    if index >= len(dram_throughput):
        break

    # Get float operation counts
    fadd = parse_float(row.get(fadd_col, 0))
    fmul = parse_float(row.get(fmul_col, 0))
    ffma = parse_float(row.get(ffma_col, 0))
    cycles = parse_float(row.get(cycles_col, 0))
    bytes_sec = parse_float(row.get(bytes_col, 0))

    if (fadd or fmul or ffma) and bytes_sec > 0:
        # Calculate FLOPS/cycle (FMA counts as 2 operations)
        flops_cycle = fadd + fmul + ffma * 2
        # flops_sec unit: inst/usecond (cycles is cycle/usecond)
        flops_sec = flops_cycle * cycles
        # Unit conversion:
        #   flops_sec is in inst/usecond = 1e6 inst/second
        #   bytes_sec is in Gbyte/second = 1e9 bytes/second
        #   AI (FLOPS/byte) = (flops_sec * 1e6) / (bytes_sec * 1e9) = flops_sec / (bytes_sec * 1000)
        ai = flops_sec / (bytes_sec * 1000) if bytes_sec > 0 else 0
        ai_list.append(ai)

        if ai > args.ai_threshold:
            roofline_prof.append(1)
            comp_bound += 1
        else:
            roofline_prof.append(0)
            mem_bound += 1
    else:
        ai_list.append(0.0)
        # Fallback to throughput-based classification
        comp_pct = parse_float(comp_throughput[index])
        dram_pct = parse_float(dram_throughput[index])

        if comp_pct >= 60.0:
            roofline_prof.append(1)
        elif dram_pct >= 60.0:
            roofline_prof.append(0)
        else:
            roofline_prof.append(-1)
        rest += 1

# Ensure lists match df_basic length
while len(ai_list) < len(df_basic):
    ai_list.append(0.0)
    roofline_prof.append(-1)
    rest += 1

df_basic['AI(flops/bytes)'] = ai_list[:len(df_basic)]
df_basic['Roofline_prof'] = roofline_prof[:len(df_basic)]
df_basic.to_csv(f'{args.results_dir}/output_ncu_sms_roofline.csv', index=False)

print(f"Compute bound: {comp_bound}, Memory bound: {mem_bound}, Unknown: {rest}, Total: {comp_bound+mem_bound+rest}")
print(f"Saved to {args.results_dir}/output_ncu_sms_roofline.csv")
