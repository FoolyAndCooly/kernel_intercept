import pandas as pd
from math import ceil, floor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, required=True,
                        help='path to directory containing the profiling files')
parser.add_argument('--max_threads_sm', type=int, default=2048,
                        help='maximum number of threads that can be active in an SM (H100: 2048, T4: 1024)')
parser.add_argument('--max_blocks_sm', type=int, default=32,
                        help='maximum number of blocks that can be active in an SM (H100: 32, T4: 16)')
parser.add_argument('--max_shmem_sm', type=int, default=233472,
                        help='maximum amount of shared memory (in bytes) per SM (H100: 233472, T4: 65536)')
parser.add_argument('--max_regs_sm', type=int, default=65536,
                        help='maximum number of registers per SM')
parser.add_argument('--num_sms', type=int, default=114,
                        help='total number of SMs on the GPU (H100 PCIe: 114, A100: 108, T4: 40)')
args = parser.parse_args()

df = pd.read_csv(f'{args.results_dir}/output_ncu_processed.csv', index_col=0)

max_threads_sm = args.max_threads_sm
max_blocks_sm = args.max_blocks_sm
max_shmem_sm = args.max_shmem_sm
max_regs_sm = args.max_regs_sm

def parse_num(val):
    """Parse numeric value, handling comma-separated numbers and floats"""
    if isinstance(val, str):
        val = val.replace(',', '').replace("'", '')
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 1  # default to 1 to avoid division by zero

sm_needed = []

for index, row in df.iterrows():
    num_blocks = parse_num(row['Grid'])
    num_threads = parse_num(row['Number_of_threads'])
    threads_per_block = parse_num(row['Block'])
    shmem_per_block = parse_num(row['Static_shmem_per_block'])
    regs_per_thread = parse_num(row['Registers_Per_Thread'])

    if threads_per_block == 0:
        threads_per_block = 1

    # from threads
    blocks_per_sm_threads = ceil(max_threads_sm/threads_per_block)

    # from shmem
    if shmem_per_block > 0:
        blocks_per_sm_shmem = ceil(max_shmem_sm/shmem_per_block)
    else:
        blocks_per_sm_shmem = blocks_per_sm_threads

    # from registers
    if regs_per_thread > 0:
        regs_per_wrap = ceil(32*regs_per_thread/256) * 256
        wraps_per_sm = floor((65536/regs_per_wrap)/4) * 4
        wraps_per_block = ceil(threads_per_block/32)
        if wraps_per_block > 0:
            blocks_per_sm_regs = int(wraps_per_sm/wraps_per_block)
        else:
            blocks_per_sm_regs = blocks_per_sm_threads
    else:
        blocks_per_sm_regs = blocks_per_sm_threads

    blocks_per_sm = min(blocks_per_sm_threads, blocks_per_sm_shmem, blocks_per_sm_regs,
                        args.max_blocks_sm)
    if blocks_per_sm == 0:
        blocks_per_sm = 1

    # 计算理论 SM 需求（不 cap）
    # 这是 kernel 真正需要的 SM 数量，可能超过 GPU 实际 SM 数
    sm_needed_kernel = ceil(num_blocks / blocks_per_sm)

    sm_needed.append(sm_needed_kernel)


less = [x for x in sm_needed if x < args.num_sms]
print(f"Kernels using fewer than all {args.num_sms} SMs: {len(less)}, Total kernels: {len(sm_needed)}")

df['SM_needed'] = sm_needed
df.to_csv(f'{args.results_dir}/output_ncu_sms.csv', index=0)
print(f"Saved to {args.results_dir}/output_ncu_sms.csv")
