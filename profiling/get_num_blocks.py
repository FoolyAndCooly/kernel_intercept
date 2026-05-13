import pandas as pd
from math import ceil, floor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, required=True,
                        help='path to directory containing the profiling files')
parser.add_argument('--max_threads_sm', type=int, default=1024,
                        help='maximum number of threads that can be active in an SM (T4: 1024)')
parser.add_argument('--max_blocks_sm', type=int, default=16,
                        help='maximum number of blocks that can be active in an SM (T4: 16)')
parser.add_argument('--max_shmem_sm', type=int, default=65536,
                        help='maximum amount of shared memory (in bytes) per SM')
parser.add_argument('--max_regs_sm', type=int, default=65536,
                        help='maximum number of registers per SM')
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

    blocks_per_sm = min(blocks_per_sm_threads, blocks_per_sm_shmem, blocks_per_sm_regs)
    if blocks_per_sm == 0:
        blocks_per_sm = 1
    sm_needed_kernel = ceil(num_blocks/blocks_per_sm)

    sm_needed.append(sm_needed_kernel)


less = [x for x in sm_needed if x < 40]  # T4 has 40 SMs
print(f"Kernels using less than 40 SMs: {len(less)}, Total kernels: {len(sm_needed)}")

df['SM_needed'] = sm_needed
df.to_csv(f'{args.results_dir}/output_ncu_sms.csv', index=0)
print(f"Saved to {args.results_dir}/output_ncu_sms.csv")
