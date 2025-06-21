from mpi4py import MPI
import numpy as np
import time
import os
import psutil
import sys

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Matrix size
N = 6000

def log(msg):
    print(f"[Rank {rank}] {msg}")

def get_mem_usage_MB():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

# Only rank 0 initializes full matrices
if rank == 0:
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    full_start_time = time.time()
else:
    A = None
    B = np.empty((N, N), dtype='d')

# --- Broadcast B ---
bcast_start = time.time()
comm.Bcast(B, root=0)
bcast_end = time.time()

# Validate matrix partitioning
rows_per_proc = N // size
if N % size != 0:
    if rank == 0:
        log(f"Matrix size {N} not divisible by number of processes {size}")
    MPI.Finalize()
    sys.exit(1)

# --- Scatter A ---
A_chunk = np.empty((rows_per_proc, N), dtype='d')
scatter_start = time.time()
comm.Scatter(A, A_chunk, root=0)
scatter_end = time.time()

# --- Local computation ---
compute_start = time.time()
C_chunk = np.matmul(A_chunk, B)
compute_end = time.time()

# --- Gather C ---
C = np.empty((N, N), dtype='d') if rank == 0 else None
gather_start = time.time()
comm.Gather(C_chunk, C, root=0)
gather_end = time.time()

# Final timing
if rank == 0:
    full_end_time = time.time()
    total_time = full_end_time - full_start_time
    log(f"MPI matrix multiplication completed in {total_time:.4f} seconds")

# --- Performance metrics ---
compute_time = compute_end - compute_start
bcast_time = bcast_end - bcast_start
scatter_time = scatter_end - scatter_start
gather_time = gather_end - gather_start
mem_usage = get_mem_usage_MB()

# Gather metrics to rank 0
rank_metrics = np.array([compute_time, bcast_time, scatter_time, gather_time, mem_usage], dtype='d')
all_metrics = None
if rank == 0:
    all_metrics = np.empty((size, 5), dtype='d')
comm.Gather(rank_metrics, all_metrics, root=0)

# --- Save metrics summary ---
if rank == 0:
    compute_avg = np.mean(all_metrics[:, 0])
    bcast_avg = np.mean(all_metrics[:, 1])
    scatter_avg = np.mean(all_metrics[:, 2])
    gather_avg = np.mean(all_metrics[:, 3])
    mem_max = np.max(all_metrics[:, 4])

    print("\n=== Performance Metrics Summary ===")
    print(f"Processes      : {size}")
    print(f"Matrix Size    : {N}")
    print(f"Total Time     : {total_time:.4f} s")
    print(f"Compute Time   : avg = {compute_avg:.4f}")
    print(f"Bcast Time     : avg = {bcast_avg:.4f}")
    print(f"Scatter Time   : avg = {scatter_avg:.4f}")
    print(f"Gather Time    : avg = {gather_avg:.4f}")
    print(f"Max Memory     : {mem_max:.2f} MB")

    with open("metrics_summary.csv", "a") as f:
        f.write(f"{N},{size},{total_time:.4f},{compute_avg:.4f},{bcast_avg:.4f},{scatter_avg:.4f},{gather_avg:.4f},{mem_max:.2f}\n")
