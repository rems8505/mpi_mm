# mpi_mm# 🧮 Distributed Matrix Multiplication using MPI (Miniconda + WSL)

This project demonstrates matrix multiplication using both **serial** and **MPI-parallel** approaches. It is built for systems using **Miniconda in WSL (Ubuntu on Windows)** and supports benchmarking and performance analysis.

---

## 📂 Project Files

| File                                 | Description                               |
|--------------------------------------|-------------------------------------------|
| `serial_matrix_multiplication.py`    | Serial matrix multiplication (nested loops) |
| `mpi_matrix_multiplication_metrics.py` | MPI version with timing and memory metrics |
| `benchmark_mpi_metrics.py`           | Benchmarks parallel runs (2–8 processes)  |
| `plot_mpi_metrics.py`                | Visualizes speedup, efficiency, overhead  |
| `metrics_summary.csv`                | Metrics per run (auto-generated)          |
| `benchmark_metrics.csv`              | Results across all runs                   |
| `benchmark_log.txt`                  | Raw logs for each process count           |

---

## ⚙️ Environment Setup (WSL + Miniconda)

### 1. Install WSL (Ubuntu)
```bash
wsl --install
```

### 2. Install Miniconda inside WSL
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### 3. Create and Activate Environment
```bash
conda create -n mpi-env python=3.11 -y
conda activate mpi-env
pip install mpi4py matplotlib pandas psutil
sudo apt install openmpi-bin libopenmpi-dev
```
or

```bash
conda env create -f environment.yml
conda activate mpi-env

```
This will:
Set up the mpi-env environment
Install all dependencies for running and plotting your MPI matrix multiplication project

---

## 🚀 How to Run

### Run Serial Version
```bash
python serial_matrix_multiplication.py
```

### Run MPI Version
```bash
mpiexec -n 4 python mpi_matrix_multiplication_metrics.py
```

### Run Benchmarks (2, 3, 4, 8 processes)
```bash
python benchmark_mpi_metrics.py
```

### Plot Results
```bash
python plot_mpi_metrics.py
```

---

## 📊 Metrics Collected

- Total Execution Time
- Average Compute Time per Process
- Broadcast / Scatter / Gather Overheads
- Maximum Memory Usage (MB)
- Speedup & Efficiency (plotted automatically)

---

## 📈 Visual Outputs

- `plot_speedup.png`  
- `plot_efficiency.png`  
- `plot_compute_comm.png`  
- `plot_memory.png`

All plots show how performance scales with process count.

---

## 🧪 Notes


- All metrics are written to CSV for reproducibility.
- Logs and raw output stored in `benchmark_log.txt`.

---

## ✅ Requirements

- WSL (Ubuntu)
- Miniconda (Python 3.11+)
- Packages: `mpi4py`, `psutil`, `pandas`, `matplotlib`
- OpenMPI (via `apt`)

---

## 💡 Optional Enhancements

- Add support for varying matrix size (strong scaling)
- Automate plotting comparisons for different N
- Extend to distributed multi-node execution via SSH
