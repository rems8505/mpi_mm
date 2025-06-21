import subprocess
import re
import csv

process_counts = [2, 3, 4]
metrics_file = "benchmark_metrics.csv"
log_file = "benchmark_log.txt"

csv_header = [
    "Processes", "MatrixSize", "TotalTime", "ComputeAvg",
    "BcastAvg", "ScatterAvg", "GatherAvg", "MaxMemoryMB"
]

with open(metrics_file, "w", newline="") as csv_out, open(log_file, "w") as log_out:
    writer = csv.writer(csv_out)
    writer.writerow(csv_header)

    for n in process_counts:
        print(f"\nRunning with {n} processes...")
        try:
            result = subprocess.run(
                ["mpiexec", "-n", str(n), "python", "mpi_matrix_multiplication_metrics.py"],
                capture_output=True,
                text=True,
                check=True
            )

            stdout = result.stdout
            stderr = result.stderr

            log_out.write(f"===== {n} PROCESSES =====\n")
            log_out.write(stdout)
            if stderr:
                log_out.write("\n[STDERR]\n" + stderr)
            log_out.write("\n" + "-"*60 + "\n")

            def extract_float(label):
                match = re.search(fr"{label}\s*:\s*([0-9.]+)", stdout)
                return float(match.group(1)) if match else None

            matrix_size = extract_float("Matrix Size")
            total_time = extract_float("Total Time")
            compute_avg = extract_float("Compute Time")
            bcast_avg = extract_float("Bcast Time")
            scatter_avg = extract_float("Scatter Time")
            gather_avg = extract_float("Gather Time")
            mem_max = extract_float("Max Memory")

            writer.writerow([
                n, matrix_size, total_time, compute_avg,
                bcast_avg, scatter_avg, gather_avg, mem_max
            ])
            print(f"✅ Success. Time: {total_time:.4f}s")

        except subprocess.CalledProcessError as e:
            print(f"❌ Error with {n} processes:\n{e.stderr}")
            writer.writerow([n, "", "", "", "", "", "", ""])
            log_out.write(f"[ERROR] {n} processes\n{e.stderr}\n" + "-"*60 + "\n")

print(f"\nBenchmark complete. Results saved to: {metrics_file}")
