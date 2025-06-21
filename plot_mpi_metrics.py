import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("benchmark_metrics.csv")
df.dropna(subset=["TotalTime"], inplace=True)

# Derived metrics
baseline_time = df["TotalTime"].iloc[0]
df["Speedup"] = baseline_time / df["TotalTime"]
df["Efficiency"] = df["Speedup"] / df["Processes"]
df["CommOverhead"] = df["BcastAvg"] + df["ScatterAvg"] + df["GatherAvg"]

# Plot: Speedup
plt.figure(figsize=(8, 5))
plt.plot(df["Processes"], df["Speedup"], marker='o', color='blue')
plt.title("Speedup vs Number of Processes")
plt.xlabel("Number of Processes")
plt.ylabel("Speedup")
plt.grid(True)
plt.savefig("plot_speedup.png")
plt.show()

# Plot: Efficiency
plt.figure(figsize=(8, 5))
plt.plot(df["Processes"], df["Efficiency"], marker='s', color='green')
plt.title("Efficiency vs Number of Processes")
plt.xlabel("Number of Processes")
plt.ylabel("Efficiency")
plt.grid(True)
plt.savefig("plot_efficiency.png")
plt.show()

# Plot: Compute vs Communication
plt.figure(figsize=(8, 5))
plt.plot(df["Processes"], df["ComputeAvg"], marker='o', label="Compute Time", color='purple')
plt.plot(df["Processes"], df["CommOverhead"], marker='^', label="Communication Overhead", color='orange')
plt.title("Computation vs Communication Time")
plt.xlabel("Number of Processes")
plt.ylabel("Time (s)")
plt.legend()
plt.grid(True)
plt.savefig("plot_compute_comm.png")
plt.show()

# Plot: Max Memory
plt.figure(figsize=(8, 5))
plt.plot(df["Processes"], df["MaxMemoryMB"], marker='d', color='red')
plt.title("Max Memory Usage per Process")
plt.xlabel("Number of Processes")
plt.ylabel("Memory (MB)")
plt.grid(True)
plt.savefig("plot_memory.png")
plt.show()
