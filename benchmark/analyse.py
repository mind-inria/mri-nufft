import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import glob
from collections import defaultdict
sns.set_theme()


# Replace with correct directory
BENCHMARK_DIR = "/volatile/pierre-antoine/mri-nufft/benchmark/3d-results/2023-11-03_11-30-13"

results_files = glob.glob(BENCHMARK_DIR + "/**/results.csv")

df = pd.concat(map(pd.read_csv, results_files))
df["coil_time"] = df["run_time"] / df["n_coils"]
df["coil_mem"] = df["mem_peak"] / df["n_coils"]

df

# +
fig, axs = plt.subplots(3,3, sharey=True, figsize=(16,9), gridspec_kw=dict(hspace=0.01, wspace=0.05))
tasks =  ["forward", "adjoint", "grad"]
metrics = {"coil_time":"time (s) /coil", "mem_peak": "Peak RAM (GB)", "gpu0_mem_GiB_peak": "Peak GPU Mem (GB)"}
palette = {k: v for k,v in zip(metrics.keys(), ["magma", "rocket", "mako"])}

xlims = {k: v for k, v in zip(metrics.keys(), [(0,10), (0, 80), (0,8)])}


for row, task in zip(axs,tasks): 
    ddf = df[df["task"]==task]
    for ax, (k , p) in zip(row,palette.items()):
        sns.barplot(ddf, x=k, y="backend", hue="n_coils", palette=p, ax=ax, errorbar=None, width=0.8)
        ax.get_legend().remove()
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xticklabels("")


# labels
for ax , xlabel in zip(axs[-1, :], metrics.values()):
    ax.set_xlabel(xlabel)
for ax , xlabel in zip(axs[0, :], metrics.values()):
    h, l = ax.get_legend_handles_labels()
    h.insert(0, matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none',
                                 visible=False))
    l.insert(0, "# Coils")
    ax.legend(h,l, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    ax.set_title(xlabel, pad=40)

for rl, task in zip(axs[:, 0], tasks):
    rl.set_ylabel(task)
# ticks 
# rescale xlim  per column: 
for col_ax, xlim in zip(axs.T, xlims.values()):
    for ax in col_ax: 
        ax.set_xlim(xlim)
    ax.set_xticklabels([f"{xt:.0f}" for xt in ax.get_xticks()])


# +
# Replace with correct directory

BENCHMARK_DIR = "/volatile/pierre-antoine/mri-nufft/benchmark/3d-results/2023-11-03_15-55-09/"
results_files = glob.glob(BENCHMARK_DIR + "/**/results.csv")
dfstacked = pd.concat(map(pd.read_csv, results_files))
dfstacked["coil_time"] = dfstacked["run_time"] / dfstacked["n_coils"]
dfstacked["coil_mem"] = dfstacked["mem_peak"] / dfstacked["n_coils"]


# +
df = dfstacked
df = df.sort_values(["backend"], ascending=False)
fig, axs = plt.subplots(3,3, sharey=True, figsize=(16,9), gridspec_kw=dict(hspace=0.01, wspace=0.05))
tasks =  ["forward", "adjoint", "grad"]
metrics = {"coil_time":"time (s) /coil", "mem_peak": "Peak RAM (GB)", "gpu0_mem_GiB_peak": "Peak GPU Mem (GB)"}
palette = {k: v for k,v in zip(metrics.keys(), ["magma", "rocket", "mako"])}

xlims = {k: v for k, v in zip(metrics.keys(), [(0,10), (0, 110), (0,8)])}


for row, task in zip(axs,tasks): 
    ddf = df[df["task"]==task]
    for ax, (k , p) in zip(row,palette.items()):
        sns.barplot(ddf, x=k, y="backend", hue="n_coils", palette=p, ax=ax, errorbar=None, width=0.8)
        ax.get_legend().remove()
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xticklabels("")


# labels
for ax , xlabel in zip(axs[-1, :], metrics.values()):
    ax.set_xlabel(xlabel)
for ax , xlabel in zip(axs[0, :], metrics.values()):
    h, l = ax.get_legend_handles_labels()
    h.insert(0, matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none',
                                 visible=False))
    l.insert(0, "# Coils")
    ax.legend(h,l, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    ax.set_title(xlabel, pad=40)

for rl, task in zip(axs[:, 0], tasks):
    rl.set_ylabel(task)
# ticks 
# rescale xlim  per column: 
for col_ax, xlim in zip(axs.T, xlims.values()):
    for ax in col_ax: 
        ax.set_xlim(xlim)
        ax.axhline(2.5, c="w", linestyle="dashed")
    ax.set_xticklabels([f"{xt:.0f}" for xt in ax.get_xticks()])

# -


