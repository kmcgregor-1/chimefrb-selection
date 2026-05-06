import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
import json

DATA_PATH = "/home/kmcgregor/chimefrb-selection/chimefrb_selection/data/fits/4d_selection_function/fluence_scattering_time_width_dm/"

model_comparison = pd.read_csv(DATA_PATH + "model_comparison_fluence-scattering_time-width-dm_snr8_sl5.0.csv")
model_comparison_snrcut12 = pd.read_csv(DATA_PATH + "model_comparison_fluence-scattering_time-width-dm_snr12_sl5.0.csv")
model_comparison_snrcut15 = pd.read_csv(DATA_PATH + "model_comparison_fluence-scattering_time-width-dm_snr15_sl5.0.csv")

model_comparison = model_comparison[1:]
model_comparison_snrcut12 = model_comparison_snrcut12#[:-1]
model_comparison_snrcut15 = model_comparison_snrcut15#[:]

# Create a figure with a 3x2 plot matrix
fig, axes = plt.subplots(3, 3, figsize=(16, 8), sharex=True)

grid_there = True

fig.subplots_adjust(hspace=0.075)

# Add titles for SNR Cut 8 and SNR Cut 12 above the plot columns
axes[0, 0].set_title("S/N > 8", fontsize=24)
axes[0, 1].set_title("S/N > 12", fontsize=24)
axes[0, 2].set_title("S/N > 15", fontsize=24)

# Exclude the first two points in each array
model_comparison_filtered = model_comparison.iloc[:]
model_comparison_snrcut12_filtered = model_comparison_snrcut12.iloc[1:]
model_comparison_snrcut15_filtered = model_comparison_snrcut15.iloc[1:-2]

# Plot Log-Likelihood for SNR Cut 8
axes[0, 0].plot(model_comparison_filtered["Order"], model_comparison_filtered["Log-Likelihood"], marker='o', linestyle='--', color="magenta")
axes[0, 0].set_ylabel("Log-Likelihood", fontsize=12)
axes[0, 0].set_xlim(model_comparison_filtered["Order"].min(), model_comparison_filtered["Order"].max())
axes[0, 0].grid(grid_there)

# Plot Log-Likelihood for SNR Cut 12
axes[0, 1].plot(model_comparison_snrcut12_filtered["Order"], model_comparison_snrcut12_filtered["Log-Likelihood"], marker='s', color="magenta")
axes[0, 1].set_ylabel("Log-Likelihood", fontsize=12)
axes[0, 1].set_xlim(model_comparison_filtered["Order"].min(), model_comparison_filtered["Order"].max())
axes[0, 1].grid(grid_there)

# Plot Log-Likelihood for SNR Cut 15
axes[0, 2].plot(model_comparison_snrcut15_filtered["Order"], model_comparison_snrcut15_filtered["Log-Likelihood"], marker='s', color="magenta")
axes[0, 2].set_ylabel("Log-Likelihood", fontsize=12)
axes[0, 2].set_xlim(model_comparison_filtered["Order"].min(), model_comparison_filtered["Order"].max())
axes[0, 2].grid(grid_there)

# Plot Information Criteria (AIC and BIC) for SNR Cut 8
axes[1, 0].plot(model_comparison_filtered["Order"], model_comparison_filtered["AIC"], label="AIC", marker='o', linestyle='--')
axes[1, 0].plot(model_comparison_filtered["Order"], model_comparison_filtered["BIC"], label="BIC", marker='s', linestyle='--')
axes[1, 0].set_ylabel("Information Criteria", fontsize=12)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(grid_there)

# Plot Information Criteria (AIC and BIC) for SNR Cut 12
axes[1, 1].plot(model_comparison_snrcut12_filtered["Order"], model_comparison_snrcut12_filtered["AIC"], label="AIC", marker='o', linestyle='-')
axes[1, 1].plot(model_comparison_snrcut12_filtered["Order"], model_comparison_snrcut12_filtered["BIC"], label="BIC", marker='s', linestyle='-')
axes[1, 1].set_ylabel("Information Criteria", fontsize=12)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(grid_there)

axes[1, 2].plot(model_comparison_snrcut15_filtered["Order"], model_comparison_snrcut15_filtered["AIC"], label="AIC", marker='o', linestyle='-')
axes[1, 2].plot(model_comparison_snrcut15_filtered["Order"], model_comparison_snrcut15_filtered["BIC"], label="BIC", marker='s', linestyle='-')
axes[1, 2].set_ylabel("Information Criteria", fontsize=12)
axes[1, 2].legend(fontsize=10)
axes[1, 2].grid(grid_there)

# Plot Sum of Squared Residuals (RSS) for SNR Cut 8
axes[2, 0].plot(model_comparison_filtered["Order"], model_comparison_filtered["Sum of Squared Residuals"], label="SNR Cut 8", marker='o', linestyle='--', color="red")
axes[2, 0].set_xlabel("Polynomial Order", fontsize=12)
axes[2, 0].set_ylabel("RSS", fontsize=12)
axes[2, 0].grid(grid_there)

# Plot Sum of Squared Residuals (RSS) for SNR Cut 12
axes[2, 1].plot(model_comparison_snrcut12_filtered["Order"], model_comparison_snrcut12_filtered["Sum of Squared Residuals"], marker='s', color="red")
axes[2, 1].set_xlabel("Polynomial Order", fontsize=12)
axes[2, 1].set_ylabel("RSS", fontsize=12)
axes[2, 1].grid(grid_there)

# Plot Sum of Squared Residuals (RSS) for SNR Cut 15
axes[2, 2].plot(model_comparison_snrcut15_filtered["Order"], model_comparison_snrcut15_filtered["Sum of Squared Residuals"], marker='s', color="red")
axes[2, 2].set_xlabel("Polynomial Order", fontsize=12)
axes[2, 2].set_ylabel("RSS", fontsize=12)
axes[2, 2].grid(grid_there)

# Adjust layout
plt.tight_layout()
plt.savefig("/home/kmcgregor/chimefrb-selection/plots/model_selection.pdf", bbox_inches='tight')