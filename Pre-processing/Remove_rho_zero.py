# -*- coding: utf-8 -*-
import pandas as pd
import os

# ======================================
# CONFIG
# ======================================
INPUT_FILE = "All_grid_data_with_tau_GDA_outputs.csv"
OUTPUT_FILE = "N2_scf1_cleaned.csv"

# ======================================
# LOAD AND FILTER
# ======================================
print(f"Reading {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE)

# Remove rows where rho == 0.0
initial_len = len(df)
df = df[df["rho"] != 0.0]
removed = initial_len - len(df)

# ======================================
# SAVE RESULT
# ======================================
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved cleaned file to {OUTPUT_FILE}")
print(f"Removed {removed} rows with rho == 0.0")
