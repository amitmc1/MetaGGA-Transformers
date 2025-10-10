# -*- coding: utf-8 -*-
import os
import pandas as pd

# === Base directory and subdirectories ===
base_dir = r"{Directory with FHI-aims outputs}"
subdirs = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10","11","12","13","14"]

all_dfs = []

# === Loop through subdirs and read each file ===
for sub in subdirs:
    fname = os.path.join(base_dir, sub, "rho_and_derivs_spin_1.dat")
    print(f"Reading {fname}")

    # Read header line, remove "#" and split into names
    with open(fname, "r") as f:
        header_line = f.readline().strip().lstrip("#").split()

    # Load file using the header
    df = pd.read_csv(fname, delim_whitespace=True, comment="#", names=header_line)

    # Remove rows where weight == 0.0
    df = df[df["weight"] != 0.0]

    # Add SCF_id column
    df["SCF_id"] = sub

    # Save cleaned version in same folder
    out_clean = os.path.join(base_dir, sub, f"rho_and_derivs_spin_1_cleaned.csv")
    df.to_csv(out_clean, index=False)
    print(f"  → Saved cleaned file to {out_clean}")

    all_dfs.append(df)

# === Merge into one DataFrame ===
merged_df = pd.concat(all_dfs, ignore_index=True)

# === Save as combined CSV ===
outname = os.path.join(base_dir, "merged_data.csv")
merged_df.to_csv(outname, index=False)
print(f"\n✅ Saved merged file to {outname}")
