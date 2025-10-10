import numpy as np
import pandas as pd

# === Config ===
input_csv = "N2_scf1_cleaned.csv"
output_csv = "N2_scf1_cleaned_descriptors.csv"

# === Load Data ===
df = pd.read_csv(input_csv)

# Ensure required columns exist
required = ["rho", "sigma", "tau_GDA"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# === Extract as arrays ===
rho = np.ascontiguousarray(df["rho"].values, dtype=np.float64)
sigma = np.ascontiguousarray(df["sigma"].values, dtype=np.float64)
tau_GDA = np.ascontiguousarray(df["tau_GDA"].values, dtype=np.float64)

# === Compute descriptors ===
A_TF = (3.0 / 10.0) * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
tau_TF = A_TF * rho ** (5.0 / 3.0)
tau_VW = np.zeros_like(rho)
tau_VW[rho != 0.0] = sigma[rho != 0.0] / (8.0 * rho[rho != 0.0])  # avoid div by zero

# z and alpha
z = np.zeros_like(rho)
alpha = np.zeros_like(rho)
valid = tau_TF != 0.0
z[valid] = tau_GDA[valid] / tau_TF[valid]
alpha[valid] = (tau_GDA[valid] - tau_VW[valid]) / tau_TF[valid]

# LDA exchange energy density
C_X = -(3.0 / 4.0) * (3.0 / np.pi) ** (1.0 / 3.0)
ex_LDA = C_X * np.cbrt(rho)  # rho^(1/3)

# Gradient magnitude and reduced gradient
grad_mag = np.sqrt(sigma)
F = 2.0 * (3.0 * np.pi ** 2) ** (1.0 / 3.0)
reduced_grad = np.zeros_like(rho)
valid = rho != 0.0
reduced_grad[valid] = grad_mag[valid] / (F * rho[valid] ** (4.0 / 3.0))

# === Add new columns ===
new_cols = {
    "tau_TF": tau_TF,
    "tau_VW": tau_VW,
    "z": z,
    "alpha": alpha,
    "ex_LDA": ex_LDA,
    "grad_mag": grad_mag,
    "reduced_grad": reduced_grad
}

for col, values in new_cols.items():
    df[col] = values

# === Save updated CSV ===
df.to_csv(output_csv, index=False)
print(f"Updated CSV with descriptors saved to: {output_csv}")
