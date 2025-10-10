import numpy as np
import pandas as pd
import pylibxc

# ======== Configuration ========
input_csv = "All_grid_data_with_tau_GDA.csv"
output_csv = "All_grid_data_with_tau_GDA_outputs.csv"
functional_name = "MGGA_X_MS2"
spin = "unpolarized"

# ======== Load input file as DataFrame ========
df = pd.read_csv(input_csv)

# Ensure required columns exist
required = ["rho", "rho_gradient_x", "rho_gradient_y", "rho_gradient_z", "tau", "tau_GDA"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ======== Compute sigma (∇ρ·∇ρ) ========
if "sigma" not in df.columns:
    df["sigma"] = (
        df["rho_gradient_x"]**2
        + df["rho_gradient_y"]**2
        + df["rho_gradient_z"]**2
    )

# Prepare inputs as contiguous arrays
rho = np.ascontiguousarray(df["rho"].values, dtype=np.float64)
sigma = np.ascontiguousarray(df["sigma"].values, dtype=np.float64)
tau = np.ascontiguousarray(df["tau"].values, dtype=np.float64)
tau_gda = np.ascontiguousarray(df["tau_GDA"].values, dtype=np.float64)
tau_half = 0.5 * tau

# ======== Setup LibXC functional ========
functional = pylibxc.LibXCFunctional(functional_name, spin)

def compute_outputs(rho_in, sigma_in, tau_in):
    data_in = {"rho": rho_in, "sigma": sigma_in, "tau": tau_in}
    out = functional.compute(data_in)

    exc = out.get("zk", None)
    vrho = out.get("vrho", None)
    vsigma = out.get("vsigma", None)
    vtau = out.get("vtau", None)

    if any(x is None for x in (exc, vrho, vsigma, vtau)):
        raise RuntimeError(f"LibXC '{functional_name}' did not return all outputs.")
    return (exc, vrho, vsigma, vtau)

# ======== Compute for tau_GDA and tau_half ========
exc_GDA, vrho_GDA, vsigma_GDA, vtau_GDA = compute_outputs(rho, sigma, tau_gda)
exc_MS2, vrho_MS2, vsigma_MS2, vtau_MS2 = compute_outputs(rho, sigma, tau_half)

# ======== Add new columns to DataFrame (no duplicates) ========
new_columns = {
    "exc_GDA": exc_GDA,
    "vrho_GDA": vrho_GDA,
    "vsigma_GDA": vsigma_GDA,
    "vtau_GDA": vtau_GDA,
    "exc_MS2": exc_MS2,
    "vrho_MS2": vrho_MS2,
    "vsigma_MS2": vsigma_MS2,
    "vtau_MS2": vtau_MS2
}

for col, values in new_columns.items():
    df[col] = values

# ======== Save updated CSV ========
df.to_csv(output_csv, index=False)
print(f"Updated file with new columns saved to: {output_csv}")
