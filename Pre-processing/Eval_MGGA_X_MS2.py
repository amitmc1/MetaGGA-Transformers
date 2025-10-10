import torch
import pandas as pd
import numpy as np
from gda import GlobalDensityApprox
from gda.libxc import eval_xc

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Load data ===
df = pd.read_csv("rho_and_derivs_spin_1_cleaned.csv")

# === Prepare inputs ===
rho = torch.tensor(df["rho"].values, dtype=torch.float32, device=device, requires_grad=True)
grad_rho = torch.stack([
    torch.tensor(df["rho_gradient_x"].values, dtype=torch.float32, device=device, requires_grad=True),
    torch.tensor(df["rho_gradient_y"].values, dtype=torch.float32, device=device, requires_grad=True),
    torch.tensor(df["rho_gradient_z"].values, dtype=torch.float32, device=device, requires_grad=True)
], dim=1)
coords = torch.stack([
    torch.tensor(df["grid_coord_x"].values, dtype=torch.float32, device=device),
    torch.tensor(df["grid_coord_y"].values, dtype=torch.float32, device=device),
    torch.tensor(df["grid_coord_z"].values, dtype=torch.float32, device=device)
], dim=1)
weights = torch.tensor(df["weight"].values, dtype=torch.float32, device=device)

# === Load pretrained GDA model ===
gda = GlobalDensityApprox(embed_dim=128, n_blocks=3).to(device)
gda.eval()

# === Chunked tau_GDA computation ===
def compute_tau_gda_chunked(gda_model, rho, grad_rho, coords, weights, chunk_size=4000, log_file="chunk_log.txt"):
    tau_chunks = []
    n_chunks = (rho.shape[0] + chunk_size - 1) // chunk_size

    with open(log_file, "w") as f:
        for i in range(0, rho.shape[0], chunk_size):
            chunk_id = i // chunk_size + 1
            rho_chunk = rho[i:i+chunk_size]
            grad_rho_chunk = grad_rho[i:i+chunk_size]
            coords_chunk = coords[i:i+chunk_size]
            weights_chunk = weights[i:i+chunk_size]

            log_tau_chunk = gda_model.log_tau(rho_chunk, grad_rho_chunk, coords_chunk, weights_chunk)
            tau_chunk = torch.exp(log_tau_chunk)
            tau_chunks.append(tau_chunk)

            f.write(f"Processed chunk {chunk_id}/{n_chunks}\n")
            f.flush()  # Ensure write in case of crash

    return torch.cat(tau_chunks, dim=0)

# === Compute tau_GDA (frozen) ===
with torch.no_grad():
    tau_GDA = compute_tau_gda_chunked(
        gda_model=gda,
        rho=rho.detach().clone().float(),
        grad_rho=grad_rho.detach().clone().float(),
        coords=coords.float(),
        weights=weights.float(),
        chunk_size=4000,
        log_file="tau_GDA_chunk_log.txt"
    )

# === Add tau_GDA column and save ===
df["tau_GDA"] = tau_GDA.cpu().numpy()
df.to_csv("All_grid_data_with_tau_GDA.csv", index=False)

print("tau_GDA added as a column and saved to All_grid_data_with_tau_GDA.csv")