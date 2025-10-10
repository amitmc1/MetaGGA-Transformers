# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import time
import os
import pylibxc
import torch.serialization
torch.serialization.add_safe_globals([slice])
from e3nn.o3 import spherical_harmonics

# ======================================
# CONFIG
# ======================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
NUM_EXPERTS = 4
D_MODEL_EXP = 512
L_MAX = 3
CSV_FILE = None
NORM_FILE = "normalization_constants_full.json"
EXPERT_DIR = r"/nfshome/store01/users/c.c21127846/phd/week1/MoE/Expert Models"
GATING_DIR = r"/nfshome/store01/users/c.c21127846/phd/week1/MoE/Gating Network"
GATING_MODEL_PATH = os.path.join(GATING_DIR, "gating_transformer_best.pth")
BATCH_SIZE = 4000
TIMING_FILE = "moe_inference_timing.json"
OUTPUT_CSV = "moe_inference_results.csv"
ALL_SYSTEMS_DIR = r"/scratch/c.c21127846/N2"


XC_FUNCTIONAL = "MGGA_X_MS2"
XC_SPIN = "unpolarized"

FEATURES = [
    "rho", "rho_gradient_x", "rho_gradient_y", "rho_gradient_z",
    "tau_GDA", "tau_TF", "tau_VW", "ex_LDA", "z", "alpha",
    "reduced_grad", "sigma", "weight",
    "exc_GDA", "vrho_GDA", "vsigma_GDA", "vtau_GDA"
]

EXPERT_FEATURES = [
    "rho", "rho_gradient_x", "rho_gradient_y", "rho_gradient_z",
    "tau_GDA", "tau_TF", "tau_VW", "ex_LDA", "z", "alpha",
    "reduced_grad", "sigma", "weight"
]

POS_COLS = ["grid_coord_x", "grid_coord_y", "grid_coord_z"]

# ======================================
# LOAD NORMALIZATION CONSTANTS
# ======================================
with open(NORM_FILE, "r") as f:
    norm_stats = json.load(f)

# ======================================
# MODEL DEFINITIONS
# ======================================
class E3NNPositionalEncoding(nn.Module):
    def __init__(self, d_model, l_max=3):
        super().__init__()
        self.l_list = list(range(l_max + 1))
        self.num_harmonics = sum(2 * l + 1 for l in self.l_list)
        self.proj = nn.Sequential(
            nn.Linear(self.num_harmonics + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        ).double()

    def forward(self, coords):
        coords = coords.to(dtype=DTYPE)
        r = torch.norm(coords, dim=-1, keepdim=True) + 1e-8
        unit_coords = torch.nn.functional.normalize(coords, dim=-1)
        Y = spherical_harmonics(self.l_list, unit_coords,
                                normalize=True, normalization='component')
        log_r = torch.log1p(r)
        feats = torch.cat([log_r, Y], dim=-1)
        return self.proj(feats)

class TransformerGatingNetwork(nn.Module):
    def __init__(self, feature_dim, d_model=512, nhead=2, num_layers=2, num_experts=4, l_max=3):
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, d_model).double()
        self.pos_enc = E3NNPositionalEncoding(d_model, l_max=l_max)
        self.norm1 = nn.LayerNorm(d_model).double()
        self.norm2 = nn.LayerNorm(d_model).double()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=0.2, batch_first=True
        ).double()
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.gating_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2).double(),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_experts).double()
        )

    def forward(self, features, coords, temperature=1.0):
        x_feat = self.feature_proj(features.double())
        pos_enc = self.pos_enc(coords.double())
        x_feat = self.norm1(x_feat + pos_enc)
        x = x_feat.unsqueeze(0)               # (1, N, d_model)
        x = self.transformer(x)               # apply transformer
        x = self.norm2(x.squeeze(0))          # (N, d_model)
        logits = self.gating_head(x)
        return torch.softmax(logits / temperature, dim=-1)

class MetaGGAEnhancementTransformer(nn.Module):
    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=8):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model).double()
        self.pos_enc = E3NNPositionalEncoding(d_model, l_max=3)
        self.norm1 = nn.LayerNorm(d_model).double()
        self.norm2 = nn.LayerNorm(d_model).double()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=0.2, batch_first=True
        ).double()
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model).double(),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model // 2).double(),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1).double()
        )

    def forward(self, x, coords):
        x_feat = self.embedding(x.double())
        pos_enc = self.pos_enc(coords.double())
        x_feat = self.norm1(x_feat + pos_enc)
        x = self.transformer(x_feat)
        x = self.norm2(x)
        return self.output(x).squeeze(-1)

csv_files = [f for f in os.listdir(ALL_SYSTEMS_DIR) if f.endswith(".csv")]
print(f"Found {len(csv_files)} system files to process.")

# ======================================
# LOAD MODELS
# ======================================
timing = {}
t0 = time.perf_counter()
gating_model = TransformerGatingNetwork(
    feature_dim=len(FEATURES),
    d_model=512,
    nhead=2,
    num_layers=2,
     num_experts=NUM_EXPERTS,
    l_max=3
).to(DEVICE)

gating_model.load_state_dict(torch.load(GATING_MODEL_PATH, map_location=DEVICE))
gating_model.eval()

expert_models = {}
expert_norms = {}

# Define expert-specific dropout rates
expert_dropouts = {1: 0.2, 2: 0.2, 3: 0.05, 4: 0.05}

for i in range(1, NUM_EXPERTS + 1):
    model_path = f"{EXPERT_DIR}/Expert{i}_transformer_finetuned.pt"
    norm_path = f"{EXPERT_DIR}/normalization_constants_expert{i}.json"
    with open(norm_path, "r") as f:
        expert_norms[i] = json.load(f)

    # Rebuild the expert with its specific dropout rate
    dropout_rate = expert_dropouts[i]
    model = MetaGGAEnhancementTransformer(
        input_dim=len(EXPERT_FEATURES),
        d_model=D_MODEL_EXP,
        nhead=8,
        num_layers=8
    ).to(DEVICE)

    # Overwrite dropout layers in both transformer encoder and MLP
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    expert_models[i] = model

t1 = time.perf_counter()
timing["model_load_time_sec"] = t1 - t0
model_load_time_sec = timing["model_load_time_sec"]

for csv_file in csv_files:
    system_name = os.path.splitext(csv_file)[0]
    print(f"Running inference for: {system_name}")

    # Set up paths
    CSV_FILE = os.path.join(ALL_SYSTEMS_DIR, csv_file)
    system_outdir = os.path.join(ALL_SYSTEMS_DIR, system_name)
    os.makedirs(system_outdir, exist_ok=True)

    OUTPUT_CSV = os.path.join(system_outdir, f"{system_name}_MoE_results.csv")
    TIMING_FILE = os.path.join(system_outdir, f"{system_name}_timing.json")

    # ======================================
    # TIMING START
    # ======================================
    timing = {}
    timing["model_load_time_sec"] = model_load_time_sec
    t_global_start = time.perf_counter()

    t0 = time.perf_counter()
    df = pd.read_csv(CSV_FILE)
    if "sigma" not in df.columns:
        df["sigma"] = df["rho_gradient_x"] ** 2 + df["rho_gradient_y"] ** 2 + df["rho_gradient_z"] ** 2
    t1 = time.perf_counter()
    timing["data_load_time_sec"] = t1 - t0

    # ======================================
    # REFRESH LibXC
    # ======================================
    t0 = time.perf_counter()
    rho_np = np.ascontiguousarray(df["rho"].values, dtype=np.float64)
    sigma_np = np.ascontiguousarray(df["sigma"].values, dtype=np.float64)
    tau_gda_np = np.ascontiguousarray(df["tau_GDA"].values, dtype=np.float64)

    xc = pylibxc.LibXCFunctional(XC_FUNCTIONAL, XC_SPIN)
    xc_out = xc.compute({"rho": rho_np, "sigma": sigma_np, "tau": tau_gda_np})

    df["exc_GDA"] = np.asarray(xc_out["zk"], dtype=np.float64)
    df["vrho_GDA"] = np.asarray(xc_out["vrho"], dtype=np.float64)
    df["vsigma_GDA"] = np.asarray(xc_out["vsigma"], dtype=np.float64)
    df["vtau_GDA"] = np.asarray(xc_out["vtau"], dtype=np.float64)

    t1 = time.perf_counter()
    timing["pylibxc_eval_time_sec"] = round(t1 - t0, 6)  # microsecond precision

    # ======================================
    # BATCHED INFERENCE (with separate gating + expert normalization)
    # ======================================
    t0 = time.perf_counter()
    results = []
    feature_names = EXPERT_FEATURES
    rho_idx   = feature_names.index("rho")
    sigma_idx = feature_names.index("sigma")
    tau_idx   = feature_names.index("tau_GDA")

    for start in range(0, len(df), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(df))  # ✅ clip to dataset length
        batch_len = end - start

        norm_feats_full = np.zeros((batch_len, len(FEATURES)), dtype=np.float64)
        for i, name in enumerate(FEATURES):
            mean = norm_stats[name]["mean"]
            std  = norm_stats[name]["std"]
            vals = df.iloc[start:end][name].values
            if norm_stats[name].get("symlog", 0):
                vals = np.sign(vals) * np.log1p(np.abs(vals))
            norm_feats_full[:, i] = (vals - mean) / (std if std != 0 else 1.0)

        # --- Global-normalized features for gating ---
        norm_feats_full = np.zeros((batch_len, len(FEATURES)), dtype=np.float64)

        for i, name in enumerate(FEATURES):
            mean = norm_stats[name]["mean"]
            std = norm_stats[name]["std"]
            vals = df.iloc[start:end][name].values
            if norm_stats[name].get("symlog", 0):
                vals = np.sign(vals) * np.log1p(np.abs(vals))
            norm_feats_full[:, i] = (vals - mean) / (std if std != 0 else 1.0)

        feats_full = torch.tensor(norm_feats_full, dtype=DTYPE, device=DEVICE)

        coords = torch.tensor(df.iloc[start:end][POS_COLS].values, dtype=DTYPE, device=DEVICE)
        exlda_batch = torch.tensor(df.iloc[start:end]["ex_LDA"].values, dtype=DTYPE, device=DEVICE)

        # --- Gating step ---
        with torch.no_grad():
            probs = gating_model(feats_full, coords)
            routed = torch.argmax(probs, dim=1) + 1  # 1-based expert index

        # storage for predictions
        exc_out   = torch.zeros(len(feats_full), device=DEVICE, dtype=DTYPE)
        vrho_out  = torch.zeros(len(feats_full), device=DEVICE, dtype=DTYPE)
        vsigma_out= torch.zeros(len(feats_full), device=DEVICE, dtype=DTYPE)
        vtau_out  = torch.zeros(len(feats_full), device=DEVICE, dtype=DTYPE)

        # --- Expert step (with per-expert normalization) ---
        for expert_id in range(1, NUM_EXPERTS+1):
            idx = (routed == expert_id).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue

            # Raw features for this expert
            #raw_feats_sub = df.iloc[start:end].iloc[idx][EXPERT_FEATURES].values
            raw_feats_sub = df.iloc[start:end].iloc[idx.detach().cpu().numpy()][EXPERT_FEATURES].values

            # Apply expert-specific normalization
            norm_feats_sub = []
            for j, name in enumerate(EXPERT_FEATURES):
                vals = torch.tensor(raw_feats_sub[:, j], dtype=DTYPE, device=DEVICE)
                mean = expert_norms[expert_id][name]["mean"]
                std  = expert_norms[expert_id][name]["std"]
                if expert_norms[expert_id][name].get("symlog", 0):
                    vals = torch.sign(vals) * torch.log1p(torch.abs(vals))
                norm_feats_sub.append((vals - mean) / (std if std != 0 else 1.0))
            norm_feats_sub = torch.stack(norm_feats_sub, dim=1).requires_grad_(True)

            coords_sub = coords[idx]
            exlda_sub = exlda_batch[idx].to(DTYPE)

            # Forward pass
            f_pred = expert_models[expert_id](norm_feats_sub, coords_sub)
            stds = expert_norms[expert_id]

            exc_pred = f_pred * exlda_sub
            grads = torch.autograd.grad(exc_pred.sum(), norm_feats_sub, create_graph=False, retain_graph=False)[0]

            vrho_pred   = grads[:, rho_idx]   / stds["rho"]["std"]
            vsigma_pred = grads[:, sigma_idx] / stds["sigma"]["std"]
            vtau_pred   = grads[:, tau_idx]   / stds["tau_GDA"]["std"]

            exc_out[idx]   = exc_pred.detach()
            vrho_out[idx]  = vrho_pred.detach()
            vsigma_out[idx]= vsigma_pred.detach()
            vtau_out[idx]  = vtau_pred.detach()

        # --- Save batch results ---
        df_out = df.iloc[start:end].copy()
        df_out["chosen_expert"] = routed.cpu().numpy()
        df_out["exc_pred"] = exc_out.cpu().numpy()
        df_out["vrho_pred"] = vrho_out.cpu().numpy()
        df_out["vsigma_pred"] = vsigma_out.cpu().numpy()
        df_out["vtau_pred"] = vtau_out.cpu().numpy()
        results.append(df_out)

    final_df = pd.concat(results).reset_index(drop=True)
    t1 = time.perf_counter()
    timing["expert_inference_time_sec"] = t1 - t0

    # ======================================
    # SAVE RESULTS
    # ======================================
    t0 = time.perf_counter()
    final_df.to_csv(OUTPUT_CSV, index=False)
    t1 = time.perf_counter()
    timing["save_results_time_sec"] = t1 - t0

    # ======================================
    # SUMMARY + TIMINGS
    # ======================================
    t_global_end = time.perf_counter()
    timing["total_runtime_sec"] = t_global_end - t_global_start
    timing["total_points"] = len(df)
    timing["output_csv"] = OUTPUT_CSV

    with open(TIMING_FILE, "w") as f:
        json.dump(timing, f, indent=2)

    print(f"Saved MoE predictions to {OUTPUT_CSV}")
    print(f"Timing info saved to {TIMING_FILE}")
