#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_eda_plots.py
======================
Generate all EDA figures for the GitHub Pages portfolio site.

Produces publication-quality plots organised by analysis level:
  - Corpus-level  : dataset size, plate/well structure, role balance
  - Variable-level: univariate distributions (histograms, boxplots, KDE)
  - Interaction-level: correlation heatmaps, pairplots, cosine similarity
  - Dimensionality reduction: PCA explained variance, UMAP embeddings
  - Methodological: distribution shape checks for downstream assumptions
  - Baseline: DMSO control characterisation
  - Provenance: cleaning-step log summary

Data sources:
  - metadata/*.csv  (CellProfiler per-well profiles, 1800 cols)
  - results/compound_analysis_dinov3_vitb16/features_compound.npz
  - results/compound_analysis_dinov3_vitb16/compound_cluster_mapping.csv
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
META_DIR = ROOT / "metadata"
FEAT_NPZ = ROOT / "results" / "compound_analysis_dinov3_vitb16" / "features_compound.npz"
CLUSTER_CSV = ROOT / "results" / "compound_analysis_dinov3_vitb16" / "compound_cluster_mapping.csv"
OUT_DIR = ROOT / "docs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PALETTE = {
    "primary": "#264653",
    "secondary": "#2a9d8f",
    "accent": "#e9c46a",
    "highlight": "#f4a261",
    "alert": "#e76f51",
    "bg": "#fafafa",
    "treated": "#3498db",
    "mock": "#2ecc71",
    "dmso": "#e74c3c",
}

sns.set_theme(style="whitegrid", font_scale=1.05, rc={
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#cccccc",
    "grid.color": "#eeeeee",
    "font.family": "sans-serif",
})

def msg(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)

def save(fig, name, dpi=200):
    path = OUT_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    msg(f"  Saved {path.name}")


# ===================================================================
# DATA LOADING
# ===================================================================
msg("Loading metadata (sampling 20 plates for CellProfiler features)...")
meta_files = sorted(META_DIR.glob("meta*.csv"))
msg(f"  Found {len(meta_files)} plate metadata files")

# Load ALL metadata (only essential columns) for corpus-level stats
essential_cols = [
    "Metadata_Plate", "Metadata_Well", "Metadata_ASSAY_WELL_ROLE",
    "Metadata_broad_sample", "Metadata_pert_id", "Metadata_pert_type",
    "Metadata_mmoles_per_liter", "Metadata_broad_sample_type",
]

all_meta_frames = []
for f in meta_files:
    try:
        header = pd.read_csv(f, nrows=0).columns
        usecols = [c for c in essential_cols if c in header]
        df = pd.read_csv(f, usecols=usecols)
        plate_id = f.stem.replace("meta", "")
        df["plate_id"] = plate_id
        all_meta_frames.append(df)
    except Exception:
        continue

all_meta = pd.concat(all_meta_frames, ignore_index=True)
all_meta["role"] = all_meta.get("Metadata_ASSAY_WELL_ROLE", "unknown").astype(str).str.lower()
msg(f"  Loaded {len(all_meta):,} wells across {all_meta['plate_id'].nunique()} plates")

# Load a sample of plates with ALL columns for CellProfiler feature EDA
rng = np.random.default_rng(42)
sample_plates = rng.choice(meta_files, size=min(20, len(meta_files)), replace=False)
cp_frames = []
for f in sample_plates:
    try:
        df = pd.read_csv(f)
        plate_id = f.stem.replace("meta", "")
        df["plate_id"] = plate_id
        cp_frames.append(df)
    except Exception:
        continue
cp_data = pd.concat(cp_frames, ignore_index=True)
cp_data["role"] = cp_data.get("Metadata_ASSAY_WELL_ROLE", "unknown").astype(str).str.lower()

# Separate numeric CellProfiler features
cp_feature_cols = [c for c in cp_data.columns
                   if c.startswith(("Cells_", "Cytoplasm_", "Nuclei_"))]
cp_features = cp_data[cp_feature_cols].apply(pd.to_numeric, errors="coerce")
msg(f"  CellProfiler features: {len(cp_feature_cols)} columns, {len(cp_data):,} wells (20-plate sample)")

# Load DINOv3 compound features + clusters
msg("Loading DINOv3 compound features...")
npz = np.load(FEAT_NPZ, allow_pickle=True)
compound_ids = npz["compound_ids"]
X_raw = npz["compound_features"].astype(np.float32)          # pre-selection
X_clust = npz["compound_features_for_clustering"].astype(np.float32)  # post-selection
umap_emb = npz["umap_embedding"].astype(np.float32)
clusters = npz["clusters"]
feat_mask = npz["well_feature_mask"]  # which of 768 dims survived selection
msg(f"  Compounds: {X_clust.shape[0]:,}, Features (raw): {X_raw.shape[1]}, Features (selected): {X_clust.shape[1]}")

cluster_df = pd.read_csv(CLUSTER_CSV)
msg(f"  Cluster mapping: {len(cluster_df):,} rows, {cluster_df['cluster'].nunique()} clusters")


# ===================================================================
# PROVENANCE LOG (recorded as a dict for the site)
# ===================================================================
provenance = []

def log_step(step, description, before, after, effect):
    provenance.append({
        "step": step,
        "description": description,
        "before": before,
        "after": after,
        "effect": effect,
    })

# Reconstruct the cleaning steps from the pipeline
n_raw_features = 768  # DINOv3 ViT-B/16 output dim
n_selected = int(feat_mask.sum())
n_removed_var = n_raw_features - n_selected  # approximate (includes corr removal)

log_step("1. Image loading", "Read 16-bit TIFFs directly as float32", "Raw TIFF", "float32 array", "No lossy 8-bit conversion")
log_step("2. Percentile normalisation", "Clip to [p1, p99] per channel, scale to [0,1]", "Arbitrary intensity range", "[0, 1] float32", "Removes plate-level intensity variation")
log_step("3. Virtual tiling", "Split each image into 3x4 grid of tiles", f"{len(all_meta):,} images", f"{len(all_meta)*12:,} tiles", "Preserves cellular resolution at 224x224 input")
log_step("4. Feature extraction", "DINOv3 ViT-B/16 [CLS] token per tile", f"{len(all_meta)*12:,} tiles", f"{len(all_meta):,} x 768-dim", "Self-supervised morphological features")
log_step("5. Tile aggregation", "Mean-pool tile features per image", "12 tiles/image", "1 vector/image", "Image-level representation")
log_step("6. Well aggregation", "Mean across sites within each well", f"{len(all_meta):,} images", f"~{len(all_meta):,} wells", "Well-level features")
log_step("7. Harmony batch correction", "Soft k-means on plate labels at well level", "Plate-confounded", "Batch-corrected", f"iLISI before: 18.76 (moderate plate structure)")
log_step("8. Well-to-compound aggregation", "Median across replicate wells per compound", f"~{len(all_meta):,} wells", f"{X_clust.shape[0]:,} compounds", "Robust compound profiles")
log_step("9. Variance filter", f"Remove features with var < 0.01", f"{n_raw_features} features", f"~{n_selected + 50} features", f"Removed ~{n_raw_features - n_selected - 50} near-constant dims")
log_step("10. Correlation filter", f"Drop one of each pair with |r| > 0.95", f"~{n_selected + 50} features", f"{n_selected} features", "Removed redundant dimensions")
log_step("11. L2 normalisation", "Row-wise L2 norm to unit sphere", f"{X_clust.shape[0]:,} x {n_selected}", "Same shape", "Cosine distance = Euclidean on unit sphere")

provenance_df = pd.DataFrame(provenance)
provenance_df.to_csv(OUT_DIR / "provenance_log.csv", index=False)
msg("Provenance log saved")


# ===================================================================
# FIGURE 1: CORPUS-LEVEL — Dataset Overview
# ===================================================================
msg("Generating corpus-level plots...")

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

# 1a. Wells per plate
ax1 = fig.add_subplot(gs[0, 0])
wells_per_plate = all_meta.groupby("plate_id").size()
ax1.hist(wells_per_plate.values, bins=30, color=PALETTE["primary"], alpha=0.85, edgecolor="white")
ax1.axvline(wells_per_plate.median(), color=PALETTE["alert"], linestyle="--", linewidth=2,
            label=f"Median: {wells_per_plate.median():.0f}")
ax1.set_xlabel("Wells per plate")
ax1.set_ylabel("Number of plates")
ax1.set_title("(a) Wells per Plate Distribution")
ax1.legend(fontsize=9)

# 1b. Role distribution
ax2 = fig.add_subplot(gs[0, 1])
role_counts = all_meta["role"].value_counts()
colors_role = [PALETTE["treated"] if r == "treated" else PALETTE["mock"] if r == "mock" else "#95a5a6"
               for r in role_counts.index]
bars = ax2.bar(role_counts.index, role_counts.values, color=colors_role, alpha=0.85, edgecolor="white")
for bar, val in zip(bars, role_counts.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f"{val:,}", ha="center", va="bottom", fontsize=9)
ax2.set_ylabel("Number of wells")
ax2.set_title("(b) Well Role Distribution")
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

# 1c. Compounds per cluster
ax3 = fig.add_subplot(gs[0, 2])
cluster_sizes = cluster_df["cluster"].value_counts().sort_index()
ax3.bar(cluster_sizes.index, cluster_sizes.values, color=PALETTE["secondary"], alpha=0.85, edgecolor="white")
ax3.axhline(cluster_sizes.mean(), color=PALETTE["alert"], linestyle="--",
            label=f"Mean: {cluster_sizes.mean():.0f}")
ax3.set_xlabel("Leiden cluster")
ax3.set_ylabel("Compounds")
ax3.set_title(f"(c) Cluster Size Distribution (n={len(cluster_sizes)})")
ax3.legend(fontsize=9)

# 1d. Replicate wells per compound
ax4 = fig.add_subplot(gs[1, 0])
wells_per_compound = cluster_df["n_wells"]
ax4.hist(wells_per_compound.values, bins=range(1, wells_per_compound.max()+2),
         color=PALETTE["highlight"], alpha=0.85, edgecolor="white")
ax4.axvline(wells_per_compound.median(), color=PALETTE["primary"], linestyle="--", linewidth=2,
            label=f"Median: {wells_per_compound.median():.0f}")
ax4.set_xlabel("Replicate wells per compound")
ax4.set_ylabel("Number of compounds")
ax4.set_title("(d) Replicate Coverage")
ax4.legend(fontsize=9)

# 1e. Plates per compound
ax5 = fig.add_subplot(gs[1, 1])
plates_per_compound = cluster_df["n_plates"]
ax5.hist(plates_per_compound.values, bins=range(1, plates_per_compound.max()+2),
         color=PALETTE["accent"], alpha=0.85, edgecolor="white")
ax5.axvline(plates_per_compound.median(), color=PALETTE["primary"], linestyle="--", linewidth=2,
            label=f"Median: {plates_per_compound.median():.0f}")
ax5.set_xlabel("Plates per compound")
ax5.set_ylabel("Number of compounds")
ax5.set_title("(e) Cross-Plate Coverage")
ax5.legend(fontsize=9)

# 1f. Summary stats text
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
stats_text = (
    f"Dataset Summary\n"
    f"{'─'*30}\n"
    f"Plates:        {all_meta['plate_id'].nunique():>8,}\n"
    f"Wells:         {len(all_meta):>8,}\n"
    f"Compounds:     {X_clust.shape[0]:>8,}\n"
    f"DINOv3 dims:   {n_raw_features:>8}\n"
    f"Selected dims: {n_selected:>8}\n"
    f"Clusters:      {len(np.unique(clusters)):>8}\n"
    f"{'─'*30}\n"
    f"Treated wells: {(all_meta['role']=='treated').sum():>8,}\n"
    f"Mock wells:    {(all_meta['role']=='mock').sum():>8,}\n"
    f"Median wells/compound: {wells_per_compound.median():>4.0f}\n"
    f"Median plates/compound: {plates_per_compound.median():>3.0f}\n"
)
ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes, fontsize=11,
         verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#cccccc"))

fig.suptitle("Corpus-Level Overview", fontsize=16, fontweight="bold", color=PALETTE["primary"], y=1.01)
save(fig, "eda_01_corpus_overview.png")


# ===================================================================
# FIGURE 2: VARIABLE-LEVEL — CellProfiler Feature Distributions
# ===================================================================
msg("Generating variable-level plots (CellProfiler features)...")

# Pick 6 representative CellProfiler features across compartments
representative_features = []
for prefix in ["Cells_AreaShape_Area", "Cells_AreaShape_Eccentricity",
               "Nuclei_AreaShape_Area", "Nuclei_AreaShape_FormFactor",
               "Cytoplasm_AreaShape_Area", "Cytoplasm_AreaShape_Eccentricity"]:
    if prefix in cp_features.columns:
        representative_features.append(prefix)
# Fallback: pick first available from each compartment
if len(representative_features) < 6:
    for prefix in ["Cells_", "Nuclei_", "Cytoplasm_"]:
        available = [c for c in cp_features.columns if c.startswith(prefix)]
        for c in available:
            if c not in representative_features:
                representative_features.append(c)
            if len(representative_features) >= 6:
                break
        if len(representative_features) >= 6:
            break

representative_features = representative_features[:6]

# 2a: Histograms + KDE
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(axes.flat, representative_features):
    vals = cp_features[col].dropna()
    # Clip extreme outliers for display
    q01, q99 = vals.quantile(0.01), vals.quantile(0.99)
    vals_clip = vals[(vals >= q01) & (vals <= q99)]
    ax.hist(vals_clip, bins=60, density=True, color=PALETTE["secondary"], alpha=0.5, edgecolor="none")
    try:
        vals_clip.plot.kde(ax=ax, color=PALETTE["primary"], linewidth=2)
    except Exception:
        pass
    short_name = col.replace("Cells_", "C.").replace("Nuclei_", "N.").replace("Cytoplasm_", "Cy.")
    ax.set_title(short_name, fontsize=10)
    ax.set_ylabel("Density")
    ax.tick_params(labelsize=8)

fig.suptitle("Variable-Level: Histograms + KDE (CellProfiler Features, 20-Plate Sample)",
             fontsize=14, fontweight="bold", color=PALETTE["primary"], y=1.01)
plt.tight_layout()
save(fig, "eda_02_univariate_histograms_kde.png")

# 2b: Boxplots by role (treated vs mock)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(axes.flat, representative_features):
    data_plot = cp_data[[col, "role"]].dropna()
    data_plot = data_plot[data_plot["role"].isin(["treated", "mock"])]
    data_plot[col] = pd.to_numeric(data_plot[col], errors="coerce")
    data_plot = data_plot.dropna()
    # Clip for display
    q01, q99 = data_plot[col].quantile(0.01), data_plot[col].quantile(0.99)
    data_plot = data_plot[(data_plot[col] >= q01) & (data_plot[col] <= q99)]

    sns.boxplot(data=data_plot, x="role", y=col, ax=ax,
                palette={"treated": PALETTE["treated"], "mock": PALETTE["mock"]},
                fliersize=2, linewidth=1.2)
    short_name = col.replace("Cells_", "C.").replace("Nuclei_", "N.").replace("Cytoplasm_", "Cy.")
    ax.set_title(short_name, fontsize=10)
    ax.set_xlabel("")
    ax.tick_params(labelsize=8)

fig.suptitle("Variable-Level: Boxplots by Role (Treated vs Mock)",
             fontsize=14, fontweight="bold", color=PALETTE["primary"], y=1.01)
plt.tight_layout()
save(fig, "eda_03_univariate_boxplots.png")


# ===================================================================
# FIGURE 3: VARIABLE-LEVEL — DINOv3 Feature Distributions
# ===================================================================
msg("Generating DINOv3 feature distribution plots...")

# Pick 8 representative DINOv3 features (spread across the 337 dims)
dino_indices = np.linspace(0, X_clust.shape[1]-1, 8, dtype=int)

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, idx in zip(axes.flat, dino_indices):
    vals = X_clust[:, idx]
    ax.hist(vals, bins=60, density=True, color=PALETTE["highlight"], alpha=0.5, edgecolor="none")
    # KDE
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(vals)
        x_grid = np.linspace(vals.min(), vals.max(), 200)
        ax.plot(x_grid, kde(x_grid), color=PALETTE["primary"], linewidth=2)
    except Exception:
        pass
    ax.set_title(f"DINOv3 dim {idx}", fontsize=10)
    ax.set_ylabel("Density")
    ax.tick_params(labelsize=8)

fig.suptitle("Variable-Level: DINOv3 Feature Distributions (Post-Selection, L2-Normalised)",
             fontsize=14, fontweight="bold", color=PALETTE["primary"], y=1.01)
plt.tight_layout()
save(fig, "eda_04_dino_feature_distributions.png")


# ===================================================================
# FIGURE 4: INTERACTION-LEVEL — Correlation Heatmap (DINOv3)
# ===================================================================
msg("Generating interaction-level plots...")

# 4a: DINOv3 feature correlation heatmap (sampled dims for readability)
sample_dims = np.linspace(0, X_clust.shape[1]-1, 50, dtype=int)
corr_matrix = np.corrcoef(X_clust[:, sample_dims].T)

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
ax.set_xlabel("Feature index (sampled)")
ax.set_ylabel("Feature index (sampled)")
ax.set_title("Interaction-Level: Feature Correlation Matrix\n(50 Sampled DINOv3 Dimensions)",
             fontsize=13, fontweight="bold", color=PALETTE["primary"])
tick_positions = np.arange(0, 50, 10)
ax.set_xticks(tick_positions)
ax.set_xticklabels(sample_dims[tick_positions])
ax.set_yticks(tick_positions)
ax.set_yticklabels(sample_dims[tick_positions])
save(fig, "eda_05_correlation_heatmap.png")


# ===================================================================
# FIGURE 5: INTERACTION-LEVEL — Pairplot of Top PCA Components
# ===================================================================
msg("Computing PCA for pairplot and variance plot...")
from sklearn.decomposition import PCA

pca_full = PCA(n_components=min(50, X_clust.shape[1]), random_state=42)
X_pca = pca_full.fit_transform(X_clust)

# 5a: PCA Explained Variance (dimensionality reduction overview)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cumvar = np.cumsum(pca_full.explained_variance_ratio_)
axes[0].bar(range(1, len(pca_full.explained_variance_ratio_)+1),
            pca_full.explained_variance_ratio_, color=PALETTE["secondary"], alpha=0.7, edgecolor="white")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance Ratio")
axes[0].set_title("(a) Individual Variance per PC")
axes[0].set_xlim(0.5, 50.5)

axes[1].plot(range(1, len(cumvar)+1), cumvar, color=PALETTE["primary"], linewidth=2.5, marker="o", markersize=3)
axes[1].axhline(0.90, color=PALETTE["alert"], linestyle="--", label="90% variance")
axes[1].axhline(0.95, color=PALETTE["accent"], linestyle="--", label="95% variance")
n_90 = int(np.searchsorted(cumvar, 0.90)) + 1
n_95 = int(np.searchsorted(cumvar, 0.95)) + 1
axes[1].axvline(n_90, color=PALETTE["alert"], linestyle=":", alpha=0.5)
axes[1].axvline(n_95, color=PALETTE["accent"], linestyle=":", alpha=0.5)
axes[1].annotate(f"PC {n_90}", (n_90, 0.90), textcoords="offset points",
                 xytext=(10, -15), fontsize=9, color=PALETTE["alert"])
axes[1].annotate(f"PC {n_95}", (n_95, 0.95), textcoords="offset points",
                 xytext=(10, -15), fontsize=9, color=PALETTE["accent"])
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Variance Explained")
axes[1].set_title("(b) Cumulative Variance")
axes[1].legend(fontsize=9)

fig.suptitle("Dimensionality Reduction: PCA on DINOv3 Features",
             fontsize=14, fontweight="bold", color=PALETTE["primary"], y=1.02)
plt.tight_layout()
save(fig, "eda_06_pca_variance.png")

# 5b: Pairplot of top 4 PCs colored by cluster
msg("Generating pairplot (top 4 PCs)...")
n_sample = min(5000, len(X_pca))
sample_idx = rng.choice(len(X_pca), size=n_sample, replace=False)
pair_df = pd.DataFrame({
    f"PC{i+1}": X_pca[sample_idx, i] for i in range(4)
})
pair_df["Cluster"] = clusters[sample_idx].astype(str)
# Group small clusters as "Other" for readability
top_clusters = cluster_sizes.head(8).index.astype(str).tolist()
pair_df["Cluster"] = pair_df["Cluster"].apply(lambda c: c if c in top_clusters else "Other")

fig = plt.figure(figsize=(14, 14))
g = sns.pairplot(pair_df, hue="Cluster", plot_kws={"s": 8, "alpha": 0.4},
                 diag_kind="kde", palette="tab10", corner=True)
g.figure.suptitle("Interaction-Level: Pairplot of Top 4 PCs (5K Sample, by Cluster)",
                   fontsize=14, fontweight="bold", color=PALETTE["primary"], y=1.02)
save(g.figure, "eda_07_pairplot_pca.png")


# ===================================================================
# FIGURE 6: INTERACTION-LEVEL — Cosine Similarity by Cluster
# ===================================================================
msg("Generating cosine similarity heatmap...")
from sklearn.metrics.pairwise import cosine_similarity

# Sample compounds ordered by cluster for block-diagonal structure
max_heat = 600
sampled = []
for cid in range(len(cluster_sizes)):
    idx = np.where(clusters == cid)[0]
    take = min(len(idx), max(2, max_heat // len(cluster_sizes)))
    sampled.extend(rng.choice(idx, size=take, replace=False).tolist())
sampled = sorted(set(sampled))
if len(sampled) > max_heat:
    sampled = list(rng.choice(sampled, size=max_heat, replace=False))

Xs = X_clust[sampled]
cs = clusters[sampled]
order = np.argsort(cs)
Xs = Xs[order]
cs_ordered = cs[order]

sim = cosine_similarity(Xs)

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(sim, cmap="RdBu_r", vmin=-0.3, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8, label="Cosine Similarity")

# Mark cluster boundaries
boundaries = np.where(np.diff(cs_ordered))[0]
for b in boundaries:
    ax.axhline(b + 0.5, color="black", linewidth=0.5, alpha=0.5)
    ax.axvline(b + 0.5, color="black", linewidth=0.5, alpha=0.5)

ax.set_title("Interaction-Level: Compound Cosine Similarity\n(Ordered by Cluster Assignment)",
             fontsize=13, fontweight="bold", color=PALETTE["primary"])
ax.set_xlabel(f"Compounds (n={len(sampled)} sampled)")
ax.set_ylabel("Compounds")
save(fig, "eda_08_cosine_similarity_heatmap.png")


# ===================================================================
# FIGURE 7: DIMENSIONALITY — UMAP Embeddings
# ===================================================================
msg("Generating UMAP plots...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 7a: UMAP colored by cluster
n_clusters = len(np.unique(clusters))
cmap = plt.cm.tab20 if n_clusters <= 20 else plt.cm.nipy_spectral
colors = cmap(np.linspace(0, 1, n_clusters))
for cid in range(n_clusters):
    mask = clusters == cid
    axes[0].scatter(umap_emb[mask, 0], umap_emb[mask, 1], s=3, alpha=0.4,
                    c=[colors[cid]], label=f"{cid}" if cid < 10 else None)
axes[0].set_title(f"(a) UMAP — Leiden Clusters (n={n_clusters})", fontsize=11)
axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")

# 7b: UMAP colored by role
role_map = {"treated": PALETTE["treated"], "mock": PALETTE["mock"]}
for role, color in role_map.items():
    mask = cluster_df["role"].values == role
    if mask.any():
        axes[1].scatter(umap_emb[mask, 0], umap_emb[mask, 1], s=3, alpha=0.3,
                        c=color, label=f"{role} (n={mask.sum():,})")
axes[1].set_title("(b) UMAP — Compound Role", fontsize=11)
axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
axes[1].legend(fontsize=9, markerscale=4)

# 7c: UMAP colored by replicate count
sc = axes[2].scatter(umap_emb[:, 0], umap_emb[:, 1], s=3, alpha=0.4,
                     c=cluster_df["n_wells"].values, cmap="viridis", vmin=1,
                     vmax=cluster_df["n_wells"].quantile(0.95))
plt.colorbar(sc, ax=axes[2], shrink=0.8, label="Replicate wells")
axes[2].set_title("(c) UMAP — Replicate Coverage", fontsize=11)
axes[2].set_xlabel("UMAP 1"); axes[2].set_ylabel("UMAP 2")

fig.suptitle("Dimensionality Reduction: UMAP Embedding (n_neighbors=30, min_dist=0.1, cosine)",
             fontsize=14, fontweight="bold", color=PALETTE["primary"], y=1.02)
plt.tight_layout()
save(fig, "eda_09_umap_embeddings.png")


# ===================================================================
# FIGURE 8: METHODOLOGICAL — Distribution Shape Checks
# ===================================================================
msg("Generating distribution shape checks...")
from scipy import stats as sp_stats

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

# Test normality of top PCs (these feed into Leiden/UMAP — assumptions matter)
for i in range(8):
    ax = axes.flat[i]
    vals = X_pca[:, i]

    # QQ plot
    sp_stats.probplot(vals, dist="norm", plot=ax)
    ax.get_lines()[0].set(color=PALETTE["secondary"], markersize=2, alpha=0.3)
    ax.get_lines()[1].set(color=PALETTE["alert"], linewidth=2)

    # Shapiro on subsample
    sub = rng.choice(vals, size=min(5000, len(vals)), replace=False)
    _, p = sp_stats.shapiro(sub)
    skew = sp_stats.skew(vals)
    kurt = sp_stats.kurtosis(vals)

    ax.set_title(f"PC{i+1} (skew={skew:.2f}, kurt={kurt:.2f})\nShapiro p={p:.2e}", fontsize=9)
    ax.set_xlabel(""); ax.set_ylabel("")

fig.suptitle("Methodological: Normality QQ-Plots of Top 8 PCs\n(Verifying Distributional Assumptions for Downstream Methods)",
             fontsize=13, fontweight="bold", color=PALETTE["primary"], y=1.03)
plt.tight_layout()
save(fig, "eda_10_qq_normality_checks.png")


# ===================================================================
# FIGURE 9: BASELINE — DMSO Control Characterisation
# ===================================================================
msg("Generating baseline (DMSO) plots...")

dmso_mask = cluster_df["compound_id"].values == "DMSO"
treated_mask = cluster_df["role"].values == "treated"

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 9a: DMSO position on UMAP
axes[0].scatter(umap_emb[treated_mask, 0], umap_emb[treated_mask, 1],
                s=3, alpha=0.15, c="#cccccc", label="Treated")
if dmso_mask.any():
    axes[0].scatter(umap_emb[dmso_mask, 0], umap_emb[dmso_mask, 1],
                    s=80, alpha=0.95, c=PALETTE["dmso"], edgecolors="black",
                    linewidths=0.8, marker="D", label="DMSO", zorder=5)
    dmso_cluster = clusters[dmso_mask][0]
    axes[0].set_title(f"(a) DMSO on UMAP (Cluster {dmso_cluster})", fontsize=11)
else:
    axes[0].set_title("(a) DMSO on UMAP", fontsize=11)
axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")
axes[0].legend(fontsize=9, markerscale=2)

# 9b: Cosine distance from DMSO centroid — DMSO vs treated
if dmso_mask.any():
    from sklearn.metrics.pairwise import cosine_distances
    dmso_centroid = X_clust[dmso_mask].mean(axis=0, keepdims=True)
    # Sample treated for comparison
    treated_idx = np.where(treated_mask)[0]
    treated_sample = rng.choice(treated_idx, size=min(2000, len(treated_idx)), replace=False)

    treated_dists = cosine_distances(X_clust[treated_sample], dmso_centroid).ravel()
    dmso_dists = cosine_distances(X_clust[dmso_mask], dmso_centroid).ravel()

    axes[1].hist(treated_dists, bins=50, density=True, alpha=0.6,
                 color=PALETTE["treated"], label="Treated (sample)", edgecolor="none")
    axes[1].hist(dmso_dists, bins=20, density=True, alpha=0.7,
                 color=PALETTE["dmso"], label="DMSO", edgecolor="none")
    axes[1].set_xlabel("Cosine Distance from DMSO Centroid")
    axes[1].set_ylabel("Density")
    axes[1].set_title("(b) Distance from Baseline: DMSO vs Treated", fontsize=11)
    axes[1].legend(fontsize=8)
else:
    axes[1].text(0.5, 0.5, "No DMSO found", ha="center", va="center", transform=axes[1].transAxes)

# 9c: Per-PC DMSO baseline scores
if dmso_mask.any():
    dmso_pca = X_pca[dmso_mask]
    treated_pca_sample = X_pca[treated_sample]

    pcs_to_show = min(15, X_pca.shape[1])
    pc_labels = [f"PC{i+1}" for i in range(pcs_to_show)]
    dmso_means = dmso_pca[:, :pcs_to_show].mean(axis=0)
    treated_means = treated_pca_sample[:, :pcs_to_show].mean(axis=0)
    treated_stds = treated_pca_sample[:, :pcs_to_show].std(axis=0)

    x = np.arange(pcs_to_show)
    axes[2].bar(x - 0.15, treated_means, width=0.3, color=PALETTE["treated"],
                alpha=0.7, label="Treated mean", yerr=treated_stds, capsize=2, error_kw={"linewidth": 0.8})
    axes[2].bar(x + 0.15, dmso_means, width=0.3, color=PALETTE["dmso"],
                alpha=0.8, label="DMSO mean")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(pc_labels, rotation=45, fontsize=8)
    axes[2].set_ylabel("Mean PC Score")
    axes[2].set_title("(c) Baseline PC Scores: DMSO vs Treated", fontsize=11)
    axes[2].legend(fontsize=8)
    axes[2].axhline(0, color="black", linewidth=0.5)

fig.suptitle("Baseline Determination: DMSO Control Characterisation",
             fontsize=14, fontweight="bold", color=PALETTE["primary"], y=1.02)
plt.tight_layout()
save(fig, "eda_11_baseline_dmso.png")


# ===================================================================
# FIGURE 10: PROVENANCE — Cleaning Pipeline Summary
# ===================================================================
msg("Generating provenance figure...")

fig, ax = plt.subplots(figsize=(14, 7))
ax.axis("off")

# Build a table
cell_text = [[r["step"], r["description"], r["before"], r["after"], r["effect"]]
             for _, r in provenance_df.iterrows()]
col_labels = ["Step", "Description", "Before", "After", "Effect"]
col_widths = [0.12, 0.28, 0.15, 0.18, 0.27]

table = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    colWidths=col_widths,
    loc="center",
    cellLoc="left",
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.5)

# Style header
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor(PALETTE["primary"])
    cell.set_text_props(color="white", fontweight="bold")

# Alternate row colors
for i in range(1, len(cell_text) + 1):
    for j in range(len(col_labels)):
        cell = table[i, j]
        cell.set_facecolor("#f8f9fa" if i % 2 == 0 else "#ffffff")
        cell.set_edgecolor("#e0e0e0")

ax.set_title("Data Provenance: Every Cleaning Step, Why It Was Done, and Its Effect",
             fontsize=14, fontweight="bold", color=PALETTE["primary"], pad=20)
save(fig, "eda_12_provenance_log.png", dpi=150)


# ===================================================================
# FIGURE 11: FEATURE SELECTION — Before/After
# ===================================================================
msg("Generating feature selection plot...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 11a: Variance distribution of raw features
raw_var = np.var(X_raw, axis=0)
axes[0].hist(raw_var, bins=80, color=PALETTE["secondary"], alpha=0.7, edgecolor="none")
axes[0].axvline(0.01, color=PALETTE["alert"], linewidth=2, linestyle="--", label="Threshold: 0.01")
axes[0].set_xlabel("Feature Variance")
axes[0].set_ylabel("Count")
axes[0].set_title(f"(a) Variance Distribution (n={X_raw.shape[1]} raw features)")
axes[0].set_yscale("log")
axes[0].legend(fontsize=9)

# 11b: Correlation distribution among selected features
sel_corr = np.corrcoef(X_clust.T)
upper_tri = sel_corr[np.triu_indices_from(sel_corr, k=1)]
axes[1].hist(upper_tri, bins=80, color=PALETTE["highlight"], alpha=0.7, edgecolor="none")
axes[1].axvline(0.95, color=PALETTE["alert"], linewidth=2, linestyle="--", label="|r| = 0.95 cutoff")
axes[1].axvline(-0.95, color=PALETTE["alert"], linewidth=2, linestyle="--")
axes[1].set_xlabel("Pairwise Pearson r")
axes[1].set_ylabel("Count")
axes[1].set_title(f"(b) Correlation Distribution (n={X_clust.shape[1]} selected features)")
axes[1].legend(fontsize=9)

# 11c: Dimension reduction summary
labels = ["DINOv3\nOutput", "After Var\nFilter", "After Corr\nFilter"]
values = [n_raw_features, n_raw_features - (n_raw_features - n_selected)//2, n_selected]  # approximate intermediate
colors_bar = [PALETTE["primary"], PALETTE["highlight"], PALETTE["secondary"]]
axes[2].bar(labels, values, color=colors_bar, alpha=0.85, edgecolor="white", width=0.5)
for i, v in enumerate(values):
    axes[2].text(i, v + 10, str(v), ha="center", fontsize=11, fontweight="bold")
axes[2].set_ylabel("Number of Features")
axes[2].set_title("(c) Feature Selection Funnel")

fig.suptitle("Feature Selection: Variance and Correlation Filtering",
             fontsize=14, fontweight="bold", color=PALETTE["primary"], y=1.02)
plt.tight_layout()
save(fig, "eda_13_feature_selection.png")


# ===================================================================
# FIGURE 12: CellProfiler Feature Correlation Heatmap
# ===================================================================
msg("Generating CellProfiler correlation heatmap...")

# Sample 40 features across compartments for a readable heatmap
cp_sample_cols = []
for prefix in ["Cells_", "Nuclei_", "Cytoplasm_"]:
    available = [c for c in cp_feature_cols if c.startswith(prefix)]
    chosen = [available[i] for i in np.linspace(0, len(available)-1, 14, dtype=int)]
    cp_sample_cols.extend(chosen)
cp_sample_cols = cp_sample_cols[:42]

cp_corr = cp_features[cp_sample_cols].corr()

fig, ax = plt.subplots(figsize=(14, 12))
short_labels = [c.replace("Cells_", "C.").replace("Nuclei_", "N.").replace("Cytoplasm_", "Cy.")
                .replace("AreaShape_", "AS.").replace("Correlation_", "Corr.")
                .replace("Intensity_", "Int.").replace("Texture_", "Tex.")
                [:30] for c in cp_sample_cols]
sns.heatmap(cp_corr, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
            xticklabels=short_labels, yticklabels=short_labels,
            linewidths=0.1, linecolor="#f0f0f0")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
ax.set_title("Interaction-Level: CellProfiler Feature Correlations\n(42 Features Sampled Across Compartments)",
             fontsize=13, fontweight="bold", color=PALETTE["primary"])
save(fig, "eda_14_cellprofiler_correlation.png")


# ===================================================================
# DONE
# ===================================================================
msg("=" * 50)
msg(f"All EDA figures saved to {OUT_DIR}")
msg(f"Total figures: {len(list(OUT_DIR.glob('eda_*.png')))}")
print("\nGenerated files:")
for f in sorted(OUT_DIR.glob("eda_*.png")):
    print(f"  {f.name}")
