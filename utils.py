"""
Shared utility functions for the microbiome-metabolome analysis pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import entropy, mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# ============================================================================
# CONSTANTS
# ============================================================================

DATA_DIR = Path(r"c:\Users\andna\Documents\KI\Data")
RESULTS_DIR = Path(r"c:\Users\andna\Documents\KI\Results")

DATASETS = [
    "ERAWIJANTARI-GASTRIC-CANCER-2020",
    "FRANZOSA-IBD-2019",
    "iHMP-IBDMDB-2019",
    "MARS-IBS-2020",
    "WANG_ESRD_2020",
    "YACHIDA-CRC-2019",
]

FILE_TYPES = ["metadata", "mtb", "mtb.map", "species"]

# Datasets that contain polyamine metabolites
POLYAMINE_DATASETS = [
    "ERAWIJANTARI-GASTRIC-CANCER-2020",
    "FRANZOSA-IBD-2019",
    "iHMP-IBDMDB-2019",
    "YACHIDA-CRC-2019",
]

# Polyamine KEGG IDs and names
POLYAMINE_KEGG = {
    "C00134": "Putrescine",
    "C00315": "Spermidine",
    "C00750": "Spermine",
    "C02714": "N-Acetylputrescine",
    "C00612": "N1-Acetylspermidine",
    "C01029": "N8-Acetylspermidine",
    "C02567": "N1-Acetylspermine",
    "C03413": "N1,N12-Diacetylspermine",
}

# Case/control comparisons per dataset
COMPARISONS = {
    "ERAWIJANTARI-GASTRIC-CANCER-2020": [("Healthy", "Gastrectomy")],
    "FRANZOSA-IBD-2019": [("Control", "CD"), ("Control", "UC")],
    "iHMP-IBDMDB-2019": [("nonIBD", "CD"), ("nonIBD", "UC")],
    "MARS-IBS-2020": [("H", "D"), ("H", "C")],
    "WANG_ESRD_2020": [("Control", "ESRD")],
    "YACHIDA-CRC-2019": [
        ("Healthy", "Stage_I_II"),
        ("Healthy", "Stage_III_IV"),
        ("Healthy", "Stage_0"),
        ("Healthy", "MP"),
        ("Healthy", "HS"),
    ],
}

# Study group numeric encoding
STUDY_GROUP_MAP = {
    "Healthy": 0, "HP": 0, "Control": 0, "nonIBD": 0, "H": 0,
    "Stage_0": 1,
    "Stage_I_II": 2,
    "MP": 3, "HS": 3,
    "Stage_III_IV": 4,
    "UC": 5,
    "CD": 6,
    "Gastrectomy": 7,
    "ESRD": 8,
    "D": 9,
    "C": 10,
    "Adenoma": 1,
    "Carcinoma": 4,
    "CRC": 4,
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_datasets(data_dir=DATA_DIR, datasets=DATASETS):
    """Load all datasets into a nested dict: data[dataset_name][file_type] -> DataFrame."""
    data = {}
    for ds in datasets:
        data[ds] = {}
        for ft in FILE_TYPES:
            fpath = data_dir / f"{ds} {ft}.tsv"
            if fpath.exists():
                data[ds][ft] = pd.read_csv(fpath, sep="\t")
            else:
                warnings.warn(f"File not found: {fpath}")
                data[ds][ft] = None
    return data


def load_species_counts(data_dir=DATA_DIR, datasets=DATASETS):
    """Stub for loading species count data (not yet available).

    When species_counts files are downloaded, this function will load them
    into the same dict structure as load_all_datasets but with 'species_counts' key.
    """
    counts = {}
    for ds in datasets:
        fpath = data_dir / f"{ds} species_counts.tsv"
        if fpath.exists():
            counts[ds] = pd.read_csv(fpath, sep="\t")
        else:
            counts[ds] = None
    return counts


# ============================================================================
# METADATA HARMONIZATION
# ============================================================================

def harmonize_metadata(meta_df, dataset_name):
    """Map dataset-specific column names to a common schema."""
    hm = pd.DataFrame()
    hm["Dataset"] = dataset_name
    hm["Sample"] = meta_df["Sample"].values
    hm["Subject"] = meta_df["Subject"].values

    # SINHA_CRC_2016 stores Study.Group as integers (0=Healthy, 1=CRC)
    sg = meta_df["Study.Group"].copy()
    if dataset_name == "SINHA_CRC_2016":
        sg = sg.map({0: "Healthy", 1: "CRC"})

    hm["Study.Group"] = sg.values
    hm["Study.Group.Numeric"] = sg.map(STUDY_GROUP_MAP).values

    # Age — Kim_adenomas_2020 stores age as range strings ('60-69', '70+')
    _AGE_MIDPOINTS = {
        'Under 20': 17.5, '20-29': 24.5, '30-39': 34.5,
        '40-49': 44.5, '50-59': 54.5, '60-69': 64.5, '70+': 75.0,
    }
    if "consent_age" in meta_df.columns:
        hm["Age"] = pd.to_numeric(meta_df["consent_age"], errors="coerce").values
    elif "Age" in meta_df.columns:
        if dataset_name == "Kim_adenomas_2020":
            hm["Age"] = meta_df["Age"].map(_AGE_MIDPOINTS).values
        else:
            hm["Age"] = pd.to_numeric(meta_df["Age"], errors="coerce").values
    else:
        hm["Age"] = np.nan

    # Gender
    if "Gender" in meta_df.columns:
        hm["Gender"] = meta_df["Gender"].values
    else:
        hm["Gender"] = np.nan

    # BMI
    if "BMI_at_baseline" in meta_df.columns:
        hm["BMI"] = pd.to_numeric(meta_df["BMI_at_baseline"], errors="coerce").values
    elif "BMI" in meta_df.columns:
        hm["BMI"] = pd.to_numeric(meta_df["BMI"], errors="coerce").values
    else:
        hm["BMI"] = np.nan

    return hm


# ============================================================================
# POLYAMINE IDENTIFICATION
# ============================================================================

def find_polyamine_columns(mtb_map_df, mtb_df, dataset_name):
    """Find polyamine metabolite columns in a dataset's mtb matrix.

    Returns a dict: {KEGG_ID: column_name_in_mtb} for found polyamines.
    """
    found = {}
    mtb_cols = set(mtb_df.columns) - {"Sample"}

    # Strategy 1: Match via KEGG in mtb.map
    kegg_col = "KEGG"
    compound_col = "Compound"

    if kegg_col in mtb_map_df.columns and compound_col in mtb_map_df.columns:
        for kegg_id, name in POLYAMINE_KEGG.items():
            matches = mtb_map_df[mtb_map_df[kegg_col] == kegg_id]
            for _, row in matches.iterrows():
                compound = row[compound_col]
                if compound in mtb_cols:
                    found[kegg_id] = compound

    # Strategy 2: Direct column name match (for ERAWIJANTARI/YACHIDA style: C00134_Putrescine...)
    for kegg_id, name in POLYAMINE_KEGG.items():
        if kegg_id in found:
            continue
        for col in mtb_cols:
            if col.startswith(kegg_id + "_"):
                found[kegg_id] = col
                break

    return found


# ============================================================================
# TAXONOMY PARSING
# ============================================================================

def parse_taxonomy(full_string):
    """Parse 'd__X;p__Y;...;s__Z' into a dict of ranks."""
    rank_map = {
        "d": "domain", "p": "phylum", "c": "class",
        "o": "order", "f": "family", "g": "genus", "s": "species",
    }
    ranks = {}
    for part in full_string.split(";"):
        if "__" in part:
            prefix, name = part.split("__", 1)
            ranks[rank_map.get(prefix, prefix)] = name
    return ranks


def extract_species_name(full_string):
    """Extract species name from full GTDB taxonomy string.
    Falls back to genus name for datasets without species-level annotation."""
    parsed = parse_taxonomy(full_string)
    return parsed.get("species", parsed.get("genus", full_string))


def extract_genus(full_string):
    """Extract genus name from full GTDB taxonomy string."""
    parsed = parse_taxonomy(full_string)
    return parsed.get("genus", "Unknown")


def reduce_species_names(species_df):
    """Reduce full taxonomy column names to species-level, summing duplicates.

    Returns:
        reduced_df: DataFrame with short species names as columns
        name_mapping: dict {short_name: [list of original full taxonomy strings]}
    """
    col_map = {}
    col_indices = {}
    for i, col in enumerate(species_df.columns):
        if col == "Sample":
            continue
        short = extract_species_name(col)
        col_map.setdefault(short, []).append(col)
        col_indices.setdefault(short, []).append(i)

    # Use positional indexing to avoid label-based KeyError with special column names
    arr = species_df.values
    data = {"Sample": species_df["Sample"].values}
    for short, indices in col_indices.items():
        if len(indices) == 1:
            data[short] = arr[:, indices[0]]
        else:
            data[short] = arr[:, indices].astype(float).sum(axis=1)

    reduced = pd.DataFrame(data)
    return reduced, col_map


def aggregate_to_genus(species_df):
    """Aggregate species-level data to genus by summing abundances."""
    genus_map = {}
    for col in species_df.columns:
        if col == "Sample":
            continue
        genus = extract_genus(col) if ";" in col else col.split()[0] if " " in col else col
        genus_map.setdefault(genus, []).append(col)

    agg = pd.DataFrame({"Sample": species_df["Sample"].values})
    for genus, cols in genus_map.items():
        agg[genus] = species_df[cols].sum(axis=1).values
    return agg


# ============================================================================
# QC FUNCTIONS
# ============================================================================

def compute_sample_qc(df, data_type="metabolite"):
    """Compute sample-level QC metrics.

    Args:
        df:        DataFrame with a 'Sample' column and feature columns.
        data_type: 'metabolite' (default) or 'species'.
                   Pass data_type='species' when analysing species data so that
                   Shannon alpha-diversity (H') is included in the output.
                   Without this flag the 'shannon_diversity' column is absent.

    Returns:
        DataFrame with per-sample QC metrics. Species mode adds:
          shannon_diversity : Shannon H' computed on positive-valued features.
    """
    feature_cols = [c for c in df.columns if c != "Sample"]
    values = df[feature_cols]

    qc = pd.DataFrame({"Sample": df["Sample"].values})
    qc["total_signal"] = values.sum(axis=1).values
    qc["n_detected"] = (values > 0).sum(axis=1).values
    qc["n_features"] = len(feature_cols)
    qc["detection_rate"] = qc["n_detected"] / qc["n_features"]

    if data_type == "species":
        qc["shannon_diversity"] = values.apply(
            lambda row: entropy(row[row > 0]) if (row > 0).any() else 0, axis=1
        ).values

    return qc


def compute_feature_qc(df):
    """Compute feature-level QC metrics.

    Returns:
        DataFrame with detection_rate, mean, std, cv per feature
    """
    feature_cols = [c for c in df.columns if c != "Sample"]
    values = df[feature_cols]

    fqc = pd.DataFrame({"Feature": feature_cols})
    fqc["detection_rate"] = (values > 0).mean().values
    fqc["mean"] = values.mean().values
    fqc["std"] = values.std().values
    fqc["cv"] = np.where(fqc["mean"] > 0, fqc["std"] / fqc["mean"], np.nan)
    fqc["variance"] = values.var().values

    return fqc


# ============================================================================
# FILTERING
# ============================================================================

def filter_by_prevalence(df, threshold=0.1, protected_cols=None):
    """Remove features detected in fewer than threshold fraction of samples.

    Notebook conventions (pass explicitly — do NOT rely on the default):
      - Species     : threshold=0.20  (20 % prevalence)
      - Metabolites : threshold=0.15  (15 % prevalence)

    Protected columns are never removed (e.g., polyamine metabolites).
    """
    feature_cols = [c for c in df.columns if c != "Sample"]
    prevalence = (df[feature_cols] > 0).mean()
    keep = prevalence[prevalence >= threshold].index.tolist()

    if protected_cols:
        for pc in protected_cols:
            if pc in feature_cols and pc not in keep:
                keep.append(pc)

    return df[["Sample"] + keep]


def filter_near_zero_variance(df, threshold=1e-10):
    """Remove features with variance below threshold."""
    feature_cols = [c for c in df.columns if c != "Sample"]
    variances = df[feature_cols].var()
    keep = variances[variances > threshold].index.tolist()
    return df[["Sample"] + keep]


# ============================================================================
# TRANSFORMATION & NORMALIZATION
# ============================================================================

def clr_transform(df, pseudocount=1e-6):
    """Centered log-ratio transformation for compositional (species) data."""
    feature_cols = [c for c in df.columns if c != "Sample"]
    X = df[feature_cols].values.astype(float) + pseudocount
    log_X = np.log(X)
    geometric_mean = log_X.mean(axis=1, keepdims=True)
    clr_values = log_X - geometric_mean

    result = pd.DataFrame(clr_values, columns=feature_cols)
    result.insert(0, "Sample", df["Sample"].values)
    return result


def log_transform(df, pseudocount=1.0, base=2):
    """Log transformation for abundance data (metabolites or species).

    Notebook conventions:
      - Metabolites : base=2,  pseudocount=1.0   (log2, default)
      - Species     : use clr_transform() instead (CLR is the standard approach)
      - This function is used for metabolites only (base=2, pseudocount=1.0)

    Args:
        df:           DataFrame with a 'Sample' column and feature columns.
        pseudocount:  Small constant added before log to handle zeros.
        base:         Logarithm base (2, 10, or natural log for any other value).
    """
    feature_cols = [c for c in df.columns if c != "Sample"]
    X = df[feature_cols].values.astype(float)
    if base == 2:
        log_X = np.log2(X + pseudocount)
    elif base == 10:
        log_X = np.log10(X + pseudocount)
    else:
        log_X = np.log(X + pseudocount)

    result = pd.DataFrame(log_X, columns=feature_cols)
    result.insert(0, "Sample", df["Sample"].values)
    return result


def quantile_normalize(df):
    """Quantile normalization across samples (for cross-dataset comparisons)."""
    from scipy.stats import rankdata

    feature_cols = [c for c in df.columns if c != "Sample"]
    X = df[feature_cols].values.astype(float)

    sorted_means = np.sort(X, axis=0).mean(axis=1)
    ranks = np.apply_along_axis(lambda col: rankdata(col, method="min") - 1, 0, X)
    normalized = sorted_means[ranks.astype(int)]

    result = pd.DataFrame(normalized, columns=feature_cols)
    result.insert(0, "Sample", df["Sample"].values)
    return result


# Stub for count-based normalization (future use)
def normalize_counts(counts_df, method="tss"):
    """Normalize species count data. Placeholder for future count-based analyses.

    Args:
        counts_df: DataFrame with raw counts (Sample column + species columns)
        method: 'tss' (total sum scaling), 'rarefaction', 'css' (cumulative sum scaling)

    Returns:
        Normalized DataFrame
    """
    feature_cols = [c for c in counts_df.columns if c != "Sample"]
    if method == "tss":
        row_sums = counts_df[feature_cols].sum(axis=1)
        normalized = counts_df[feature_cols].div(row_sums, axis=0)
        result = pd.DataFrame(normalized, columns=feature_cols)
        result.insert(0, "Sample", counts_df["Sample"].values)
        return result
    else:
        raise NotImplementedError(f"Method '{method}' not yet implemented. "
                                  "Add rarefaction or CSS when counts data is available.")


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def differential_abundance(df, meta, group1, group2, group_col="Study.Group"):
    """Mann-Whitney U test for each feature between two groups with FDR correction.

    Returns:
        DataFrame with Feature, MeanGroup1, MeanGroup2, Log2FC, Statistic, PValue,
        CliffsD, QValue, Significant columns.
    """
    merged = df.merge(meta[["Sample", group_col]], on="Sample")
    g1 = merged[merged[group_col] == group1]
    g2 = merged[merged[group_col] == group2]

    feature_cols = [c for c in df.columns if c != "Sample"]
    results = []

    for feat in feature_cols:
        vals1 = g1[feat].dropna()
        vals2 = g2[feat].dropna()

        if len(vals1) < 3 or len(vals2) < 3:
            continue

        try:
            stat, pval = mannwhitneyu(vals1, vals2, alternative="two-sided")
        except ValueError:
            continue

        n1, n2 = len(vals1), len(vals2)
        cliffs_d = (2 * stat / (n1 * n2)) - 1

        mean1 = vals1.mean()
        mean2 = vals2.mean()
        log2fc = np.log2((mean2 + 1e-10) / (mean1 + 1e-10))

        results.append({
            "Feature": feat,
            "MeanGroup1": mean1,
            "MeanGroup2": mean2,
            "Log2FC": log2fc,
            "Statistic": stat,
            "PValue": pval,
            "CliffsD": cliffs_d,
        })

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    reject, qvals, _, _ = multipletests(results_df["PValue"], method="fdr_bh")
    results_df["QValue"] = qvals
    results_df["Significant"] = reject

    return results_df.sort_values("QValue")


def compute_correlations(species_df, mtb_df, target_metabolites=None):
    """Spearman correlations between species and metabolites with FDR correction.

    Args:
        species_df: Transformed species DataFrame (Sample + species columns)
        mtb_df: Transformed metabolite DataFrame (Sample + metabolite columns)
        target_metabolites: Optional list of metabolite columns to focus on (e.g., polyamines)

    Returns:
        DataFrame with Species, Metabolite, Rho, PValue, QValue columns
    """
    merged = species_df.merge(mtb_df, on="Sample", suffixes=("_spc", "_mtb"))
    spc_cols = [c for c in species_df.columns if c != "Sample"]

    if target_metabolites:
        mtb_cols = [c for c in target_metabolites if c in mtb_df.columns]
    else:
        mtb_cols = [c for c in mtb_df.columns if c != "Sample"]

    results = []
    for spc in spc_cols:
        for mtb in mtb_cols:
            spc_key = spc + "_spc" if spc + "_spc" in merged.columns else spc
            mtb_key = mtb + "_mtb" if mtb + "_mtb" in merged.columns else mtb
            x = merged[spc_key].values
            y = merged[mtb_key].values
            if x.std() == 0 or y.std() == 0:
                continue
            try:
                rho, pval = spearmanr(x, y)
                if not np.isnan(rho):
                    results.append({
                        "Species": spc, "Metabolite": mtb,
                        "Rho": rho, "PValue": pval,
                    })
            except Exception:
                continue

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    _, qvals, _, _ = multipletests(results_df["PValue"], method="fdr_bh")
    results_df["QValue"] = qvals

    return results_df


def find_consistent_correlations(corr_dicts, min_datasets=2, q_threshold=0.1):
    """Find species-metabolite pairs significant in >= min_datasets.

    Args:
        corr_dicts: dict {dataset_name: correlation_results_df}
        min_datasets: minimum number of datasets for consistency
        q_threshold: FDR threshold for significance

    Returns:
        DataFrame with Species, Metabolite, n_datasets, mean_rho, datasets
    """
    all_results = []
    for ds, corr_df in corr_dicts.items():
        if corr_df is None or corr_df.empty:
            continue
        sig = corr_df[corr_df["QValue"] < q_threshold].copy()
        sig["Dataset"] = ds
        all_results.append(sig)

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    counts = combined.groupby(["Species", "Metabolite"]).agg(
        n_datasets=("Dataset", "nunique"),
        mean_rho=("Rho", "mean"),
        datasets=("Dataset", lambda x: list(x)),
    ).reset_index()

    return counts[counts["n_datasets"] >= min_datasets].sort_values(
        "n_datasets", ascending=False
    )


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def _draw_confidence_ellipse(ax, x, y, color, n_std=1.96, **kwargs):
    """Draw a 95% confidence ellipse for 2D data."""
    from matplotlib.patches import Ellipse
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)), width=width, height=height,
                      angle=angle, facecolor=color, alpha=0.15,
                      edgecolor=color, linewidth=1.5, linestyle='--', **kwargs)
    ax.add_patch(ellipse)


def plot_pca(df, meta, title, color_col="Study.Group", n_components=2,
             figsize=(12, 8), save_path=None, show_ellipses=True,
             show_variance_bar=True):
    """PCA scatter plot colored by a metadata variable with confidence ellipses."""
    from matplotlib.gridspec import GridSpec

    feature_cols = [c for c in df.columns if c != "Sample"]
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_comp_full = min(10, min(X.shape))
    pca_full = PCA(n_components=n_comp_full)
    pca_full.fit(X)

    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X)

    plot_df = pd.DataFrame({
        "PC1": coords[:, 0], "PC2": coords[:, 1],
        "Sample": df["Sample"].values,
    })
    plot_df = plot_df.merge(meta[["Sample", color_col]], on="Sample", how="left")

    if show_variance_bar:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 5, figure=fig, width_ratios=[4, 0.1, 0.8, 0.1, 0.1])
        ax = fig.add_subplot(gs[0, 0])
        ax_var = fig.add_subplot(gs[0, 2])
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_var = None

    palette = sns.color_palette("Set2", n_colors=plot_df[color_col].nunique())
    groups = sorted(plot_df[color_col].dropna().unique())
    color_map = {g: palette[i] for i, g in enumerate(groups)}

    for group in groups:
        mask = plot_df[color_col] == group
        subset = plot_df[mask]
        ax.scatter(subset["PC1"], subset["PC2"], c=[color_map[group]],
                   label=f"{group} (n={len(subset)})", alpha=0.7, s=50,
                   edgecolors='white', linewidth=0.5)
        if show_ellipses and len(subset) >= 3:
            _draw_confidence_ellipse(ax, subset["PC1"].values,
                                     subset["PC2"].values, color_map[group])

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.0, 1), loc="upper left", fontsize=8,
              framealpha=0.9)

    if ax_var is not None:
        var_ratios = pca_full.explained_variance_ratio_[:n_comp_full] * 100
        ax_var.barh(range(len(var_ratios)), var_ratios,
                           color='steelblue', edgecolor='white')
        ax_var.set_yticks(range(len(var_ratios)))
        ax_var.set_yticklabels([f"PC{i+1}" for i in range(len(var_ratios))],
                               fontsize=7)
        ax_var.set_xlabel("% Var", fontsize=8)
        ax_var.set_title("Scree", fontsize=9)
        ax_var.invert_yaxis()
        for i, v in enumerate(var_ratios):
            ax_var.text(v + 0.3, i, f"{v:.1f}", va='center', fontsize=6)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def volcano_plot(results_df, title, fc_thresh=1.0, q_thresh=0.05,
                 highlight_features=None, figsize=(10, 8), save_path=None):
    """Volcano plot from differential abundance results."""
    fig, ax = plt.subplots(figsize=figsize)

    neg_log_q = -np.log10(results_df["QValue"].clip(lower=1e-300))

    colors = np.where(
        (results_df["QValue"] < q_thresh) & (results_df["Log2FC"].abs() > fc_thresh),
        "red", "grey"
    )

    ax.scatter(results_df["Log2FC"], neg_log_q, c=colors, alpha=0.5, s=10)
    ax.axhline(-np.log10(q_thresh), ls="--", color="blue", alpha=0.5)
    ax.axvline(fc_thresh, ls="--", color="blue", alpha=0.5)
    ax.axvline(-fc_thresh, ls="--", color="blue", alpha=0.5)

    if highlight_features:
        for feat in highlight_features:
            row = results_df[results_df["Feature"] == feat]
            if not row.empty:
                x = row["Log2FC"].values[0]
                y = -np.log10(row["QValue"].values[0])
                ax.annotate(feat, (x, y), fontsize=7, alpha=0.8)

    ax.set_xlabel("Log2 Fold Change")
    ax.set_ylabel("-log10(Q-value)")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_detection_histogram(feature_qc, title, figsize=(10, 5), save_path=None):
    """Histogram of feature detection rates."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(feature_qc["detection_rate"], bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0.1, color="red", ls="--", label="10% threshold")
    ax.set_xlabel("Detection Rate (fraction of samples)")
    ax.set_ylabel("Number of Features")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_correlation_heatmap(corr_df, title, top_n=20, figsize=(14, 10), save_path=None):
    """Heatmap of species-metabolite correlations."""
    if corr_df.empty:
        return None, None

    top = corr_df.nsmallest(top_n * 5, "QValue") if len(corr_df) > top_n * 5 else corr_df
    pivot = top.pivot_table(index="Species", columns="Metabolite", values="Rho", aggfunc="first")

    if pivot.empty:
        return None, None

    # Limit to manageable size
    if pivot.shape[0] > top_n:
        row_vars = pivot.var(axis=1).nlargest(top_n).index
        pivot = pivot.loc[row_vars]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                ax=ax, xticklabels=True, yticklabels=True)
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_sample_alignment(data, dataset_name):
    """Check that Sample IDs align across metadata, mtb, and species files."""
    meta_samples = set(data[dataset_name]["metadata"]["Sample"])
    mtb_samples = set(data[dataset_name]["mtb"]["Sample"])
    spc_samples = set(data[dataset_name]["species"]["Sample"])

    report = {
        "dataset": dataset_name,
        "meta_n": len(meta_samples),
        "mtb_n": len(mtb_samples),
        "species_n": len(spc_samples),
        "meta_only": meta_samples - mtb_samples - spc_samples,
        "mtb_only": mtb_samples - meta_samples,
        "spc_only": spc_samples - meta_samples,
        "common": meta_samples & mtb_samples & spc_samples,
    }
    return report


def validate_metabolite_alignment(data, dataset_name):
    """Check that mtb column names match Compound in mtb.map."""
    mtb_cols = set(data[dataset_name]["mtb"].columns) - {"Sample"}
    map_compounds = set(data[dataset_name]["mtb.map"]["Compound"])

    return {
        "dataset": dataset_name,
        "mtb_features": len(mtb_cols),
        "map_compounds": len(map_compounds),
        "in_mtb_not_map": mtb_cols - map_compounds,
        "in_map_not_mtb": map_compounds - mtb_cols,
    }


def print_dataset_summary(data):
    """Print a summary table of all loaded datasets."""
    rows = []
    for ds in DATASETS:
        if ds not in data:
            continue
        row = {
            "Dataset": ds,
            "Samples (meta)": len(data[ds]["metadata"]),
            "Samples (mtb)": len(data[ds]["mtb"]),
            "Samples (species)": len(data[ds]["species"]),
            "Metabolites": len(data[ds]["mtb"].columns) - 1,
            "Species": len(data[ds]["species"].columns) - 1,
            "Map entries": len(data[ds]["mtb.map"]),
        }
        rows.append(row)
    return pd.DataFrame(rows)
