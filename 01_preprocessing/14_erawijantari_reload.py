

# --------------------------------------------------------------------------
"""
Step 14 — Reload the full ERAWIJANTARI dataset (Healthy + Gastrectomy) for within-cohort comparisons; re-run preprocessing pipeline.
"""


# =====================================================================
# Reload FULL ERAWIJANTARI dataset (Healthy + Gastrectomy, pre-dedup)
# so that within-dataset Healthy vs Gastrectomy analysis is possible.
# The dedup only removed ERAWIJANTARI Healthy from the *combined* pool;
# for within-dataset comparison we need them back.
# =====================================================================

eraw_ds = 'ERAWIJANTARI-GASTRIC-CANCER-2020'

# --- Reload original files ---
eraw_orig = {}
for ft in ['metadata', 'mtb', 'mtb.map', 'species']:
    fpath = DATA_DIR / f"{eraw_ds} {ft}.tsv"
    eraw_orig[ft] = pd.read_csv(fpath, sep="\t")

print(f"Original ERAWIJANTARI: {len(eraw_orig['metadata'])} samples")
print(f"  Study groups: {eraw_orig['metadata']['Study.Group'].value_counts().to_dict()}")

# --- Harmonize metadata ---
harmonized_meta[eraw_ds] = harmonize_metadata(eraw_orig['metadata'], eraw_ds)

# --- Polyamine columns (already identified, but refresh from original map) ---
polyamine_columns[eraw_ds] = find_polyamine_columns(
    eraw_orig['mtb.map'], eraw_orig['mtb'], eraw_ds
)

# --- Reduce species names ---
eraw_reduced, _ = reduce_species_names(eraw_orig['species'])

# --- Prevalence filtering ---
protected_mtb = list(polyamine_columns[eraw_ds].values())
eraw_spc_filtered = filter_by_prevalence(eraw_reduced, threshold=SPECIES_PREVALENCE)
eraw_mtb_filtered = filter_by_prevalence(
    eraw_orig['mtb'], threshold=MTB_PREVALENCE, protected_cols=protected_mtb
)

# --- Variance filtering ---
eraw_spc_var = filter_near_zero_variance(eraw_spc_filtered)
eraw_mtb_var = filter_near_zero_variance(eraw_mtb_filtered)

# --- Transform ---
transformed_species[eraw_ds] = clr_transform(eraw_spc_var, pseudocount=1e-6)
transformed_mtb[eraw_ds] = log_transform(eraw_mtb_var)

# --- Refresh polyamine columns against filtered metabolites ---
polyamine_columns[eraw_ds] = find_polyamine_columns(
    eraw_orig['mtb.map'], eraw_mtb_filtered, eraw_ds
)

print(f"\nFull ERAWIJANTARI preprocessed:")
print(f"  Samples: {len(transformed_species[eraw_ds])}")
print(f"  Species: {len(transformed_species[eraw_ds].columns) - 1}")
print(f"  Metabolites: {len(transformed_mtb[eraw_ds].columns) - 1}")
print(f"  Polyamines: {len(polyamine_columns[eraw_ds])}")
print(f"  Study groups: {harmonized_meta[eraw_ds]['Study.Group'].value_counts().to_dict()}")

# ── PCA visualisation: ERAWIJANTARI with Healthy + Gastrectomy ──────────────
print("\nGenerating PCA for ERAWIJANTARI (full cohort: Healthy + Gastrectomy)...")
(CRC_RESULTS_DIR / 'figures' / 'pca').mkdir(parents=True, exist_ok=True)

for data_label, df_dict in [('species', transformed_species),
                             ('metabolites', transformed_mtb)]:
    if eraw_ds not in df_dict:
        print(f"  {data_label}: data not found, skipping")
        continue
    transform_label = 'CLR' if data_label == 'species' else 'log2'
    fig, ax = plot_pca(
        df_dict[eraw_ds],
        harmonized_meta[eraw_ds],
        title=(f'{eraw_ds}\n'
               f'{data_label.capitalize()} ({transform_label}-transformed) — All Groups'),
        color_col='Study.Group',
        figsize=(10, 7),
        save_path=(CRC_RESULTS_DIR / 'figures' / 'pca'
                   / f'{eraw_ds}_{data_label}_pca_full.png')
    )
    plt.show()
    plt.close()
    print(f"  Saved: figures/pca/{eraw_ds}_{data_label}_pca_full.png")
