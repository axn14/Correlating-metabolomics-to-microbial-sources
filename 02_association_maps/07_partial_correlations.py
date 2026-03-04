

# --------------------------------------------------------------------------"""
Partial correlations — Spearman correlations controlling for Age, BMI, Gender, and dataset-specific confounders (medications, smoking, alcohol) via pingouin.
"""


# Partial correlations controlling for confounders
# Confounders: Age, BMI, Gender, and medication/lifestyle where available
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False
    print("pingouin not installed -- skipping partial correlations.")

partial_corr_results = {}

if HAS_PINGOUIN:
    from statsmodels.stats.multitest import multipletests

    # Dataset-specific confounder definitions
    # ERAWIJANTARI has: Gastric acid medication, Analgesic, Anticoagulant, DiabetesMed
    # YACHIDA has: Brinkman Index (smoking), Alcohol
    EXTRA_CONFOUNDERS = {
        'ERAWIJANTARI-GASTRIC-CANCER-2020': [
            'Gastric acid medication', 'Analgesic', 'Anticoagulant', 'DiabetesMed',
        ],
        'YACHIDA-CRC-2019': [
            'Brinkman Index', 'Alcohol',
        ],

    }

    for ds in CRC_DATASETS:
        meta = harmonized_meta[ds].copy()

        # Load raw metadata for extra confounders not in harmonized_meta
        raw_meta = pd.read_csv(DATA_DIR / f'{ds} metadata.tsv', sep='\t')
        extra_cols = EXTRA_CONFOUNDERS.get(ds, [])
        for col in extra_cols:
            if col in raw_meta.columns:
                meta = meta.merge(
                    raw_meta[['Sample', col]], on='Sample', how='left'
                )

        # Build covariate list
        covariates = []

        # Standard confounders
        if meta['Age'].notna().mean() > 0.5:
            covariates.append('Age')
        if meta['BMI'].notna().mean() > 0.5:
            covariates.append('BMI')
        if meta['Gender'].notna().mean() > 0.5:
            meta['Gender_num'] = meta['Gender'].map({'Male': 0, 'Female': 1})
            if meta['Gender_num'].notna().mean() > 0.5:
                covariates.append('Gender_num')

        # Medication/lifestyle confounders (encode Yes/No as 1/0, numeric as-is)
        for col in extra_cols:
            if col not in meta.columns:
                continue
            if meta[col].notna().mean() < 0.5:
                continue
            _BINARY_MAPS = [
                {'Yes': 1, 'No': 0},
                {'Smoke': 1, 'Non-smoke': 0},
            ]
            if meta[col].dtype == 'object':
                safe_name = col.replace(' ', '_').replace('/', '_')
                mapped = None
                for bmap in _BINARY_MAPS:
                    if set(meta[col].dropna().unique()) <= set(bmap.keys()) | {'Missing'}:
                        mapped = meta[col].map(bmap)
                        break
                if mapped is not None:
                    meta[safe_name] = mapped
                    if meta[safe_name].notna().mean() > 0.5:
                        covariates.append(safe_name)
                else:
                    # Try numeric conversion
                    numeric = pd.to_numeric(meta[col], errors='coerce')
                    if numeric.notna().mean() > 0.5:
                        meta[safe_name] = numeric
                        covariates.append(safe_name)
            else:
                safe_name = col.replace(' ', '_').replace('/', '_')
                meta[safe_name] = meta[col]
                if meta[safe_name].notna().mean() > 0.5:
                    covariates.append(safe_name)

        if not covariates:
            print(f"{ds}: no covariates available, skipping.")
            continue

        print(f"\n{ds}: partial correlations adjusting for {covariates}")

        # Rename columns with deterministic prefixes BEFORE merging
        # to avoid suffix collision issues
        spc_df = transformed_species[ds].copy()
        mtb_df = transformed_mtb[ds].copy()

        spc_cols_orig = [c for c in spc_df.columns if c != 'Sample']
        mtb_cols_orig = [c for c in mtb_df.columns if c != 'Sample']

        spc_rename = {c: f'spc__{c}' for c in spc_cols_orig}
        mtb_rename = {c: f'mtb__{c}' for c in mtb_cols_orig}

        spc_df = spc_df.rename(columns=spc_rename)
        mtb_df = mtb_df.rename(columns=mtb_rename)

        # Merge species + metabolites + covariates
        merged = spc_df.merge(mtb_df, on='Sample').merge(
            meta[['Sample'] + covariates], on='Sample'
        ).dropna(subset=covariates)

        print(f"  Samples after covariate filtering: {len(merged)}")

        # Select polyamine metabolite columns (with prefix)
        pa_cols_orig = list(polyamine_columns[ds].values())
        pa_cols_prefixed = [f'mtb__{c}' for c in pa_cols_orig if f'mtb__{c}' in merged.columns]

        # Select top species from prior correlations
        # Derive top species from stage_corr filtered to polyamine metabolites
        _all_sc = pd.concat(
            [df for df in stage_corr.get(ds, {}).values() if not df.empty],
            ignore_index=True
        ) if stage_corr.get(ds) else pd.DataFrame()
        _pa_sc = (
            _all_sc[_all_sc['Metabolite'].isin(pa_cols_orig)]
            if not _all_sc.empty else pd.DataFrame()
        )
        if not _pa_sc.empty:
            top_species_orig = _pa_sc.nsmallest(50, 'QValue')['Species'].unique().tolist()
        else:
            top_species_orig = spc_cols_orig[:50]

        top_species_prefixed = [f'spc__{s}' for s in top_species_orig if f'spc__{s}' in merged.columns]

        pc_rows = []
        for spc_pref in top_species_prefixed:
            spc_name = spc_pref.replace('spc__', '', 1)
            for pa_pref in pa_cols_prefixed:
                pa_name = pa_pref.replace('mtb__', '', 1)
                try:
                    result = pg.partial_corr(
                        data=merged, x=spc_pref, y=pa_pref,
                        covar=covariates, method='spearman'
                    )
                    pc_rows.append({
                        'Species': spc_name,
                        'Metabolite': pa_name,
                        'Rho_partial': result['r'].values[0],
                        'PValue_partial': result['p-val'].values[0],
                    })
                except Exception:
                    continue

        if pc_rows:
            pc_df = pd.DataFrame(pc_rows)
            _, qvals, _, _ = multipletests(pc_df['PValue_partial'], method='fdr_bh')
            pc_df['QValue_partial'] = qvals
            partial_corr_results[ds] = pc_df
            n_sig = (pc_df['QValue_partial'] < 0.05).sum()
            print(f"  {len(pc_df)} pairs tested, {n_sig} significant (q<0.05)")

            if n_sig > 0:
                display(pc_df[pc_df['QValue_partial'] < 0.05].sort_values('QValue_partial').head(10))

            pc_df.to_csv(CRC_RESULTS_DIR / 'tables' / f'{ds}_partial_correlations.csv', index=False)
        else:
            print(f"  No valid pairs computed.")

# # Inter-cohort consistency
# consistent_pairs = find_consistent_correlations(
#     polyamine_corr, min_datasets=2, q_threshold=0.1
# )

# if not consistent_pairs.empty:
#     print(f"\n{len(consistent_pairs)} species-polyamine pairs consistent across >= 2 datasets:")
#     display(consistent_pairs.head(15))
#     consistent_pairs.to_csv(
#         CRC_RESULTS_DIR / 'tables' / 'consistent_polyamine_correlations.csv', index=False
#     )
# else:
#     print("\nNo consistent pairs found across datasets at q<0.1.")