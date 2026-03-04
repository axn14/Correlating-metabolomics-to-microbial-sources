

# --------------------------------------------------------------------------"""
Full unstratified correlations — top 200 most variable species × metabolites; used as input to global network analysis.
"""


# Full species-metabolite correlations (capped at top 200 most variable features)
full_corr = {}
MAX_FEATURES = 200

for ds in CRC_DATASETS:
    print(f"\nFull correlations for {ds}...")
    spc_cols = [c for c in transformed_species[ds].columns if c != 'Sample']
    mtb_cols = [c for c in transformed_mtb[ds].columns if c != 'Sample']

    spc_df = transformed_species[ds]
    if len(spc_cols) > MAX_FEATURES:
        top_spc = transformed_species[ds][spc_cols].var().nlargest(MAX_FEATURES).index.tolist()
        spc_df = transformed_species[ds][['Sample'] + top_spc]

    mtb_df = transformed_mtb[ds]
    if len(mtb_cols) > MAX_FEATURES:
        top_mtb = transformed_mtb[ds][mtb_cols].var().nlargest(MAX_FEATURES).index.tolist()
        mtb_df = transformed_mtb[ds][['Sample'] + top_mtb]

    corr_df = compute_correlations(spc_df, mtb_df)
    full_corr[ds] = corr_df

    if not corr_df.empty:
        n_sig = (corr_df['QValue'] < 0.05).sum()
        print(f"  {len(corr_df)} pairs tested, {n_sig} significant (q<0.05)")