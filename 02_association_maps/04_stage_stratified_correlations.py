

# --------------------------------------------------------------------------"""
Stage-stratified Spearman correlations — all species × all metabolites within each disease stage (q<0.05). Heatmaps of top associations per stage.
"""


# ============================================================
# ALL-METABOLITE SPECIES CORRELATIONS — stratified by disease stage
# ============================================================
# stage_corr[ds][stage] = DataFrame(Species, Metabolite, Rho, PValue, QValue)
# Uses ALL prevalence-filtered metabolites (not only polyamines).

stage_corr = {}

for ds in CRC_DATASETS:
    stage_corr[ds] = {}
    stages = harmonized_meta[ds]['Study.Group'].unique().tolist()
    print(f"\n{ds}: {len(stages)} stage(s) -> {stages}")

    for stage in stages:
        stage_samples = harmonized_meta[ds].loc[
            harmonized_meta[ds]['Study.Group'] == stage, 'Sample'
        ].values

        spc_sub = transformed_species[ds][
            transformed_species[ds]['Sample'].isin(stage_samples)
        ]
        mtb_sub = transformed_mtb[ds][
            transformed_mtb[ds]['Sample'].isin(stage_samples)
        ]

        if len(stage_samples) < 10:
            print(f"  [{stage}] SKIP - only {len(stage_samples)} samples")
            continue

        corr_df = compute_correlations(spc_sub, mtb_sub)   # all metabolites
        stage_corr[ds][stage] = corr_df

        n_sig = (corr_df['QValue'] < 0.05).sum() if not corr_df.empty else 0
        print(f"  [{stage}] n={len(stage_samples)}, "
              f"{len(corr_df)} pairs, {n_sig} significant (q<0.05)")

# -- Heatmaps: top 20 associations per stage ----------------------------------
for ds in CRC_DATASETS:
    for stage, corr_df in stage_corr[ds].items():
        if corr_df.empty:
            continue
        fig, ax = plot_correlation_heatmap(
            corr_df,
            title=f'{ds} [{stage}] - All Metabolites vs Species',
            top_n=20,
            save_path=CRC_RESULTS_DIR / 'figures' / 'correlations'
                      / f'{ds}_{stage}_all_mtb_heatmap.png'
        )
        if fig:
            plt.show()
            plt.close()