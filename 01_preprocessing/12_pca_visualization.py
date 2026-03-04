

# --------------------------------------------------------------------------"""
Step 12 — PCA on CLR-transformed species and log2-transformed metabolite data; colour by Study.Group.
"""


# PCA on species data
for ds in CRC_DATASETS:
    print(f"\nGenerating PCA for {ds}...")
    
    fig, ax = plot_pca(
        transformed_species[ds],
        harmonized_meta[ds],
        title=f'{ds} - Species (CLR-transformed)',
        color_col='Study.Group',
        figsize=(12, 8),
        save_path=CRC_RESULTS_DIR / 'figures' / 'pca' / f'{ds}_species_pca.png'
    )
    plt.show()
    plt.close()

# PCA on metabolite data
for ds in CRC_DATASETS:
    print(f"\nGenerating PCA for {ds} metabolites...")
    
    fig, ax = plot_pca(
        transformed_mtb[ds],
        harmonized_meta[ds],
        title=f'{ds} - Metabolites (log2-transformed)',
        color_col='Study.Group',
        figsize=(12, 8),
        save_path=CRC_RESULTS_DIR / 'figures' / 'pca' / f'{ds}_metabolites_pca.png'
    )
    plt.show()
    plt.close()