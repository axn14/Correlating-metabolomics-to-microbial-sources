

# --------------------------------------------------------------------------
"""
Step 8 — Compute and plot sample-level and feature-level QC metrics (detection rates, total signal).
"""


# Compute sample-level QC metrics
sample_qc_species = {}
sample_qc_mtb = {}

for ds in CRC_DATASETS:
    print(f"\n{ds}:")
    
    # Species QC
    sample_qc_species[ds] = compute_sample_qc(data[ds]['species'], data_type='species')
    print(f"  Species QC:")
    print(f"    Mean features detected: {sample_qc_species[ds]['n_features'].mean():.0f}")
    print(f"    Mean total signal: {sample_qc_species[ds]['total_signal'].mean():.2e}")
    
    # Metabolite QC
    sample_qc_mtb[ds] = compute_sample_qc(data[ds]['mtb'])
    print(f"  Metabolite QC:")
    print(f"    Mean features detected: {sample_qc_mtb[ds]['n_features'].mean():.0f}")
    print(f"    Mean total signal: {sample_qc_mtb[ds]['total_signal'].mean():.2e}")

# Save QC reports
for ds in CRC_DATASETS:
    sample_qc_species[ds].to_csv(
        CRC_RESULTS_DIR / 'tables' / f'{ds}_sample_qc_species.csv', index=False
    )
    sample_qc_mtb[ds].to_csv(
        CRC_RESULTS_DIR / 'tables' / f'{ds}_sample_qc_metabolites.csv', index=False
    )

print("\n✓ Sample QC reports saved")

# Compute feature-level QC metrics
feature_qc_species = {}
feature_qc_mtb = {}

for ds in CRC_DATASETS:
    print(f"\n{ds}:")
    
    # Species QC
    feature_qc_species[ds] = compute_feature_qc(data[ds]['species'])
    print(f"  Species features:")
    print(f"    Total: {len(feature_qc_species[ds])}")
    print(f"    Mean detection rate: {feature_qc_species[ds]['detection_rate'].mean():.2%}")
    print(f"    Features >10% prevalence: {(feature_qc_species[ds]['detection_rate'] >= 0.1).sum()}")
    
    # Metabolite QC
    feature_qc_mtb[ds] = compute_feature_qc(data[ds]['mtb'])
    print(f"  Metabolite features:")
    print(f"    Total: {len(feature_qc_mtb[ds])}")
    print(f"    Mean detection rate: {feature_qc_mtb[ds]['detection_rate'].mean():.2%}")
    print(f"    Features >10% prevalence: {(feature_qc_mtb[ds]['detection_rate'] >= 0.1).sum()}")

# Plot detection rate distributions
n_ds = len(CRC_DATASETS)
fig, axes = plt.subplots(n_ds, 2, figsize=(14, 5 * n_ds))
if n_ds == 1:
    axes = axes[np.newaxis, :]  # ensure 2-D indexing

for i, ds in enumerate(CRC_DATASETS):
    # Species
    axes[i, 0].hist(feature_qc_species[ds]['detection_rate'], bins=50, edgecolor='black', alpha=0.7)
    axes[i, 0].axvline(0.1, color='red', ls='--', label='10% threshold')
    axes[i, 0].set_title(f'{ds}Species Detection Rates')
    axes[i, 0].set_xlabel('Detection Rate')
    axes[i, 0].set_ylabel('Count')
    axes[i, 0].legend()

    # Metabolites
    axes[i, 1].hist(feature_qc_mtb[ds]['detection_rate'], bins=50, edgecolor='black', alpha=0.7)
    axes[i, 1].axvline(0.1, color='red', ls='--', label='10% threshold')
    axes[i, 1].set_title(f'{ds}Metabolite Detection Rates')
    axes[i, 1].set_xlabel('Detection Rate')
    axes[i, 1].set_ylabel('Count')
    axes[i, 1].legend()

plt.tight_layout()
plt.savefig(CRC_RESULTS_DIR / 'figures' / 'qc' / 'detection_rate_distributions.png',
            dpi=100, bbox_inches='tight')
plt.show()
plt.close()
