

# --------------------------------------------------------------------------"""
Step 11 — CLR transform for species (pseudocount=1e-6); log2 transform for metabolites (skipped if pre-normalised).
"""


# Apply transformations
transformed_species = {}
transformed_mtb = {}

print("=" * 80)
print("TRANSFORMATION")
print("=" * 80)

for ds in CRC_DATASETS:
    print(f"{ds}:")

    # CLR transform species (compositional; pseudocount handles zeros)
    transformed_species[ds] = clr_transform(species_var_filtered[ds], pseudocount=1e-6)
    print(f"  Species: CLR transformation applied  (pseudocount=1e-6)")
    print(f"    Shape: {transformed_species[ds].shape}")
    print(f"    Range: [{transformed_species[ds].iloc[:, 1:].min().min():.2f}, "
          f"{transformed_species[ds].iloc[:, 1:].max().max():.2f}]")

    # Metabolite transform: skip log2 if data already pre-normalised (contains negatives)
    numeric_cols = [c for c in mtb_var_filtered[ds].columns if c != 'Sample']
    mtb_vals = mtb_var_filtered[ds][numeric_cols].values
    if (mtb_vals < 0).any():
        print(f"  Metabolites: pre-normalised data detected (negatives present) — skipping log2.")
        transformed_mtb[ds] = mtb_var_filtered[ds].copy()
    else:
        transformed_mtb[ds] = log_transform(mtb_var_filtered[ds])
        print(f"  Metabolites: log2 transformation applied")
    print(f"    Shape: {transformed_mtb[ds].shape}")
    print(f"    Range: [{transformed_mtb[ds].iloc[:, 1:].min().min():.2f}, "
          f"{transformed_mtb[ds].iloc[:, 1:].max().max():.2f}]")
