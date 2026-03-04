

# --------------------------------------------------------------------------
"""
Step 10 — Prevalence filtering (species ≥20%, metabolites ≥15%) and near-zero variance filtering. Polyamines are protected.
"""


# Apply prevalence filtering
species_filtered = {}
mtb_filtered = {}

SPECIES_PREVALENCE = 0.20   # stricter: remove species detected in <20% of samples
MTB_PREVALENCE     = 0.15   # stricter: remove metabolites detected in <15% of samples

print("\n" + "=" * 80)
print(f"PREVALENCE FILTERING  (species ≥{SPECIES_PREVALENCE*100:.0f}%,  metabolites ≥{MTB_PREVALENCE*100:.0f}%)")
print("=" * 80)

for ds in CRC_DATASETS:
    print(f"\n{ds}:")

    # Get polyamine columns to protect during metabolite filtering
    protected_mtb = list(polyamine_columns.get(ds, {}).values())

    # Species
    species_filtered[ds] = filter_by_prevalence(
        species_reduced[ds], threshold=SPECIES_PREVALENCE
    )
    print(f"  Species:")
    print(f"    Before: {len(species_reduced[ds].columns) - 1}")
    print(f"    After:  {len(species_filtered[ds].columns) - 1}")
    print(f"    Removed: {len(species_reduced[ds].columns) - len(species_filtered[ds].columns)}")

    # Metabolites (polyamines are protected)
    mtb_filtered[ds] = filter_by_prevalence(
        data[ds]['mtb'], threshold=MTB_PREVALENCE, protected_cols=protected_mtb
    )
    print(f"  Metabolites:")
    print(f"    Before: {len(data[ds]['mtb'].columns) - 1}")
    print(f"    After:  {len(mtb_filtered[ds].columns) - 1}")
    print(f"    Removed: {len(data[ds]['mtb'].columns) - len(mtb_filtered[ds].columns)}")

    # Verify polyamines are preserved
    mtb_cols = set(mtb_filtered[ds].columns)
    for kegg_id, col_name in polyamine_columns[ds].items():
        status = "KEPT" if col_name in mtb_cols else "REMOVED"
        print(f"    Polyamine {POLYAMINE_KEGG[kegg_id]}: {status}")


# Apply variance filtering
species_var_filtered = {}
mtb_var_filtered = {}

for ds in CRC_DATASETS:
    print(f"\n{ds}:")
    
    # Species variance filtering
    species_var_filtered[ds] = filter_near_zero_variance(species_filtered[ds])
    print(f"  Species after variance filter: {len(species_var_filtered[ds].columns) - 1}")
    
    # Metabolites variance filtering
    mtb_var_filtered[ds] = filter_near_zero_variance(mtb_filtered[ds])
    print(f"  Metabolites after variance filter: {len(mtb_var_filtered[ds].columns) - 1}")
