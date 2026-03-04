

# --------------------------------------------------------------------------
"""
Step 9 — Parse GTDB taxonomy strings to species-level names; aggregate duplicates by summing abundances.
"""


# Reduce species names
species_reduced = {}
name_mappings = {}

for ds in CRC_DATASETS:
    print(f"\n{ds}:")
    reduced, mapping = reduce_species_names(data[ds]['species'])
    species_reduced[ds] = reduced
    name_mappings[ds] = mapping
    
    print(f"  Original columns: {len(data[ds]['species'].columns) - 1}")
    print(f"  Reduced columns: {len(reduced.columns) - 1}")
    print(f"  Reduction: {len(data[ds]['species'].columns) - len(reduced.columns)} features collapsed")
