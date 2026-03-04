

# --------------------------------------------------------------------------"""
Step 7 — Map KEGG IDs to metabolite column names; identify putrescine, spermidine, spermine, cadaverine, etc.
"""


# Find polyamine metabolites in each dataset
polyamine_columns = {}

print("\n" + "=" * 80)
print("POLYAMINE IDENTIFICATION")
print("=" * 80)

for ds in CRC_DATASETS:
    found = find_polyamine_columns(data[ds]['mtb.map'], data[ds]['mtb'], ds)
    polyamine_columns[ds] = found
    
    print(f"\n{ds}: {len(found)} polyamines found")
    for kegg_id, col_name in found.items():
        pa_name = POLYAMINE_KEGG[kegg_id]
        print(f"  {kegg_id} ({pa_name}): {col_name}")