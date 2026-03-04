

# --------------------------------------------------------------------------"""
Step 6 — Harmonize metadata across cohorts into a common schema (Study.Group, Age, Gender, BMI).
"""


# Create harmonized metadata with common schema
harmonized_meta = {}
for ds in CRC_DATASETS:
    harmonized_meta[ds] = harmonize_metadata(data[ds]['metadata'], ds)

# Display study group distributions
print("\n" + "=" * 80)
print("HARMONIZED METADATA")
print("=" * 80)

for ds in CRC_DATASETS:
    print(f"\n{ds}:")
    print("Study Group Distribution:")
    print(harmonized_meta[ds]['Study.Group'].value_counts().to_string())
    
    # Demographics
    hm = harmonized_meta[ds]
    print(f"\nDemographics:")
    print(f"  Age: mean={hm['Age'].mean():.1f}, range=[{hm['Age'].min():.0f}-{hm['Age'].max():.0f}]")
    if hm['Gender'].notna().any():
        print(f"  Gender: {hm['Gender'].value_counts().to_dict()}")
    if hm['BMI'].notna().any():
        print(f"  BMI: mean={hm['BMI'].mean():.1f}")

# All study groups retained for both cohorts (no group filtering applied)
print("\nFinal study group distribution (all groups retained):")
for ds in CRC_DATASETS:
    grp_counts = harmonized_meta[ds]['Study.Group'].value_counts()
    print(f"\n{ds}  ({len(harmonized_meta[ds])} samples total)")
    for g, n in grp_counts.items():
        print(f"  {g}: {n}")
