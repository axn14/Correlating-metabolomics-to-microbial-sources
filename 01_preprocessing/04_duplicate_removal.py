

# --------------------------------------------------------------------------
"""
Step 4 — Plan and apply removal of duplicate samples; retain duplicates in YACHIDA, remove from ERAWIJANTARI.
"""


# Determine which samples to remove
samples_to_remove = {}
removal_reason = {}

# Strategy 1: Remove exact Sample ID matches
if len(sample_overlap) > 0:
    for sample in sample_overlap:
        samples_to_remove[sample] = 'ERAWIJANTARI-GASTRIC-CANCER-2020'
        removal_reason[sample] = 'Exact Sample ID match in YACHIDA'

# Strategy 2: Remove exact Subject ID matches (different sample, same person)
if len(subject_overlap) > 0:
    eraw_subject_samples = data['ERAWIJANTARI-GASTRIC-CANCER-2020']['metadata'][
        data['ERAWIJANTARI-GASTRIC-CANCER-2020']['metadata']['Subject'].isin(subject_overlap)
    ]['Sample'].tolist()
    
    for sample in eraw_subject_samples:
        if sample not in samples_to_remove:
            samples_to_remove[sample] = 'ERAWIJANTARI-GASTRIC-CANCER-2020'
            removal_reason[sample] = 'Subject ID match in YACHIDA'

# Strategy 3: Remove metadata fingerprint matches (likely duplicates)
if len(fingerprint_overlap) > 0:
    eraw_fingerprint_samples = eraw_meta[
        eraw_meta['fingerprint'].isin(fingerprint_overlap)
    ]['Sample'].tolist()
    
    for sample in eraw_fingerprint_samples:
        if sample not in samples_to_remove:
            samples_to_remove[sample] = 'ERAWIJANTARI-GASTRIC-CANCER-2020'
            removal_reason[sample] = 'Metadata fingerprint match in YACHIDA (Age+Gender+Group)'

print("\n" + "=" * 80)
print("DUPLICATE REMOVAL PLAN")
print("=" * 80)

if len(samples_to_remove) > 0:
    print(f"\n⚠️  {len(samples_to_remove)} samples will be removed from ERAWIJANTARI")
    
    # Create removal report
    removal_df = pd.DataFrame([
        {'Sample': sample, 'Dataset': dataset, 'Reason': removal_reason[sample]}
        for sample, dataset in samples_to_remove.items()
    ])
    
    print("\nFirst 10 samples to be removed:")
    display(removal_df.head(10))
    
    # Save removal report
    removal_df.to_csv(CRC_RESULTS_DIR / 'tables' / 'duplicate_samples_removed.csv', index=False)
    print(f"\n✓ Full removal report saved to: duplicate_samples_removed.csv")
else:
    print("\n✓ No duplicates detected. Both datasets are independent.")
    print("   Proceeding with all samples from both cohorts.")

# Apply duplicate removal
data_deduplicated = {}

for ds in CRC_DATASETS:
    data_deduplicated[ds] = {}
    
    # Get samples to keep
    all_samples = set(data[ds]['metadata']['Sample'])
    removed_samples = set([s for s, d in samples_to_remove.items() if d == ds])
    keep_samples = all_samples - removed_samples
    
    print(f"\n{ds}:")
    print(f"  Original samples: {len(all_samples)}")
    print(f"  Removed samples: {len(removed_samples)}")
    print(f"  Retained samples: {len(keep_samples)}")
    
    # Filter each data type
    for ft in ['metadata', 'mtb', 'species']:
        if data[ds][ft] is not None:
            data_deduplicated[ds][ft] = data[ds][ft][
                data[ds][ft]['Sample'].isin(keep_samples)
            ].copy()
        else:
            data_deduplicated[ds][ft] = None
    
    # Keep map file unchanged
    data_deduplicated[ds]['mtb.map'] = data[ds]['mtb.map']

# Update main data dict
data = data_deduplicated

print("\n" + "=" * 80)
print("FINAL SAMPLE COUNTS")
print("=" * 80)

for ds in CRC_DATASETS:
    print(f"\n{ds}:")
    print(f"  Metadata: {len(data[ds]['metadata'])} samples")
    print(f"  Species: {len(data[ds]['species'])} samples")
    print(f"  Metabolites: {len(data[ds]['mtb'])} samples")
