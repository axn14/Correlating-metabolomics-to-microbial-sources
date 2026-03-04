

# --------------------------------------------------------------------------
"""
Step 3 — Detect duplicate samples between cohorts via Sample ID, Subject ID, and metadata fingerprint.
"""


# Examine metadata columns to find potential linking identifiers
print("=" * 80)
print("METADATA STRUCTURE EXAMINATION")
print("=" * 80)

for ds in CRC_DATASETS:
    print(f"\n{ds} metadata columns:")
    print(data[ds]['metadata'].columns.tolist())
    print(f"\nFirst 3 samples:")
    display(data[ds]['metadata'].head(3))

# Check for overlapping Sample IDs or Subject IDs
eraw_samples = set(data['ERAWIJANTARI-GASTRIC-CANCER-2020']['metadata']['Sample'])
yach_samples = set(data['YACHIDA-CRC-2019']['metadata']['Sample'])

eraw_subjects = set(data['ERAWIJANTARI-GASTRIC-CANCER-2020']['metadata']['Subject'])
yach_subjects = set(data['YACHIDA-CRC-2019']['metadata']['Subject'])

print("\n" + "=" * 80)
print("OVERLAP DETECTION")
print("=" * 80)

print(f"\nERAWIJANTARI samples: {len(eraw_samples)}")
print(f"YACHIDA samples: {len(yach_samples)}")

# Direct overlap in Sample IDs
sample_overlap = eraw_samples & yach_samples
print(f"\nDirect Sample ID overlap: {len(sample_overlap)}")
if sample_overlap:
    print(f"  Overlapping samples: {sorted(list(sample_overlap))[:10]} ...")

# Direct overlap in Subject IDs
subject_overlap = eraw_subjects & yach_subjects
print(f"\nDirect Subject ID overlap: {len(subject_overlap)}")
if subject_overlap:
    print(f"  Overlapping subjects: {sorted(list(subject_overlap))[:10]} ...")

# Additional checks for hidden duplicates
# Sometimes samples have different IDs but same underlying subject
# Check by comparing metadata fields (Age, Gender, Study.Group)

print("\n" + "=" * 80)
print("METADATA-BASED DUPLICATE DETECTION")
print("=" * 80)

# Create metadata fingerprints (Age + Gender + Study.Group)
def create_fingerprint(row):
    """Create a unique fingerprint for potential duplicate detection."""
    return f"{row.get('Age', 'NA')}_{row.get('Gender', 'NA')}_{row.get('Study.Group', 'NA')}"

eraw_meta = data['ERAWIJANTARI-GASTRIC-CANCER-2020']['metadata'].copy()
yach_meta = data['YACHIDA-CRC-2019']['metadata'].copy()

# Add fingerprint column
eraw_meta['fingerprint'] = eraw_meta.apply(create_fingerprint, axis=1)
yach_meta['fingerprint'] = yach_meta.apply(create_fingerprint, axis=1)

# Find duplicates by fingerprint
eraw_fingerprints = set(eraw_meta['fingerprint'])
yach_fingerprints = set(yach_meta['fingerprint'])

fingerprint_overlap = eraw_fingerprints & yach_fingerprints
print(f"\nPotential duplicates by metadata fingerprint: {len(fingerprint_overlap)}")

if len(fingerprint_overlap) > 0:
    print(f"\nWARNING: {len(fingerprint_overlap)} potential duplicate subjects detected!")
    print("\nExamining potential duplicates:")
    
    for fp in sorted(list(fingerprint_overlap))[:5]:
        eraw_match = eraw_meta[eraw_meta['fingerprint'] == fp][['Sample', 'Subject', 'Age', 'Gender', 'Study.Group']]
        yach_match = yach_meta[yach_meta['fingerprint'] == fp][['Sample', 'Subject', 'Age', 'Gender', 'Study.Group']]
        
        print(f"\nFingerprint: {fp}")
        print("  ERAWIJANTARI:")
        display(eraw_match)
        print("  YACHIDA:")
        display(yach_match)
else:
    print("\n✓ No metadata-based duplicates detected.")
