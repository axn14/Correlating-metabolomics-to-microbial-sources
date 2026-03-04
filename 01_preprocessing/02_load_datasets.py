

# --------------------------------------------------------------------------"""
Step 2 — Load raw TSV files for ERAWIJANTARI-GASTRIC-CANCER-2020 and YACHIDA-CRC-2019.
"""


# Load CRC-related datasets (Yachida and Erawijantari only)
CRC_DATASETS = [
    'ERAWIJANTARI-GASTRIC-CANCER-2020',
    'YACHIDA-CRC-2019',
]

data = {}
for ds in CRC_DATASETS:
    data[ds] = {}
    for ft in ['metadata', 'mtb', 'mtb.map', 'species']:
        fpath = DATA_DIR / f"{ds} {ft}.tsv"
        # Some datasets use 'mtb_map.tsv' (underscore) instead of 'mtb.map.tsv'
        if not fpath.exists() and ft == 'mtb.map':
            fpath = DATA_DIR / f"{ds} mtb_map.tsv"
        if fpath.exists():
            data[ds][ft] = pd.read_csv(fpath, sep='	')
        else:
            warnings.warn(f'File not found: {fpath}')
            data[ds][ft] = None

print('Datasets loaded:')
for ds in CRC_DATASETS:
    print(f'{ds}:')
    print(f'  Metadata: {len(data[ds]["metadata"])} samples')
    print(f'  Species: {len(data[ds]["species"])} samples, {len(data[ds]["species"].columns)-1} features')
    print(f'  Metabolites: {len(data[ds]["mtb"])} samples, {len(data[ds]["mtb"].columns)-1} features')
