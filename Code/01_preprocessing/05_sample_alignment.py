

# --------------------------------------------------------------------------
"""
Step 5 — Validate that metadata, species, and metabolite tables share identical sample IDs.
"""


# Validate alignment
print("\n" + "=" * 80)
print("SAMPLE ALIGNMENT VALIDATION")
print("=" * 80)

for ds in CRC_DATASETS:
    print(f"\n{ds}:")
    
    meta_samples = set(data[ds]['metadata']['Sample'])
    mtb_samples = set(data[ds]['mtb']['Sample'])
    spc_samples = set(data[ds]['species']['Sample'])
    
    common = meta_samples & mtb_samples & spc_samples
    
    print(f"  Metadata samples: {len(meta_samples)}")
    print(f"  Metabolite samples: {len(mtb_samples)}")
    print(f"  Species samples: {len(spc_samples)}")
    print(f"  Common (all 3): {len(common)}")
    
    meta_only = meta_samples - mtb_samples - spc_samples
    mtb_only = mtb_samples - meta_samples
    spc_only = spc_samples - meta_samples
    
    if meta_only or mtb_only or spc_only:
        print(f"  ⚠️  WARNING: Misalignment detected!")
        if meta_only:
            print(f"     Samples in metadata only: {len(meta_only)}")
        if mtb_only:
            print(f"     Samples in metabolites only: {len(mtb_only)}")
        if spc_only:
            print(f"     Samples in species only: {len(spc_only)}")
    else:
        print(f"  ✓ Perfect alignment across all data types")
