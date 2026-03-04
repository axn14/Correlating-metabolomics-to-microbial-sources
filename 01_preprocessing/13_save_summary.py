

# --------------------------------------------------------------------------"""
Step 13 — Save preprocessed data as pickle; write text summary report.
"""


# Save preprocessed data
import pickle

preprocessed_data = {
    'transformed_species': transformed_species,
    'transformed_mtb': transformed_mtb,
    'harmonized_meta': harmonized_meta,
    'polyamine_columns': polyamine_columns,
    'datasets': CRC_DATASETS,
    'removal_summary': {
        'total_removed': len(samples_to_remove),
        'samples_removed': list(samples_to_remove.keys()),
        'removal_reasons': removal_reason
    }
}

output_file = CRC_RESULTS_DIR / 'intermediate' / 'preprocessed_crc_data.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(preprocessed_data, f)

print("\n" + "=" * 80)
print("PREPROCESSING COMPLETE")
print("=" * 80)
print(f"\n✓ Preprocessed data saved to: {output_file}")
print(f"\nFinal dataset summary:")
for ds in CRC_DATASETS:
    print(f"\n{ds}:")
    print(f"  Samples: {len(transformed_species[ds])}")
    print(f"  Species: {len(transformed_species[ds].columns) - 1}")
    print(f"  Metabolites: {len(transformed_mtb[ds].columns) - 1}")
    print(f"  Polyamines: {len(polyamine_columns[ds])}")

# Generate summary report
summary_lines = []
summary_lines.append("=" * 80)
summary_lines.append("CRC-GUT MICROBIOME-METABOLOME ANALYSIS")
summary_lines.append("Preprocessing Summary Report")
summary_lines.append("=" * 80)
summary_lines.append("")
summary_lines.append("DATASETS ANALYZED:")
summary_lines.append("  1. ERAWIJANTARI-GASTRIC-CANCER-2020")
summary_lines.append("  2. YACHIDA-CRC-2019")
summary_lines.append("")
summary_lines.append("DUPLICATE REMOVAL:")
summary_lines.append(f"  Total duplicates removed: {len(samples_to_remove)}")
if len(samples_to_remove) > 0:
    summary_lines.append(f"  Removed from: ERAWIJANTARI-GASTRIC-CANCER-2020")
    summary_lines.append(f"  Reason: Duplicate samples also present in YACHIDA-CRC-2019")
else:
    summary_lines.append("  No duplicates detected - datasets are independent")
summary_lines.append("")
summary_lines.append("FINAL SAMPLE COUNTS:")
for ds in CRC_DATASETS:
    summary_lines.append(f"  {ds}: {len(transformed_species[ds])} samples")
summary_lines.append("")
summary_lines.append("PREPROCESSING STEPS APPLIED:")
summary_lines.append("  1. ✓ Quality control (sample & feature level)")
summary_lines.append("  2. ✓ Taxonomic name reduction")
summary_lines.append(
    f"  3. ✓ Prevalence filtering  "
    f"(species ≥{SPECIES_PREVALENCE*100:.0f}%,  metabolites ≥{MTB_PREVALENCE*100:.0f}%)")
summary_lines.append("  4. ✓ Variance filtering (near-zero variance removed)")
summary_lines.append("  5. ✓ CLR transformation  (species, pseudocount=1e-6)")
summary_lines.append("  6. ✓ log2 transformation (metabolites)")
summary_lines.append("  7. ✓ PCA visualization")
summary_lines.append("")
summary_lines.append("FINAL FEATURE COUNTS:")
for ds in CRC_DATASETS:
    n_spc = len(transformed_species[ds].columns) - 1
    n_mtb = len(transformed_mtb[ds].columns) - 1
    n_pa = len(polyamine_columns[ds])
    summary_lines.append(f"  {ds}:")
    summary_lines.append(f"    Species: {n_spc}")
    summary_lines.append(f"    Metabolites: {n_mtb}")
    summary_lines.append(f"    Polyamines: {n_pa}")
summary_lines.append("")
summary_lines.append("OUTPUTS GENERATED:")
summary_lines.append(f"  - Preprocessed data: {output_file}")
summary_lines.append(f"  - QC reports: {CRC_RESULTS_DIR / 'tables'}")
summary_lines.append(f"  - PCA plots: {CRC_RESULTS_DIR / 'figures' / 'pca'}")
if len(samples_to_remove) > 0:
    summary_lines.append(f"  - Duplicate removal report: {CRC_RESULTS_DIR / 'tables' / 'duplicate_samples_removed.csv'}")
summary_lines.append("")
summary_lines.append("NEXT STEPS:")
summary_lines.append("  1. Differential abundance analysis (Healthy vs CRC)")
summary_lines.append("  2. Species-polyamine correlation analysis")
summary_lines.append("  3. Cross-cohort validation")
summary_lines.append("  4. Machine learning models")
summary_lines.append("")
summary_lines.append("=" * 80)

summary_text = "\n".join(summary_lines)
print(summary_text)

# Save summary report
with open(CRC_RESULTS_DIR / 'preprocessing_summary.txt', 'w') as f:
    f.write(summary_text)

print(f"\n✓ Summary report saved to: {CRC_RESULTS_DIR / 'preprocessing_summary.txt'}")