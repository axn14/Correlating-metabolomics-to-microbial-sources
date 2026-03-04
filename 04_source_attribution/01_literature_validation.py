

# --------------------------------------------------------------------------"""
Literature validation — Curated LITERATURE_PRODUCERS dictionary (Wolken 2018, Gao 2019, Hanfrey 2011). Genus concordance between identified source species and published polyamine producers. Helper functions: extract_genus(), polyamine_display_name().
"""


# ============================================================
# LITERATURE VALIDATION — Identified source species vs known producers
# ============================================================

# Curated literature: known polyamine-producing taxa (genus or species level).
# Sources: Wolken et al. 2018 (putrescine), Gao et al. 2019 (spermidine/spermine),
#          Hanfrey et al. 2011, Perez-Cano & Borges 2011.
LITERATURE_PRODUCERS = {
    "Putrescine":   ["Bacteroides", "Fusobacterium", "Proteus", "Escherichia",
                     "Clostridium", "Enterococcus", "Lactobacillus"],
    "Spermidine":   ["Escherichia", "Pseudomonas", "Lactobacillus", "Bacteroides",
                     "Bifidobacterium", "Porphyromonas"],
    "Spermine":     ["Escherichia", "Salmonella", "Pseudomonas", "Klebsiella"],
    "Cadaverine":   ["Escherichia", "Proteus", "Fusobacterium",
                     "Clostridium", "Hafnia"],
    "N-Acetylputrescine": ["Bacteroides", "Ruminococcus", "Prevotella"],
    "N1-Acetylspermidine": ["Bacteroides", "Ruminococcus"],
}

def extract_genus(species_name):
    """Return the genus (first word) from a full species or taxonomy string."""
    name = species_name.split(";")[-1].strip()
    return name.split("__")[-1].split("_")[0].split(" ")[0]


def polyamine_display_name(col_name):
    """Map KEGG column name to a human-readable polyamine name."""
    mapping = {
        "C00134": "Putrescine",
        "C00750": "Spermine",
        "C00315": "Spermidine",
        "C01672": "Cadaverine",
        "C02813": "N-Acetylputrescine",
        "C00488": "N1-Acetylspermidine",
    }
    for kegg_id, display in mapping.items():
        if kegg_id in col_name:
            return display
    return col_name.split("_")[0]


validation_rows = []

print("=" * 80)
print("LITERATURE VALIDATION OF IDENTIFIED PRODUCER SPECIES")
print("=" * 80)

for ds in CRC_DATASETS:
    if ds not in polyamine_columns:
        continue
    print(f"\n{ds}:")

    for kegg_id, col_name in polyamine_columns[ds].items():
        pa_display = polyamine_display_name(col_name)
        known_genera = LITERATURE_PRODUCERS.get(pa_display, [])
        if not known_genera:
            continue

        # Gather identified sources from derive_source_labels (all stages)
        if ds not in stage_corr or not stage_corr[ds]:
            continue
        try:
            y_labels = derive_source_labels(
                stage_corr[ds], transformed_species[ds],
                harmonized_meta[ds], col_name
            )
        except Exception:
            y_labels = None

        if y_labels is None or len(y_labels) == 0:
            print(f"  {pa_display}: no labels derived — skipping")
            continue

        # Extract genus from identified species
        identified_counts = y_labels.value_counts()
        total_labelled = len(y_labels)

        print(f"\n  {pa_display} ({total_labelled} labelled samples):")
        print(f"  Known genera: {known_genera}")

        validated = []
        novel     = []
        for spc, cnt in identified_counts.items():
            genus = extract_genus(str(spc))
            is_known = any(genus.lower().startswith(k.lower()) for k in known_genera)
            pct = 100 * cnt / total_labelled
            if is_known:
                validated.append((spc, cnt, pct, genus))
            else:
                novel.append((spc, cnt, pct, genus))

        validated_pct = sum(r[2] for r in validated)
        print(f"  Literature-validated producers ({validated_pct:.1f}% of samples):")
        for spc, cnt, pct, genus in sorted(validated, key=lambda x: -x[1]):
            print(f"    [✓] {str(spc)[:60]}  (n={cnt}, {pct:.1f}%)")

        print(f"  Novel (not in literature) producers:")
        for spc, cnt, pct, genus in sorted(novel, key=lambda x: -x[1])[:5]:
            print(f"    [?] {str(spc)[:60]}  (n={cnt}, {pct:.1f}%)")

        validation_rows.append({
            "Dataset":         ds,
            "Polyamine":       pa_display,
            "N_labelled":      total_labelled,
            "Pct_validated":   round(validated_pct, 2),
            "Validated_spc":   "; ".join(str(r[0])[:30] for r in validated),
            "Top_novel_spc":   "; ".join(str(r[0])[:30] for r in novel[:3]),
        })

if validation_rows:
    val_df = pd.DataFrame(validation_rows)
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    display(val_df)
    val_df.to_csv(
        CRC_RESULTS_DIR / "tables" / "literature_validation_producers.csv",
        index=False
    )
    print("Saved: tables/literature_validation_producers.csv")

    # Bar chart: % validated per dataset x polyamine
    if len(val_df) > 0:
        pivot_val = val_df.pivot_table(
            index="Polyamine", columns="Dataset",
            values="Pct_validated", aggfunc="mean"
        ).fillna(0)
        ax = pivot_val.plot(kind="bar", figsize=(max(8, len(pivot_val) * 1.2), 5),
                            color=["#2980b9", "#27ae60"][:len(pivot_val.columns)])
        ax.set_ylabel("% samples with literature-validated producer")
        ax.set_title("Literature Validation: Identified Producer Species")
        ax.set_ylim(0, 110)
        ax.axhline(50, linestyle="--", color="gray", linewidth=0.8)
        ax.legend(title="Dataset", fontsize=8)
        for bar in ax.patches:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                        f"{h:.0f}%", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(
            CRC_RESULTS_DIR / "figures" / "correlations" /
            "literature_validation_bar.png",
            dpi=150, bbox_inches="tight"
        )
        plt.show()
        plt.close()
else:
    print("No validation results produced.")
