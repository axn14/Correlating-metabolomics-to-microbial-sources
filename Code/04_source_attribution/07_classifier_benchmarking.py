

# --------------------------------------------------------------------------"""
Classifier benchmarking — A1: overall CV model comparison → best_model_name. A2: per-polyamine model comparison. A3: literature concordance (external validation) bar chart. Saves polyamine_concordance.csv.
"""


# ============================================================
# BENCHMARKING — Classifier performance + literature concordance
# Assesses accuracy of source attribution tools against:
#   (1) Internal CV metrics (balanced accuracy, F1-macro)
#   (2) External validation: genus-level match with LITERATURE_PRODUCERS
# ============================================================

# ── A1. Overall CV model comparison ──────────────────────────────────────────
if 'clf_df' not in dir() or clf_df.empty:
    print("[WARN] clf_df not available — run classification cells first.")
    model_bench = pd.DataFrame()
    best_model_name = 'RandomForest'
else:
    model_bench = (
        clf_df.groupby('Model')
        .agg(
            BA_mean   = ('balanced_accuracy_mean', 'mean'),
            BA_std    = ('balanced_accuracy_mean', 'std'),
            F1_mean   = ('f1_macro_mean', 'mean'),
            F1_std    = ('f1_macro_mean', 'std'),
            n_tasks   = ('Metabolite', 'nunique'),
        )
        .sort_values('BA_mean', ascending=False)
        .round(4)
    )
    best_model_name = model_bench.index[0]

    print("=" * 60)
    print("CLASSIFIER BENCHMARK  (10×3 repeated stratified k-fold CV)")
    print("=" * 60)
    display(model_bench)
    print(f"\nBest overall model: {best_model_name} "
          f"(BA = {model_bench.loc[best_model_name, 'BA_mean']:.3f} "
          f"± {model_bench.loc[best_model_name, 'BA_std']:.3f})")

    # ── A2. Per-polyamine model comparison ───────────────────────────────────
    pa_col_all = {}
    for ds in CRC_DATASETS:
        for kegg, col in polyamine_columns.get(ds, {}).items():
            pa_name = polyamine_display_name(col)
            pa_col_all[col] = pa_name

    pa_clf = clf_df[clf_df['Metabolite'].isin(pa_col_all)].copy()
    pa_clf['Polyamine'] = pa_clf['Metabolite'].map(pa_col_all)

    if not pa_clf.empty:
        pa_bench = (
            pa_clf.groupby(['Polyamine', 'Model'])
            .agg(BA=('balanced_accuracy_mean', 'mean'),
                 F1=('f1_macro_mean', 'mean'))
            .reset_index()
            .sort_values(['Polyamine', 'BA'], ascending=[True, False])
            .round(4)
        )
        print("\nPer-polyamine model comparison:")
        display(pa_bench)

        # Best model per polyamine
        best_pa_model = {}
        for (ds, met), grp in pa_clf.groupby(['Dataset', 'Metabolite']):
            best_row = grp.loc[grp['balanced_accuracy_mean'].idxmax()]
            pa_name  = polyamine_display_name(met)
            best_pa_model[(ds, pa_name)] = {
                'model': best_row['Model'],
                'BA':    round(best_row['balanced_accuracy_mean'], 3),
                'F1':    round(best_row['f1_macro_mean'], 3),
            }
    else:
        print("\n[INFO] No polyamine metabolites found in clf_df.")
        best_pa_model = {}

# ── A3. Literature concordance (external validation) ─────────────────────────
print("\n" + "=" * 60)
print("LITERATURE CONCORDANCE  (genus match with known producers)")
print("=" * 60)

lit_concordance_rows = []

for ds in CRC_DATASETS:
    for kegg, col in polyamine_columns.get(ds, {}).items():
        pa_name     = polyamine_display_name(col)
        known_genera = LITERATURE_PRODUCERS.get(pa_name, [])
        if not known_genera:
            continue

        try:
            y_labels = derive_source_labels(
                stage_corr[ds], transformed_species[ds],
                harmonized_meta[ds], col
            )
        except Exception as e:
            print(f"  [{ds}] {pa_name}: error — {e}")
            continue

        if y_labels is None or len(y_labels) == 0:
            print(f"  [{ds}] {pa_name}: no labels (< 20 samples)")
            continue

        freq       = y_labels.value_counts()
        n_total    = len(y_labels)
        lit_lower  = {g.lower() for g in known_genera}

        matched_n = sum(
            cnt for spc, cnt in freq.items()
            if extract_genus(spc).lower() in lit_lower
        )
        concordance = matched_n / n_total

        print(f"  [{ds}] {pa_name}: concordance={concordance:.1%} "
              f"({matched_n}/{n_total} samples, "
              f"known genera: {', '.join(known_genera)})")

        lit_concordance_rows.append({
            'Dataset':          ds,
            'Polyamine':        pa_name,
            'n_samples':        n_total,
            'lit_match_n':      matched_n,
            'lit_concordance':  round(concordance, 3),
            'known_genera':     '; '.join(known_genera),
        })

if lit_concordance_rows:
    lit_conc_df = pd.DataFrame(lit_concordance_rows)
    out_csv = CRC_RESULTS_DIR / 'tables' / 'polyamine_concordance.csv'
    lit_conc_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv.name}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(max(8, len(lit_concordance_rows) * 1.2), 5))
    labels  = [f"{r['Polyamine']}\n({r['Dataset'][:12]})"
               for r in lit_concordance_rows]
    vals    = [r['lit_concordance'] for r in lit_concordance_rows]
    colours = ['#1565C0' if v >= 0.5 else '#FF9800' for v in vals]

    ax.bar(range(len(vals)), vals, color=colours, edgecolor='white', linewidth=0.8)
    ax.axhline(0.5, ls='--', lw=1.2, color='#555', label='50% threshold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Literature concordance\n(fraction of samples with known-genus source)')
    ax.set_ylim(0, 1.05)
    ax.set_title('External Validation: Source Attribution Concordance\nwith Published Polyamine Producers')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    conc_fig = CLF_FIG_DIR / 'literature_concordance.png'
    plt.savefig(conc_fig, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Saved: literature_concordance.png")
else:
    print("\n[INFO] No literature concordance data computed.")
