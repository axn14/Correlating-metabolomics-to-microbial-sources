

# --------------------------------------------------------------------------"""
Polyamine microbial source attribution — Applies best-validated classifier labels to all polyamines. Horizontal bar charts (green=known producer, grey=novel). Saves polyamine_source_attribution.csv.
"""


# ============================================================
# POLYAMINE MICROBIAL SOURCE ATTRIBUTION
# Applies the best-validated classifier to identify microbial
# origin of polyamines in each dataset.
# Colours bars: green = genus matches literature, grey = novel
# ============================================================

# Identify best model from benchmarking (falls back to RandomForest)
_best_model = best_model_name if 'best_model_name' in dir() else 'RandomForest'
print(f"Using best model: {_best_model}")
print("=" * 60)

pa_attribution_rows = []

for ds in CRC_DATASETS:
    pa_cols = polyamine_columns.get(ds, {})
    if not pa_cols:
        continue
    print(f"\n{ds}")

    for kegg, col in pa_cols.items():
        pa_name     = polyamine_display_name(col)
        known_genera = {g.lower() for g in LITERATURE_PRODUCERS.get(pa_name, [])}

        # ── Derive per-sample source labels ──────────────────────────────────
        try:
            y_labels = derive_source_labels(
                stage_corr[ds], transformed_species[ds],
                harmonized_meta[ds], col
            )
        except Exception as e:
            print(f"  [{pa_name}] error in label derivation: {e}")
            continue

        if y_labels is None or len(y_labels) < 10:
            print(f"  [{pa_name}] insufficient labels — skipping")
            continue

        freq_norm = y_labels.value_counts(normalize=True)
        freq_abs  = y_labels.value_counts()
        n_total   = len(y_labels)

        # ── Retrieve best-model CV performance for this polyamine ─────────────
        ba_score = f1_score_val = float('nan')
        if 'clf_df' in dir() and not clf_df.empty:
            pa_rows = clf_df[
                (clf_df['Dataset']    == ds) &
                (clf_df['Metabolite'] == col) &
                (clf_df['Model']      == _best_model)
            ]
            if not pa_rows.empty:
                ba_score     = pa_rows['balanced_accuracy_mean'].iloc[0]
                f1_score_val = pa_rows['f1_macro_mean'].iloc[0]

        # ── Literature concordance ────────────────────────────────────────────
        matched_n = sum(
            cnt for spc, cnt in freq_abs.items()
            if extract_genus(spc).lower() in known_genera
        )
        concordance = matched_n / n_total if n_total > 0 else 0.0

        print(f"  [{pa_name}] n={n_total}, classes={len(freq_norm)}, "
              f"BA={ba_score:.3f}, lit_concordance={concordance:.1%}")

        # ── Bar chart: source species attribution ─────────────────────────────
        top_spc  = freq_norm.head(15)
        genera   = [extract_genus(s) for s in top_spc.index]
        colours  = ['#2E7D32' if g.lower() in known_genera else '#90A4AE'
                    for g in genera]

        fig, ax = plt.subplots(figsize=(9, max(4, len(top_spc) * 0.38)))
        bars = ax.barh(range(len(top_spc)), top_spc.values * 100,
                       color=colours, edgecolor='white', linewidth=0.6)

        ax.set_yticks(range(len(top_spc)))
        ax.set_yticklabels(
            [s.split('|')[-1].replace('_', ' ') for s in top_spc.index],
            fontsize=8)
        ax.set_xlabel('% of labelled samples attributed to species')
        ax.set_title(
            f'{pa_name} — Microbial Source Attribution\n'
            f'{ds}\n'
            f'Model: {_best_model}  ·  BA={ba_score:.3f}  ·  '
            f'Lit. concordance={concordance:.1%}',
            fontsize=9)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Legend
        known_patch = mpatches.Patch(color='#2E7D32', label='Known producer (literature)')
        novel_patch = mpatches.Patch(color='#90A4AE', label='Novel candidate')
        ax.legend(handles=[known_patch, novel_patch], fontsize=8,
                  loc='lower right')

        plt.tight_layout()
        safe_ds  = ds.replace('/', '_').replace(' ', '_')
        safe_pa  = pa_name.replace(' ', '_').replace('-', '')
        fig_path = CLF_FIG_DIR / f'{safe_ds}__{safe_pa}_attribution.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"    Saved: {fig_path.name}")

        # ── Collect summary rows ──────────────────────────────────────────────
        for spc, frac in freq_norm.items():
            genus    = extract_genus(spc)
            lit_known = genus.lower() in known_genera
            pa_attribution_rows.append({
                'Dataset':          ds,
                'Polyamine':        pa_name,
                'Source_Species':   spc,
                'Genus':            genus,
                'Lit_Known':        lit_known,
                'Pct_Samples':      round(frac * 100, 2),
                'BA_best_model':    round(ba_score, 4) if not np.isnan(ba_score) else None,
                'F1_best_model':    round(f1_score_val, 4) if not np.isnan(f1_score_val) else None,
                'Lit_Concordance':  round(concordance, 3),
            })

# ── Save attribution table ────────────────────────────────────────────────────
if pa_attribution_rows:
    pa_attr_df = pd.DataFrame(pa_attribution_rows)
    out_csv = CRC_RESULTS_DIR / 'tables' / 'polyamine_source_attribution.csv'
    pa_attr_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv.name}")

    print("\n" + "=" * 60)
    print("TOP PREDICTED MICROBIAL SOURCES PER POLYAMINE")
    print("=" * 60)
    top_per_pa = (
        pa_attr_df.sort_values('Pct_Samples', ascending=False)
        .groupby(['Dataset', 'Polyamine'])
        .head(3)
        [['Dataset', 'Polyamine', 'Source_Species', 'Genus',
          'Lit_Known', 'Pct_Samples', 'BA_best_model', 'Lit_Concordance']]
    )
    display(top_per_pa)
