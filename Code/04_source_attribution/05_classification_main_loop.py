

# --------------------------------------------------------------------------"""
Classification main loop — For each dataset × metabolite: derive labels, filter degenerate distributions, RF pre-screen (top 50 features, np.isin vectorised), run all 4 classifiers. Saves source_classification_results.csv.
"""


# ============================================================
# SOURCE SPECIES CLASSIFICATION -- Cell 4: Main loop
# ============================================================
MAX_FEATURES_CLF = 50
clf_results = []
CLF_FIG_DIR = CRC_RESULTS_DIR / "figures" / "classification"
CLF_FIG_DIR.mkdir(parents=True, exist_ok=True)

for ds in CRC_DATASETS:
    print(f"\n{'='*60}\nDataset: {ds}")

    # Candidate metabolites: union across all stage correlation results
    all_mtb_cols = set()
    for corr_df in stage_corr[ds].values():
        if not corr_df.empty:
            all_mtb_cols.update(corr_df['Metabolite'].unique())
    print(f"  {len(all_mtb_cols)} candidate metabolites")

    for met_col in sorted(all_mtb_cols):
        y_labels = derive_source_labels(
            stage_corr[ds], transformed_species[ds], harmonized_meta[ds], met_col)
        if y_labels is None:
            continue

        # Skip degenerate label distributions
        vc             = y_labels.value_counts()
        dominant_frac  = vc.iloc[0] / len(y_labels)
        n_viable       = (vc >= 20).sum()
        if dominant_frac > 0.90 or n_viable < 2:
            continue

        X, y, feat_names, le = prepare_clf_data(
            transformed_mtb[ds], harmonized_meta[ds], y_labels)

        # RF feature pre-screen -> top MAX_FEATURES_CLF
        rf_screen = RandomForestClassifier(
            n_estimators=100, max_depth=8, n_jobs=-1, random_state=42)
        rf_screen.fit(X, y)
        imp       = pd.Series(rf_screen.feature_importances_, index=feat_names)
        top_names = imp.nlargest(min(MAX_FEATURES_CLF, len(feat_names))).index.tolist()

        # Vectorised index lookup: np.isin O(n+m) vs list.index O(n*m)
        feat_arr = np.array(feat_names)
        top_idx  = np.flatnonzero(np.isin(feat_arr, top_names))
        X_sel    = X[:, top_idx]

        print(f"  {met_col}: {len(le.classes_)} source classes, n={len(y)}")
        for clf_name, clf in get_classifiers().items():
            scores = run_cv_clf(X_sel, y, clf)
            clf_results.append({
                'Dataset':   ds,
                'Metabolite': met_col,
                'Model':     clf_name,
                'n_samples': len(y),
                'n_classes': len(le.classes_),
                **scores,
            })

clf_df = pd.DataFrame(clf_results)
out_csv = CRC_RESULTS_DIR / "tables" / "source_classification_results.csv"
clf_df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")
display(clf_df.sort_values('balanced_accuracy_mean', ascending=False).head(20))
