

# --------------------------------------------------------------------------"""
Classification plots — Feature importance bar chart + training confusion matrix for the best metabolite per dataset (highest balanced accuracy with RandomForest).
"""


# ============================================================
# SOURCE SPECIES CLASSIFICATION -- Cell 5: Plots
# Feature importance + confusion matrix for best metabolite per dataset
# ============================================================
from sklearn.metrics import ConfusionMatrixDisplay

for ds in CRC_DATASETS:
    ds_rf = clf_df[(clf_df['Dataset'] == ds) & (clf_df['Model'] == 'RandomForest')]
    if ds_rf.empty:
        print(f"[{ds}] No RandomForest results -- skipping plots.")
        continue

    best_row = ds_rf.sort_values('balanced_accuracy_mean', ascending=False).iloc[0]
    met_col  = best_row['Metabolite']
    bal_acc  = best_row['balanced_accuracy_mean']
    print(f"[{ds}] Best metabolite: {met_col}  (balanced_acc={bal_acc:.3f})")

    y_labels = derive_source_labels(
        stage_corr[ds], transformed_species[ds], harmonized_meta[ds], met_col)
    if y_labels is None:
        print(f"  [{ds}] Could not derive labels for {met_col} -- skipping plots.")
        continue
    X, y, feat_names, le = prepare_clf_data(
        transformed_mtb[ds], harmonized_meta[ds], y_labels)

    rf_final = RandomForestClassifier(
        n_estimators=500, max_depth=10,
        class_weight='balanced', n_jobs=-1, random_state=42)
    rf_final.fit(X, y)

    imp = pd.Series(rf_final.feature_importances_, index=feat_names).nlargest(15)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: feature importance bar chart
    imp.sort_values().plot.barh(ax=axes[0], color='steelblue')
    axes[0].set_xlabel('Feature Importance')
    axes[0].set_title(f'Top 15 Features\n[{ds}] {met_col}')

    # Right: confusion matrix on training data (indicative)
    ConfusionMatrixDisplay.from_estimator(
        rf_final, X, y,
        display_labels=le.classes_,
        ax=axes[1],
        xticks_rotation=45,
        colorbar=False)
    axes[1].set_title(f'Confusion Matrix (train)\nbal_acc(CV)={bal_acc:.3f}')

    plt.tight_layout()
    safe_ds  = ds.replace('/', '_').replace(' ', '_')
    safe_met = met_col.replace('/', '_').replace(' ', '_')
    save_path = CLF_FIG_DIR / f"{safe_ds}__{safe_met}_clf.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_path}")