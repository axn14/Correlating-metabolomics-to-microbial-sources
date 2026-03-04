

# --------------------------------------------------------------------------"""
Feature matrix builder — Builds X (log2 metabolites + one-hot Study.Group) and integer-encoded y (source species labels). Drops classes with <5 samples. StandardScaler applied. Returns X, y, feature_cols, LabelEncoder.
"""


# ============================================================
# SOURCE SPECIES CLASSIFICATION -- Cell 2: Feature matrix builder
# ============================================================
def prepare_clf_data(transformed_mtb_ds, harmonized_meta_ds, y_labels):
    """
    Build X (all log2 metabolites + one-hot disease stage) and
    integer-encoded y (source species label).
    Returns: X (np.ndarray), y (np.ndarray), feature_cols (list), le (LabelEncoder)
    """
    meta_sub = harmonized_meta_ds[['Sample', 'Study.Group']].copy()
    group_dummies = pd.get_dummies(meta_sub['Study.Group'], prefix='Group')
    meta_enc = pd.concat([meta_sub[['Sample']], group_dummies], axis=1)

    merged = transformed_mtb_ds.merge(meta_enc, on='Sample').set_index('Sample')
    common = merged.index.intersection(y_labels.index)
    X_df   = merged.loc[common]
    y_raw  = y_labels.loc[common]

    # Drop classes with fewer than 5 samples so XGBoost never sees a label gap
    # (StratifiedKFold with n_splits=5 requires >= 5 samples per class)
    vc = y_raw.value_counts()
    valid_cls = vc[vc >= 5].index
    mask = y_raw.isin(valid_cls)
    X_df  = X_df[mask]
    y_raw = y_raw[mask]

    feature_cols = X_df.columns.tolist()
    X  = StandardScaler().fit_transform(X_df.fillna(0))
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    return X, y, feature_cols, le

print("prepare_clf_data defined.")