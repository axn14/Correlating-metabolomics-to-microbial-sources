

# --------------------------------------------------------------------------"""
Source label derivation — For each sample, assign the highest-CLR-abundance species among positively correlated species in that sample's disease stage. NumPy-vectorised (np.argmax replaces iterrows). Returns None if <20 labelled samples.
"""


# ============================================================
# SOURCE SPECIES CLASSIFICATION -- Cell 1: Imports & label derivation
# ============================================================
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import balanced_accuracy_score, f1_score
from xgboost import XGBClassifier

def derive_source_labels(stage_corr_ds, transformed_species_ds, harmonized_meta_ds,
                          metabolite_col, q_thresh=0.05):
    """
    For each sample, look up correlations for that sample's disease stage and
    return the highest-CLR-abundance positively-correlated species as the label.
    Returns pd.Series indexed by Sample, or None if fewer than 20 labelled samples.

    Vectorised with numpy: np.argmax over the full sub-matrix replaces the per-row loop.
    """
    labels = {}

    for stage, corr_df in stage_corr_ds.items():
        if corr_df.empty:
            continue
        sig = corr_df[
            (corr_df['Metabolite'] == metabolite_col) &
            (corr_df['QValue']     < q_thresh) &
            (corr_df['Rho']        > 0)
        ]['Species'].tolist()

        available = [s for s in sig if s in transformed_species_ds.columns]
        if len(available) < 2:
            continue

        stage_samples = harmonized_meta_ds.loc[
            harmonized_meta_ds['Study.Group'] == stage, 'Sample'
        ].values
        sub = transformed_species_ds[
            transformed_species_ds['Sample'].isin(stage_samples)
        ][['Sample'] + available].set_index('Sample')

        # Vectorised: one C-level argmax scan over the whole sub-matrix
        vals    = sub.to_numpy()                          # (n_samples, n_cols)
        col_idx = np.argmax(vals, axis=1)                # argmax column per row
        labels.update(dict(zip(sub.index, sub.columns[col_idx])))

    if len(labels) < 20:
        return None

    return pd.Series(labels, name='source_species')

print("derive_source_labels defined (numpy-vectorised).")
