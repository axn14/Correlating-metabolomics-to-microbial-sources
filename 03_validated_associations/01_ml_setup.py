

# --------------------------------------------------------------------------"""
ML setup — Import sklearn/xgboost/shap; define feature matrix builder (prepare_ml_data_regression); model colour palette; dataset-specific confounders.
"""


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import (
    cross_val_score, KFold, RepeatedKFold, RandomizedSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    r2_score, mean_squared_error,
    roc_curve, auc as sk_auc,
    accuracy_score, recall_score, precision_score, f1_score, average_precision_score,
)
from sklearn.preprocessing import StandardScaler
from matplotlib import gridspec

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    print("xgboost not installed -- skipping XGBoost.")
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
    print(f"SHAP version {shap.__version__} loaded")
except ImportError:
    HAS_SHAP = False
    print("shap not installed -- SHAP interpretation will be skipped.")

import os
os.makedirs(CRC_RESULTS_DIR / 'figures' / 'ml', exist_ok=True)

# Dataset-specific extra confounders (same as partial correlations)
EXTRA_CONFOUNDERS = {
    'ERAWIJANTARI-GASTRIC-CANCER-2020': [
        'Gastric acid medication', 'Analgesic', 'Anticoagulant', 'DiabetesMed',
    ],
    'YACHIDA-CRC-2019': [
        'Brinkman Index', 'Alcohol',
    ],

}

MODEL_COLORS = {
    'RandomForest':     '#2980b9',
    'GradientBoosting': '#27ae60',
    'ElasticNet':       '#e74c3c',
    'XGBoost':          '#8e44ad',
}


def prepare_ml_data_regression(mtb_df, meta_df, species_df, target_species_col,
                                groups=None, ds_name=None):
    """Prepare X (metabolites + disease status + confounders) and y (species abundance).

    Features include:
    - Metabolite abundances (log2-transformed)
    - One-hot encoded Study.Group (disease status)
    - Standard confounders: Age, BMI, Gender (median/mode imputed)
    - Dataset-specific confounders: medications, smoking, alcohol

    Returns: X, y, feature_names
    """
    merged = mtb_df.copy()

    # --- One-hot encoded Study.Group ---
    meta_sub = meta_df[['Sample', 'Study.Group']].copy()
    if groups is not None:
        meta_sub = meta_sub[meta_sub['Study.Group'].isin(groups)]
    group_dummies = pd.get_dummies(meta_sub['Study.Group'], prefix='Group')
    meta_encoded = pd.concat([meta_sub[['Sample']], group_dummies], axis=1)
    merged = merged.merge(meta_encoded, on='Sample')

    # --- Standard confounders (Age, BMI, Gender) with imputation ---
    conf_df = meta_df[['Sample']].copy()

    if meta_df['Age'].notna().mean() > 0.3:
        conf_df['Conf_Age'] = meta_df['Age'].fillna(meta_df['Age'].median())

    if meta_df['BMI'].notna().mean() > 0.3:
        conf_df['Conf_BMI'] = meta_df['BMI'].fillna(meta_df['BMI'].median())

    if meta_df['Gender'].notna().mean() > 0.3:
        gender_num = meta_df['Gender'].map({'Male': 0, 'Female': 1})
        conf_df['Conf_Gender'] = gender_num.fillna(gender_num.mode().iloc[0] if not gender_num.mode().empty else 0)

    # --- Dataset-specific confounders ---
    if ds_name is not None:
        extra_cols = EXTRA_CONFOUNDERS.get(ds_name, [])
        if extra_cols:
            try:
                raw_meta = pd.read_csv(DATA_DIR / f'{ds_name} metadata.tsv', sep='\t')
                for col in extra_cols:
                    if col not in raw_meta.columns:
                        continue
                    if raw_meta[col].notna().mean() < 0.3:
                        continue
                    _BINARY_MAPS_ML = [
                        {'Yes': 1, 'No': 0},
                        {'Smoke': 1, 'Non-smoke': 0},
                    ]
                    safe_name = f'Conf_{col.replace(" ", "_").replace("/", "_")}'
                    if raw_meta[col].dtype == 'object':
                        mapped = None
                        for bmap in _BINARY_MAPS_ML:
                            if set(raw_meta[col].dropna().unique()) <= set(bmap.keys()) | {'Missing'}:
                                mapped = raw_meta[col].map(bmap)
                                break
                        if mapped is not None:
                            conf_df = conf_df.merge(
                                pd.DataFrame({'Sample': raw_meta['Sample'], safe_name: mapped.fillna(0)}),
                                on='Sample', how='left'
                            )
                        else:
                            numeric = pd.to_numeric(raw_meta[col], errors='coerce')
                            if numeric.notna().mean() > 0.3:
                                conf_df = conf_df.merge(
                                    pd.DataFrame({'Sample': raw_meta['Sample'],
                                                  safe_name: numeric.fillna(numeric.median())}),
                                    on='Sample', how='left'
                                )
                    else:
                        conf_df = conf_df.merge(
                            pd.DataFrame({'Sample': raw_meta['Sample'],
                                          safe_name: raw_meta[col].fillna(raw_meta[col].median())}),
                            on='Sample', how='left'
                        )
            except FileNotFoundError:
                pass

    merged = merged.merge(conf_df, on='Sample', how='left')

    # --- Target species ---
    merged = merged.merge(species_df[['Sample', target_species_col]], on='Sample')
    merged = merged.dropna(subset=[target_species_col]).reset_index(drop=True)

    feature_cols = [c for c in merged.columns if c not in ('Sample', target_species_col)]
    # Fill any remaining NaN in features
    merged[feature_cols] = merged[feature_cols].fillna(0)

    X = merged[feature_cols].values
    y = merged[target_species_col].values

    return X, y, feature_cols


def get_regressors():
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=500, max_depth=10, min_samples_leaf=5,
            n_jobs=-1, random_state=42
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            min_samples_leaf=5, subsample=0.8, random_state=42
        ),
        'ElasticNet': ElasticNet(
            alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42
        ),
    }
    if HAS_XGB:
        models['XGBoost'] = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0
        )
    return models


def run_cv_reg(X, y, model, n_splits=10, n_repeats=3):
    """Run repeated cross-validation and return R2 scores."""
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    return cross_val_score(model, X, y, cv=rkf, scoring='r2')

def prepare_ml_data_regression_rev(species_df, meta_df, mtb_df, target_metabolite_col,
                                    groups=None, ds_name=None):
    """Prepare X (species + disease status + confounders) and y (metabolite abundance).

    Correct causal direction: Species -> Metabolites (microbial producers drive metabolite levels).

    Features:
    - Species abundances (log10-transformed)
    - One-hot encoded Study.Group (disease status)
    - Standard confounders: Age, BMI, Gender (median/mode imputed)
    - Dataset-specific confounders: medications, smoking, alcohol

    Returns: X (ndarray), y (ndarray), feature_names (list[str])
    """
    merged = species_df.copy()

    # One-hot Study.Group
    meta_sub = meta_df[['Sample', 'Study.Group']].copy()
    if groups is not None:
        meta_sub = meta_sub[meta_sub['Study.Group'].isin(groups)]
    group_dummies = pd.get_dummies(meta_sub['Study.Group'], prefix='Group')
    meta_encoded = pd.concat([meta_sub[['Sample']], group_dummies], axis=1)
    merged = merged.merge(meta_encoded, on='Sample')

    # Standard confounders
    conf_df = meta_df[['Sample']].copy()
    if meta_df['Age'].notna().mean() > 0.3:
        conf_df['Conf_Age'] = meta_df['Age'].fillna(meta_df['Age'].median())
    if meta_df['BMI'].notna().mean() > 0.3:
        conf_df['Conf_BMI'] = meta_df['BMI'].fillna(meta_df['BMI'].median())
    if meta_df['Gender'].notna().mean() > 0.3:
        gender_num = meta_df['Gender'].map({'Male': 0, 'Female': 1})
        conf_df['Conf_Gender'] = gender_num.fillna(
            gender_num.mode().iloc[0] if not gender_num.mode().empty else 0)

    # Dataset-specific confounders
    if ds_name is not None:
        extra_cols = EXTRA_CONFOUNDERS.get(ds_name, [])
        if extra_cols:
            try:
                raw_meta = pd.read_csv(DATA_DIR / f'{ds_name} metadata.tsv', sep='\t')
                for col in extra_cols:
                    if col not in raw_meta.columns or raw_meta[col].notna().mean() < 0.3:
                        continue
                    _MAPS = [{'Yes': 1, 'No': 0}, {'Smoke': 1, 'Non-smoke': 0}]
                    safe_name = f'Conf_{col.replace(" ", "_").replace("/", "_")}'
                    if raw_meta[col].dtype == 'object':
                        mapped = None
                        for bmap in _MAPS:
                            if set(raw_meta[col].dropna().unique()) <= set(bmap.keys()) | {'Missing'}:
                                mapped = raw_meta[col].map(bmap)
                                break
                        if mapped is not None:
                            conf_df = conf_df.merge(
                                pd.DataFrame({'Sample': raw_meta['Sample'],
                                              safe_name: mapped.fillna(0)}),
                                on='Sample', how='left')
                        else:
                            numeric = pd.to_numeric(raw_meta[col], errors='coerce')
                            if numeric.notna().mean() > 0.3:
                                conf_df = conf_df.merge(
                                    pd.DataFrame({'Sample': raw_meta['Sample'],
                                                  safe_name: numeric.fillna(numeric.median())}),
                                    on='Sample', how='left')
                    else:
                        conf_df = conf_df.merge(
                            pd.DataFrame({'Sample': raw_meta['Sample'],
                                          safe_name: raw_meta[col].fillna(raw_meta[col].median())}),
                            on='Sample', how='left')
            except FileNotFoundError:
                pass

    merged = merged.merge(conf_df, on='Sample', how='left')

    # Target metabolite
    merged = merged.merge(mtb_df[['Sample', target_metabolite_col]], on='Sample')
    merged = merged.dropna(subset=[target_metabolite_col]).reset_index(drop=True)

    feature_cols = [c for c in merged.columns
                    if c not in ('Sample', target_metabolite_col)]
    merged[feature_cols] = merged[feature_cols].fillna(0)

    X = merged[feature_cols].values
    y = merged[target_metabolite_col].values
    return X, y, feature_cols, merged['Sample'].values


print("ML regression utilities ready (including reversed-direction function).")
print(f"Models: RF, GradientBoosting, ElasticNet" + (", XGBoost" if HAS_XGB else ""))