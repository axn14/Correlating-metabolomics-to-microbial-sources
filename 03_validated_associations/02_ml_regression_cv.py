

# --------------------------------------------------------------------------"""
ML regression CV — Repeated 10×3 KFold CV predicting polyamine log2 abundance from species (log10) + disease status + confounders. RF pre-screen to top 80 features. Four models: RF, GB, ElasticNet, XGBoost.
"""


# === SELECT TARGET METABOLITES (Full-Cohort, Causal Direction: Species -> Metabolites) ===
# Polyamine metabolites are the primary targets; species abundance are the predictors.

target_metabolites = {}

for ds in CRC_DATASETS:
    pa_cols = list(polyamine_columns.get(ds, {}).values())
    avail_pa = [c for c in pa_cols if c in transformed_mtb[ds].columns]
    target_metabolites[ds] = avail_pa

    print(f"\nTarget metabolites for {ds}: {len(target_metabolites[ds])}")
    for i, m in enumerate(target_metabolites[ds], 1):
        print(f"  {i}. {m}")

# === FULL-COHORT REPEATED 5-FOLD CV (Species -> Metabolite) ===
# X = species (log10) + disease status + confounders  -->  y = metabolite (log2) abundance
print("\n" + "=" * 80)
print("REGRESSION (Full Cohort, Causal): Species + Disease Status -> Metabolite")
print("=" * 80)

cv_results = []
MAX_FEATURES = 80

for ds in CRC_DATASETS:
    if ds not in target_metabolites or not target_metabolites[ds]:
        continue

    print(f"\n{'='*60}")
    print(f"Dataset: {ds}  |  Groups: ALL (groups=None)")

    for met_target in target_metabolites[ds]:
        X, y, feature_names, _ = prepare_ml_data_regression_rev(
            transformed_species[ds], harmonized_meta[ds],
            transformed_mtb[ds], met_target,
            groups=None, ds_name=ds
        )

        if len(X) < 20 or np.std(y) < 1e-10:
            continue

        # Feature pre-selection: RF importance -> top MAX_FEATURES
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf_screen = RandomForestRegressor(
            n_estimators=100, max_depth=8, n_jobs=-1, random_state=42
        )
        rf_screen.fit(X_scaled, y)
        importances = pd.Series(rf_screen.feature_importances_, index=feature_names)
        top_features = importances.nlargest(min(MAX_FEATURES, len(feature_names))).index.tolist()
        sel_idx = [feature_names.index(f) for f in top_features]
        X_sel = X_scaled[:, sel_idx]

        # Count feature types
        n_spc  = sum(1 for f in top_features if not f.startswith('Group_') and not f.startswith('Conf_'))
        n_grp  = sum(1 for f in top_features if f.startswith('Group_'))
        n_conf = sum(1 for f in top_features if f.startswith('Conf_'))

        models = get_regressors()
        for model_name, model in models.items():
            scores = run_cv_reg(X_sel, y, model, n_splits=10, n_repeats=3)
            cv_results.append({
                'Dataset':               ds,
                'Target_Metabolite':     met_target[:50],
                'Model':                 model_name,
                'N_features':            X_sel.shape[1],
                'N_species_features':    n_spc,
                'N_group_features':      n_grp,
                'N_confounder_features': n_conf,
                'N_samples':             len(X_sel),
                'Mean_R2':               scores.mean(),
                'Std_R2':                scores.std(),
            })

        best_r2 = max(r['Mean_R2'] for r in cv_results[-len(models):])
        print(f"  {met_target[:45]}: best R2={best_r2:.3f}  "
              f"[{n_spc} spc, {n_grp} grp, {n_conf} conf features]")

cv_df = pd.DataFrame(cv_results)
print("\n" + "=" * 80)
print("Full-Cohort Regression CV Results (Species -> Metabolite):")
display(cv_df.sort_values('Mean_R2', ascending=False).head(20))
cv_df.to_csv(CRC_RESULTS_DIR / 'tables' / 'ml_regression_cv_results_full_cohort.csv', index=False)

# === HYPERPARAMETER TUNING for best metabolite per dataset ===
print("\n" + "=" * 80)
print("HYPERPARAMETER TUNING (RandomizedSearchCV on best metabolite per dataset)")
print("=" * 80)

tuning_results = []

if not cv_df.empty:
    param_dist = {
        'n_estimators':     [200, 500, 800],
        'max_depth':        [5, 8, 12, None],
        'min_samples_leaf': [3, 5, 10],
        'max_features':     ['sqrt', 'log2', 0.5],
    }

    for ds in CRC_DATASETS:
        ds_cv = cv_df[cv_df['Dataset'] == ds]
        if ds_cv.empty:
            continue

        best_row = ds_cv.loc[ds_cv['Mean_R2'].idxmax()]
        met_target = [m for m in target_metabolites.get(ds, [])
                      if m[:50] == best_row['Target_Metabolite']]
        if not met_target:
            continue
        met_target = met_target[0]

        print(f"\n  Tuning RF for {ds[:20]} | {met_target[:40]} "
              f"(baseline R2={best_row['Mean_R2']:.3f})")

        X, y, feature_names, _ = prepare_ml_data_regression_rev(
            transformed_species[ds], harmonized_meta[ds],
            transformed_mtb[ds], met_target,
            groups=None, ds_name=ds
        )
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        rf_screen = RandomForestRegressor(n_estimators=100, max_depth=8, n_jobs=-1, random_state=42)
        rf_screen.fit(X_scaled, y)
        imp = pd.Series(rf_screen.feature_importances_, index=feature_names)
        top_feat = imp.nlargest(min(MAX_FEATURES, len(feature_names))).index.tolist()
        sel_idx = [feature_names.index(f) for f in top_feat]
        X_sel = X_scaled[:, sel_idx]

        search = RandomizedSearchCV(
            RandomForestRegressor(n_jobs=-1, random_state=42),
            param_dist, n_iter=20, cv=5, scoring='r2',
            random_state=42, n_jobs=-1
        )
        search.fit(X_sel, y)
        print(f"  Best params: {search.best_params_}")
        print(f"  Tuned R2: {search.best_score_:.3f}  (baseline: {best_row['Mean_R2']:.3f})")

        tuning_results.append({
            'Dataset':     ds,
            'Metabolite':  met_target[:50],
            'Baseline_R2': best_row['Mean_R2'],
            'Tuned_R2':    search.best_score_,
            'Best_params': str(search.best_params_),
        })

if tuning_results:
    tuning_df = pd.DataFrame(tuning_results)
    display(tuning_df)
    tuning_df.to_csv(
        CRC_RESULTS_DIR / 'tables' / 'ml_hyperparameter_tuning_full_cohort.csv', index=False)

# Overfitting guard
OVERFIT_R2_THRESHOLD = 0.75
OVERFIT_STD_THRESHOLD = 0.08

overfit_datasets = set()
print("\n" + "=" * 60)
print("OVERFITTING CHECK")
print("=" * 60)
for ds in CRC_DATASETS:
    ds_cv = cv_df[cv_df['Dataset'] == ds]
    if ds_cv.empty:
        print(f"  {ds}: no results")
        continue
    best_row = ds_cv.loc[ds_cv['Mean_R2'].idxmax()]
    max_r2 = best_row['Mean_R2']
    std_at_max = best_row['Std_R2']
    if max_r2 > OVERFIT_R2_THRESHOLD and std_at_max < OVERFIT_STD_THRESHOLD:
        print(f"  OVERFIT: {ds}  best R2={max_r2:.3f}  Std={std_at_max:.3f}  "
              f"[{best_row['Target_Metabolite'][:40]}]")
        overfit_datasets.add(ds)
    else:
        print(f"  OK:      {ds}  best R2={max_r2:.3f}  Std={std_at_max:.3f}")

ACTIVE_DATASETS = [ds for ds in CRC_DATASETS if ds not in overfit_datasets]
