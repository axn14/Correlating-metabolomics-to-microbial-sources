

# --------------------------------------------------------------------------"""
SHAP feature importance — TreeExplainer (or LinearExplainer) for the best model per metabolite; beeswarm summary plots identifying top species drivers of polyamine prediction.
"""


# ============================================================
# FULL-COHORT SHAP  (Causal direction: Species -> Metabolite)
# Shows which species (and group dummies) drive polyamine metabolite
# prediction across the full sample set.
# High species SHAP importance supports that species as a producer.
# ============================================================

if not HAS_SHAP:
    print('SHAP not available -- pip install shap')
elif 'cv_df' not in dir() or cv_df is None or cv_df.empty:
    print('cv_df not found -- run the regression cell first.')
else:
    fc_shap_results = {}

    # Select best model per (dataset, metabolite) from full-cohort AUC
    fc_best_model_map = {}
    if 'fc_auc_df' in dir() and not fc_auc_df.empty:
        for (ds_key, met_key), grp in fc_auc_df.groupby(['Dataset', 'Target_Metabolite']):
            best_row = grp.loc[grp['Mean_AUC'].idxmax()]
            fc_best_model_map[(ds_key, met_key)] = best_row['Model']
        print('Full-cohort best model per metabolite (by AUC-ROC):')
        for k, v in fc_best_model_map.items():
            print(f'  {k[0][:15]} | {k[1][:40]} -> {v}')
    else:
        print('fc_auc_df not available -- defaulting to RandomForest')

    for ds in ACTIVE_DATASETS:
        ds_cv = cv_df[cv_df['Dataset'] == ds]
        if ds_cv.empty:
            continue

        print(f"\n{'=' * 60}")
        print(f'Full-Cohort SHAP [{ds}]  --  ALL groups (disease + healthy)')
        print(f"{'=' * 60}")

        best_per_met = ds_cv.loc[ds_cv.groupby('Target_Metabolite')['Mean_R2'].idxmax()]
        top3         = best_per_met.nlargest(3, 'Mean_R2')

        for _, row in top3.iterrows():
            met_short  = row['Target_Metabolite']
            met_target = next((m for m in target_metabolites.get(ds, [])
                               if m[:50] == met_short), None)
            if met_target is None:
                continue

            print(f"\n  [{ds[:15]}] {met_target[:50]} (CV R²={row['Mean_R2']:.3f})")

            # groups=None -> ALL samples (disease + healthy)
            X, y, feature_names, _ = prepare_ml_data_regression_rev(
                transformed_species[ds], harmonized_meta[ds],
                transformed_mtb[ds], met_target,
                groups=None, ds_name=ds
            )
            print(f'  Samples (full cohort): {X.shape[0]},  Features: {X.shape[1]}')

            if X.shape[0] < 20:
                continue

            scaler    = StandardScaler()
            X_scaled  = scaler.fit_transform(X)
            rf_screen = RandomForestRegressor(
                n_estimators=100, max_depth=8, n_jobs=-1, random_state=42
            )
            rf_screen.fit(X_scaled, y)
            importances  = pd.Series(rf_screen.feature_importances_, index=feature_names)
            top_features = importances.nlargest(
                               min(MAX_FEATURES, len(feature_names))).index.tolist()
            sel_idx  = [feature_names.index(f) for f in top_features]
            X_sel    = X_scaled[:, sel_idx]

            # Select and train best model for SHAP
            fc_best_name   = fc_best_model_map.get((ds, met_short), 'RandomForest')
            all_regressors = get_regressors()
            if fc_best_name in all_regressors:
                base   = all_regressors[fc_best_name]
                params = base.get_params()
                if 'n_estimators' in params:
                    params['n_estimators'] = max(params['n_estimators'], 300)
                final_model = type(base)(**params)
            else:
                final_model = RandomForestRegressor(
                    n_estimators=500, max_depth=10, min_samples_leaf=5,
                    n_jobs=-1, random_state=42
                )
                fc_best_name = 'RandomForest'

            final_model.fit(X_sel, y)
            print(f'  SHAP model (full cohort): {fc_best_name}')

            try:
                from sklearn.linear_model import ElasticNet as _EN
                if isinstance(final_model, _EN):
                    explainer   = shap.LinearExplainer(final_model, X_sel)
                    shap_values = explainer.shap_values(X_sel)
                    base_value  = float(explainer.expected_value)
                else:
                    explainer   = shap.TreeExplainer(final_model)
                    shap_values = explainer.shap_values(X_sel)
                    base_value  = float(explainer.expected_value)
                explanation = shap.Explanation(
                    values        = shap_values,
                    base_values   = np.full(shap_values.shape[0], base_value),
                    data          = X_sel,
                    feature_names = top_features
                )

                safe_name = met_target[:30].replace('/', '_').replace(';', '_')
                ds_short  = ds[:15]

                # 1. Beeswarm
                fig, ax = plt.subplots(figsize=(12, 8))
                shap.plots.beeswarm(explanation, max_display=20, show=False)
                plt.title(f'Beeswarm (full cohort) [{ds_short}]: {met_target[:40]}',
                          fontsize=11)
                plt.tight_layout()
                plt.savefig(
                    CRC_RESULTS_DIR / 'figures' / 'ml' /
                    f'{ds}_SHAP_beeswarm_full_{safe_name}.png',
                    dpi=150, bbox_inches='tight')
                plt.show(); plt.close()

                # 2. Bar plot (mean |SHAP|)
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.plots.bar(explanation, max_display=20, show=False)
                plt.title(f'Feature Importance (full cohort) [{ds_short}]: {met_target[:40]}',
                          fontsize=11)
                plt.tight_layout()
                plt.savefig(
                    CRC_RESULTS_DIR / 'figures' / 'ml' /
                    f'{ds}_SHAP_bar_full_{safe_name}.png',
                    dpi=150, bbox_inches='tight')
                plt.show(); plt.close()

                # 3. Waterfall (highest + lowest predicted sample)
                y_pred = final_model.predict(X_sel)
                for label, idx in [('highest', np.argmax(y_pred)),
                                   ('lowest',  np.argmin(y_pred))]:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.plots.waterfall(explanation[idx], max_display=15, show=False)
                    plt.title(
                        f'Waterfall ({label}, full cohort) [{ds_short}]: {met_target[:35]}',
                        fontsize=10)
                    plt.tight_layout()
                    plt.savefig(
                        CRC_RESULTS_DIR / 'figures' / 'ml' /
                        f'{ds}_SHAP_waterfall_{label}_full_{safe_name}.png',
                        dpi=150, bbox_inches='tight')
                    plt.show(); plt.close()

                # 4. Dependence plots (top 2 features)
                mean_abs = np.abs(shap_values).mean(axis=0)
                for feat_idx in np.argsort(mean_abs)[::-1][:2]:
                    feat_name = top_features[feat_idx]
                    fig, ax = plt.subplots(figsize=(8, 6))
                    shap.plots.scatter(explanation[:, feat_idx],
                                       color=explanation, show=False)
                    plt.title(
                        f'Dependence: {feat_name[:40]} (full cohort) [{ds_short}]',
                        fontsize=10)
                    plt.tight_layout()
                    safe_feat = feat_name[:20].replace('/', '_').replace(';', '_')
                    plt.savefig(
                        CRC_RESULTS_DIR / 'figures' / 'ml' /
                        f'{ds}_SHAP_dep_{safe_feat}_full_{safe_name}.png',
                        dpi=150, bbox_inches='tight')
                    plt.show(); plt.close()

                # Summary: feature type breakdown
                top_idx = np.argsort(mean_abs)[::-1]
                print(f'  Top 10 features by |SHAP| (species -> {met_target[:25]}):')
                for i in top_idx[:10]:
                    f = top_features[i]
                    ftype = ('GROUP' if f.startswith('Group_') else
                             'CONF'  if f.startswith('Conf_')  else 'SPECIES')
                    marker = ' <-- disease status' if ftype == 'GROUP' else ''
                    print(f'    [{ftype}] {f[:55]}: |SHAP|={mean_abs[i]:.4f}{marker}')

                # Proportion: group vs species features
                group_imp   = sum(mean_abs[i] for i in top_idx[:20]
                                  if top_features[i].startswith('Group_'))
                species_imp = sum(mean_abs[i] for i in top_idx[:20]
                                  if not top_features[i].startswith('Group_')
                                  and not top_features[i].startswith('Conf_'))
                total_imp   = sum(mean_abs[i] for i in top_idx[:20])
                if total_imp > 0:
                    print(f'  Disease-status features: {group_imp/total_imp*100:.1f}%'
                          f' of top-20 SHAP importance')
                    print(f'  Species features:        {species_imp/total_imp*100:.1f}%'
                          f' of top-20 SHAP importance')

                fc_shap_results[(ds, met_target)] = {
                    'top_features':         [(top_features[i], float(mean_abs[i]))
                                             for i in top_idx[:20]],
                    'group_shap_fraction':  group_imp / total_imp if total_imp > 0 else 0,
                    'species_shap_fraction': species_imp / total_imp if total_imp > 0 else 0,
                }

            except Exception as e:
                print(f'  SHAP failed: {e}')
                import traceback; traceback.print_exc()

    print(f'\nFull-cohort SHAP complete. {len(fc_shap_results)} metabolites analysed.')
    if fc_shap_results:
        print('\nDisease-status SHAP fraction summary:')
        for (ds, met), r in fc_shap_results.items():
            frac = r.get('group_shap_fraction', 0)
            spc_frac = r.get('species_shap_fraction', 0)
            print(f'  [{ds[:20]}] {met[:35]}: '
                  f'group={frac*100:.1f}%  species={spc_frac*100:.1f}% of top-20 SHAP')
