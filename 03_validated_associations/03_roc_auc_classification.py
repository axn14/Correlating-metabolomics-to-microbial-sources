

# --------------------------------------------------------------------------"""
ROC-AUC classification — Binarise metabolite log2 by median; stratified 10-fold OOF classification; compute AUC-ROC, AUPRC, F1, Accuracy, Recall, Precision. ROC and PR curves plotted per metabolite.
"""


# ============================================================
# FULL-COHORT ROC-AUC  (Causal direction: Species -> Metabolite)
# Research question: Can species abundance + disease status classify
#                    high vs low polyamine metabolite level?
# Strategy: groups=None -> all samples (disease + healthy)
#           Binarise metabolite log2 by median -> ROC-AUC via StratifiedKFold OOF
# ============================================================

try:
    average_precision_score  # already imported
except NameError:
    from sklearn.metrics import (accuracy_score, recall_score,
                                 precision_score, f1_score,
                                 average_precision_score)

if 'cv_df' not in dir() or cv_df is None or cv_df.empty:
    print('cv_df not found -- run the regression cell first.')
else:
    ROC_N_SPLITS_FC = 10
    TOP_METS_FC     = 5   # top metabolites by R²

    fc_auc_rows = []

    for ds in ACTIVE_DATASETS:
        ds_cv = cv_df[cv_df['Dataset'] == ds]
        if ds_cv.empty:
            continue

        best_per_met = ds_cv.loc[ds_cv.groupby('Target_Metabolite')['Mean_R2'].idxmax()]
        top_mets     = best_per_met.nlargest(TOP_METS_FC, 'Mean_R2')
        n_met = len(top_mets)
        if n_met == 0:
            continue

        fig = plt.figure(figsize=(6 * n_met, 11))
        gs  = gridspec.GridSpec(2, n_met, height_ratios=[2, 1], hspace=0.45, wspace=0.35)
        fig.suptitle(
            f'ROC-AUC — Full Cohort (Disease + Healthy)\n'
            f'Can species abundance + disease status classify metabolite level?\n{ds}',
            fontsize=12, y=1.02
        )

        for col, (_, row) in enumerate(top_mets.iterrows()):
            met_short  = row['Target_Metabolite']
            met_target = next((m for m in target_metabolites.get(ds, [])
                               if m[:50] == met_short), None)
            if met_target is None:
                continue

            # groups=None -> ALL samples (disease + healthy)
            X, y, feature_names, _ = prepare_ml_data_regression_rev(
                transformed_species[ds], harmonized_meta[ds],
                transformed_mtb[ds], met_target,
                groups=None, ds_name=ds
            )

            if len(X) < 20 or np.std(y) < 1e-10:
                continue

            y_bin = (y > np.median(y)).astype(int)
            if y_bin.sum() < 4 or (len(y_bin) - y_bin.sum()) < 4:
                continue

            scaler    = StandardScaler()
            X_scaled  = scaler.fit_transform(X)
            rf_screen = RandomForestRegressor(
                n_estimators=100, max_depth=8, n_jobs=-1, random_state=42
            )
            rf_screen.fit(X_scaled, y)
            importances = pd.Series(rf_screen.feature_importances_, index=feature_names)
            top_feats   = importances.nlargest(
                              min(MAX_FEATURES, len(feature_names))).index.tolist()
            sel_idx     = [feature_names.index(f) for f in top_feats]
            X_sel       = X_scaled[:, sel_idx]

            skf      = StratifiedKFold(n_splits=ROC_N_SPLITS_FC, shuffle=True, random_state=42)
            splits   = list(skf.split(X_sel, y_bin))
            ax_roc   = fig.add_subplot(gs[0, col])
            ax_bar   = fig.add_subplot(gs[1, col])
            model_aucs = {}
            mean_fpr   = np.linspace(0, 1, 200)

            for model_name, model in get_regressors().items():
                color       = MODEL_COLORS.get(model_name, '#555555')
                fold_tprs   = []
                fold_aucs   = []
                fold_auprcs = []
                fold_acc    = []
                fold_rec    = []
                fold_prec   = []
                fold_f1     = []

                for train_idx, test_idx in splits:
                    m = type(model)(**model.get_params())
                    m.fit(X_sel[train_idx], y[train_idx])
                    y_score = m.predict(X_sel[test_idx])
                    if len(np.unique(y_bin[test_idx])) < 2:
                        continue
                    fpr, tpr, _ = roc_curve(y_bin[test_idx], y_score)
                    interp_tpr  = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0
                    fold_tprs.append(interp_tpr)
                    fold_aucs.append(sk_auc(fpr, tpr))
                    fold_auprcs.append(average_precision_score(y_bin[test_idx], y_score))
                    y_pred_bin = (y_score >= np.median(y_score)).astype(int)
                    fold_acc.append(accuracy_score(y_bin[test_idx], y_pred_bin))
                    fold_rec.append(recall_score(y_bin[test_idx], y_pred_bin, zero_division=0))
                    fold_prec.append(precision_score(y_bin[test_idx], y_pred_bin, zero_division=0))
                    fold_f1.append(f1_score(y_bin[test_idx], y_pred_bin, zero_division=0))

                if not fold_aucs:
                    continue
                mean_tpr     = np.mean(fold_tprs, axis=0)
                mean_tpr[-1] = 1.0
                std_tpr      = np.std(fold_tprs, axis=0)
                mean_auc     = float(np.mean(fold_aucs))
                std_auc      = float(np.std(fold_aucs))
                model_aucs[model_name] = (mean_auc, std_auc)

                ax_roc.plot(mean_fpr, mean_tpr, color=color, lw=2,
                            label=f'{model_name}\nAUC={mean_auc:.3f}±{std_auc:.3f}')
                ax_roc.fill_between(mean_fpr, mean_tpr - std_tpr,
                                    mean_tpr + std_tpr, color=color, alpha=0.12)

                fc_auc_rows.append({
                    'Dataset':          ds,
                    'Target_Metabolite': met_short,
                    'Model':            model_name,
                    'Mean_AUC':         round(mean_auc, 4),
                    'Std_AUC':          round(std_auc,  4),
                    'N_samples':        len(y),
                    'Pos_rate':         round(float(y_bin.mean()), 3),
                    'Mean_AUPRC':       round(float(np.mean(fold_auprcs)), 4),
                    'Std_AUPRC':        round(float(np.std(fold_auprcs)),  4),
                    'Mean_Accuracy':    round(float(np.mean(fold_acc)),    4),
                    'Std_Accuracy':     round(float(np.std(fold_acc)),     4),
                    'Mean_Recall':      round(float(np.mean(fold_rec)),    4),
                    'Std_Recall':       round(float(np.std(fold_rec)),     4),
                    'Mean_Precision':   round(float(np.mean(fold_prec)),   4),
                    'Std_Precision':    round(float(np.std(fold_prec)),    4),
                    'Mean_F1':          round(float(np.mean(fold_f1)),     4),
                    'Std_F1':           round(float(np.std(fold_f1)),      4),
                    'Cohort':           'full',
                })

            ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.5)')
            ax_roc.set(xlim=[0, 1], ylim=[0, 1.02],
                       xlabel='False Positive Rate', ylabel='True Positive Rate')
            ax_roc.set_title(met_short[:38], fontsize=8, pad=4)
            ax_roc.legend(fontsize=6.5, loc='lower right', framealpha=0.7)
            ax_roc.grid(alpha=0.25)
            n_grp_feats = sum(1 for f in top_feats if f.startswith('Group_'))
            ax_roc.text(0.02, 0.96,
                        f'N={len(y)} | {n_grp_feats} group features',
                        transform=ax_roc.transAxes,
                        fontsize=6.5, va='top', color='dimgray')

            if model_aucs:
                names  = list(model_aucs.keys())
                aucs   = [model_aucs[n][0] for n in names]
                errs   = [model_aucs[n][1] for n in names]
                cbars  = [MODEL_COLORS.get(n, '#555') for n in names]
                bars   = ax_bar.bar(names, aucs, yerr=errs, color=cbars,
                                    capsize=4, edgecolor='white', linewidth=0.5)
                ax_bar.axhline(0.5, color='black', linestyle='--', linewidth=0.8)
                ax_bar.set_ylim(0, 1.1)
                ax_bar.set_ylabel('Mean AUC', fontsize=8)
                ax_bar.set_xticklabels(names, rotation=30, ha='right', fontsize=7)
                ax_bar.grid(axis='y', alpha=0.25)
                for bar, a in zip(bars, aucs):
                    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.02,
                                f'{a:.3f}', ha='center', va='bottom', fontsize=7)

        fig_path = CRC_RESULTS_DIR / 'figures' / 'ml' / f'{ds}_ROC_AUC_full_cohort.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f'Saved: {fig_path.name}')

    fc_auc_df = pd.DataFrame(fc_auc_rows)
    if not fc_auc_df.empty:
        print('\n' + '=' * 70)
        print('FULL-COHORT ROC-AUC SUMMARY  (Species -> Metabolite direction)')
        print('=' * 70)
        display(fc_auc_df.sort_values('Mean_AUC', ascending=False))
        fc_auc_df.to_csv(
            CRC_RESULTS_DIR / 'tables' / 'ml_classification_metrics_full_cohort.csv', index=False
        )
        print('Saved tables/ml_classification_metrics_full_cohort.csv')

        # Comparison heatmap (disease-only vs full-cohort) if auc_df exists
        if 'auc_df' in dir() and auc_df is not None and not auc_df.empty:
            comb = pd.concat([
                auc_df.assign(Cohort='disease_only').rename(
                    columns={'Target_Species': 'Target_Metabolite'}, errors='ignore'),
                fc_auc_df.assign(Cohort='full_cohort'),
            ])
            if 'Target_Metabolite' in comb.columns:
                pivot = comb.pivot_table(
                    index='Target_Metabolite', columns=['Model', 'Cohort'],
                    values='Mean_AUC', aggfunc='mean'
                )
                fig, ax = plt.subplots(
                    figsize=(max(8, len(pivot.columns) * 1.5), max(4, len(pivot) * 0.8))
                )
                im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=0.4, vmax=1.0, aspect='auto')
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_yticks(range(len(pivot.index)))
                ax.set_xticklabels(
                    [f'{m}\n({c})' for m, c in pivot.columns],
                    fontsize=8, rotation=45, ha='right'
                )
                ax.set_yticklabels([s[:40] for s in pivot.index], fontsize=8)
                plt.colorbar(im, ax=ax, label='Mean AUC', shrink=0.6)
                for i in range(len(pivot.index)):
                    for j in range(len(pivot.columns)):
                        v = pivot.values[i, j]
                        if not (isinstance(v, float) and v != v):
                            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=7,
                                    color='white' if v > 0.9 or v < 0.45 else 'black')
                ax.set_title(
                    'Mean AUC: Species -> Metabolite (median log2 split)',
                    fontsize=11
                )
                plt.tight_layout()
                cmp_path = (CRC_RESULTS_DIR / 'figures' / 'ml' /
                            'ROC_AUC_disease_vs_full_cohort.png')
                plt.savefig(cmp_path, dpi=150, bbox_inches='tight')
                plt.show()
                plt.close()
                print(f'Saved comparison heatmap: {cmp_path.name}')
    else:
        print('No AUC results computed.')
