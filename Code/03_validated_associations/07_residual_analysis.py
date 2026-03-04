

# --------------------------------------------------------------------------
"""
OOF residual diagnostics — 5-fold CV residuals for top 3 metabolites per dataset. 6-panel figure: residuals vs fitted, Q-Q, predicted vs actual, histogram, scale-location, residuals by group.
"""


# ============================================================
# Residual Analysis — Full-Cohort ML Models
# OOF (out-of-fold) 5-fold CV residuals  ·  Best model per metabolite
# 6-panel diagnostics:
#   (0,0) Residuals vs Fitted   (0,1) Q-Q            (0,2) Predicted vs Actual
#   (1,0) Residual Histogram    (1,1) Scale-Location  (1,2) Residuals by Group
# ============================================================
from scipy import stats as scipy_stats
from sklearn.base import clone
from sklearn.model_selection import KFold

if 'cv_df' not in dir() or cv_df is None or cv_df.empty:
    print('cv_df not found -- run regression CV cell first.')
else:
    resid_rows = []

    for ds in ACTIVE_DATASETS:
        ds_cv = cv_df[cv_df['Dataset'] == ds]
        if ds_cv.empty:
            continue

        best_per_met = ds_cv.loc[ds_cv.groupby('Target_Metabolite')['Mean_R2'].idxmax()]
        top3 = best_per_met.nlargest(3, 'Mean_R2')

        print(f"\n{'='*60}")
        print(f'Residual Analysis [{ds}]  --  OOF 5-fold CV')
        print(f"{'='*60}")

        for _, row in top3.iterrows():
            met_short  = row['Target_Metabolite']
            met_target = next(
                (m for m in target_metabolites.get(ds, []) if m[:50] == met_short), None
            )
            if met_target is None:
                continue

            # ── Data prep ───────────────────────────────────────────────────────
            X, y, feature_names, sample_ids = prepare_ml_data_regression_rev(
                transformed_species[ds], harmonized_meta[ds],
                transformed_mtb[ds], met_target, groups=None, ds_name=ds
            )
            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # ── Feature screen ─────────────────────────────────────────────────────
            rf_screen = RandomForestRegressor(
                n_estimators=100, max_depth=8, n_jobs=-1, random_state=42
            )
            rf_screen.fit(X_scaled, y)
            importances = pd.Series(rf_screen.feature_importances_, index=feature_names)
            top_feats   = importances.nlargest(
                min(MAX_FEATURES, len(feature_names))).index.tolist()
            sel_idx     = [feature_names.index(f) for f in top_feats]
            X_sel       = X_scaled[:, sel_idx]

            # ── Best model selection ───────────────────────────────────────────────
            best_model_name = (
                fc_best_model_map.get((ds, met_short))
                if 'fc_best_model_map' in dir() and fc_best_model_map
                else None
            ) or 'RandomForest'
            base_model = clone(get_regressors()[best_model_name])

            # ── OOF predictions via 5-fold CV ─────────────────────────────────────────
            kf    = KFold(n_splits=10, shuffle=True, random_state=42)
            y_oof = np.full(len(y), np.nan)
            for tr_idx, te_idx in kf.split(X_sel):
                m = clone(base_model)
                m.fit(X_sel[tr_idx], y[tr_idx])
                y_oof[te_idx] = m.predict(X_sel[te_idx])
            residuals = y - y_oof
            y_pred    = y_oof
            r2_oof    = float(1 - np.sum(residuals**2) / np.sum((y - y.mean())**2))

            sw_stat, sw_pval = scipy_stats.shapiro(residuals[:min(5000, len(residuals))])
            rmse             = float(np.sqrt(np.mean(residuals**2)))
            mean_resid       = float(residuals.mean())

            safe_name = met_target[:30].replace('/', '_').replace(';', '_')
            print(f"\n  [{ds[:15]}] {met_target[:50]}")
            print(f"    Model: {best_model_name}  |  OOF R\u00b2={r2_oof:.3f}  "
                  f"RMSE={rmse:.4f}  SW={'normal' if sw_pval > 0.05 else 'non-normal'}")

            # ── Study-group alignment ───────────────────────────────────────────────
            grp_map          = harmonized_meta[ds].set_index('Sample')['Study.Group']
            study_groups_arr = grp_map.reindex(sample_ids).values
            unique_grps      = sorted({g for g in study_groups_arr if pd.notna(g)})

            # ── 6-panel figure ───────────────────────────────────────────────────────
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(
                f'Residual Diagnostics '
                f'(OOF 5-fold CV \u00b7 {best_model_name} \u00b7 Species\u2192Metabolite)\n'
                f'[{ds[:25]}]: {met_short[:45]}',
                fontsize=12
            )

            bins_stat = np.linspace(y_pred.min(), y_pred.max(), 15)

            # Panel (0,0): Residuals vs Fitted
            ax = axes[0, 0]
            ax.scatter(y_pred, residuals, alpha=0.4, s=20, color='#2980b9')
            bin_means, bin_edges, _ = scipy_stats.binned_statistic(
                y_pred, residuals, statistic='mean', bins=bins_stat)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.plot(bin_centers, bin_means, color='red', lw=2, label='Binned mean')
            ax.axhline(0, color='black', linestyle='--', lw=1)
            ax.set_xlabel('OOF Fitted Values (log10 metabolite)')
            ax.set_ylabel('Residuals')
            ax.set_title('Residuals vs Fitted')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)

            # Panel (0,1): Q-Q plot
            ax = axes[0, 1]
            (osm, osr), (slope, intercept, _r) = scipy_stats.probplot(
                residuals, dist='norm')
            ax.scatter(osm, osr, alpha=0.5, s=20, color='#2980b9')
            line_x = np.array([osm.min(), osm.max()])
            ax.plot(line_x, slope * line_x + intercept, color='red', lw=2)
            ax.set_xlabel('Theoretical Quantiles')
            ax.set_ylabel('Sample Quantiles')
            ax.set_title(f'Q-Q Plot  (Shapiro-Wilk p={sw_pval:.3f})')
            ax.grid(alpha=0.25)

            # Panel (0,2): Predicted vs Actual
            ax = axes[0, 2]
            ax.scatter(y, y_pred, alpha=0.4, s=20, color='#27ae60')
            lo = min(y.min(), y_pred.min())
            hi = max(y.max(), y_pred.max())
            ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Ideal fit')
            ax.set_xlabel('Actual (log10 metabolite)')
            ax.set_ylabel('OOF Predicted')
            ax.set_title(f'Predicted vs Actual  (OOF R\u00b2={r2_oof:.3f})')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)

            # Panel (1,0): Residual histogram
            ax = axes[1, 0]
            ax.hist(residuals, bins=30, density=True, alpha=0.6,
                    color='#27ae60', edgecolor='white', label='Residuals')
            xr = np.linspace(residuals.min(), residuals.max(), 200)
            ax.plot(xr,
                    scipy_stats.norm.pdf(xr, residuals.mean(), residuals.std()),
                    'r-', lw=2, label='Normal fit')
            ax.set_xlabel('Residual')
            ax.set_ylabel('Density')
            ax.set_title('Residual Distribution')
            ax.legend(fontsize=8)

            # Panel (1,1): Scale-Location
            ax = axes[1, 1]
            sqrt_abs = np.sqrt(np.abs(residuals))
            ax.scatter(y_pred, sqrt_abs, alpha=0.4, s=20, color='#8e44ad')
            bin_means2, _, _ = scipy_stats.binned_statistic(
                y_pred, sqrt_abs, statistic='mean', bins=bins_stat)
            ax.plot(bin_centers, bin_means2, color='red', lw=2, label='Binned mean')
            ax.set_xlabel('OOF Fitted Values (log10 metabolite)')
            ax.set_ylabel('\u221a|Residuals|')
            ax.set_title('Scale-Location (Homoscedasticity)')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)

            # Panel (1,2): Residuals by Study Group
            ax = axes[1, 2]
            for xi, g in enumerate(unique_grps):
                mask = study_groups_arr == g
                grp_resid = residuals[mask]
                if len(grp_resid) > 1:
                    grp_color = (
                        GROUP_PALETTE.get(g, '#888888')
                        if 'GROUP_PALETTE' in dir() else '#888888'
                    )
                    parts = ax.violinplot(
                        grp_resid, positions=[xi], widths=0.6,
                        showmedians=True, showextrema=False
                    )
                    for pc in parts.get('bodies', []):
                        pc.set_facecolor(grp_color)
                        pc.set_alpha(0.6)
                    if 'cmedians' in parts:
                        parts['cmedians'].set_color('black')
                        parts['cmedians'].set_linewidth(2)
            ax.axhline(0, color='black', linestyle='--', lw=1)
            ax.set_xticks(range(len(unique_grps)))
            ax.set_xticklabels(
                [g.replace('_', ' ') for g in unique_grps],
                rotation=20, ha='right', fontsize=8)
            ax.set_xlabel('Study Group')
            ax.set_ylabel('OOF Residuals')
            ax.set_title('Residuals by Group')
            ax.grid(alpha=0.25)

            plt.tight_layout()
            fig_path = (CRC_RESULTS_DIR / 'figures' / 'ml' /
                        f'residuals_{ds[:15]}_{safe_name}.png')
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.show()
            plt.close()
            print(f'  Saved: {fig_path.name}')

            resid_rows.append({
                'Dataset':       ds,
                'Metabolite':    met_short,
                'Model':         best_model_name,
                'N_samples':     len(y),
                'OOF_R2':        round(r2_oof, 4),
                'RMSE':          round(rmse, 4),
                'Mean_Residual': round(mean_resid, 6),
                'SW_statistic':  round(float(sw_stat), 4),
                'SW_pvalue':     round(float(sw_pval), 4),
                'Normality':     'normal' if sw_pval > 0.05 else 'non-normal',
            })

    if resid_rows:
        resid_df = pd.DataFrame(resid_rows)
        print('\n' + '=' * 70)
        print('RESIDUAL DIAGNOSTICS SUMMARY  (Species -> Metabolite, OOF 5-fold CV)')
        print('=' * 70)
        display(resid_df)
        resid_df.to_csv(
            CRC_RESULTS_DIR / 'tables' / 'ml_residual_diagnostics.csv', index=False
        )
        print('\nSaved: tables/ml_residual_diagnostics.csv')
