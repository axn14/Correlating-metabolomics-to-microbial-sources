

# --------------------------------------------------------------------------"""
ML evaluation metrics summary — Aggregate all classification metrics across models and polyamine targets; grouped bar chart and CSV export.
"""


# ============================================================
# FULL-COHORT ML EVALUATION METRICS SUMMARY
# Comprehensive table + grouped bar chart for all classification
# metrics (AUC-ROC, AUPRC, F1, Accuracy, Recall, Precision)
# across the full cohort (disease + healthy samples).
# ============================================================

import os
os.makedirs(CRC_RESULTS_DIR / 'figures' / 'ml', exist_ok=True)

if 'fc_auc_df' not in dir() or fc_auc_df is None or fc_auc_df.empty:
    print('fc_auc_df not found -- run full-cohort ROC-AUC cell first.')
else:
    metric_cols   = ['Mean_AUC',  'Mean_AUPRC', 'Mean_F1',  'Mean_Accuracy',
                     'Mean_Recall', 'Mean_Precision']
    std_cols      = ['Std_AUC',   'Std_AUPRC',  'Std_F1',   'Std_Accuracy',
                     'Std_Recall', 'Std_Precision']
    metric_labels = ['AUC-ROC', 'AUPRC', 'F1', 'Accuracy', 'Recall', 'Precision']

    # -- 1. Summary table ----------------------------------------------------
    summary_rows = []
    for (ds, model), grp in fc_auc_df.groupby(['Dataset', 'Model']):
        row = {'Dataset': ds, 'Model': model, 'N_metabolites': len(grp)}
        for mc, sc, label in zip(metric_cols, std_cols, metric_labels):
            row[f'{label}_mean'] = round(grp[mc].mean(), 4)
            row[f'{label}_std']  = round(grp[sc].mean(), 4)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    for label in metric_labels:
        summary_df[f'{label} (mean±std)'] = (
            summary_df[f'{label}_mean'].round(3).astype(str) + ' ± ' +
            summary_df[f'{label}_std'].round(3).astype(str)
        )

    display_cols = ['Dataset', 'Model'] + [f'{l} (mean±std)' for l in metric_labels]
    print('=' * 80)
    print('FULL-COHORT ML EVALUATION METRICS SUMMARY')
    print('(Averaged across polyamine metabolite targets per dataset)')
    print('=' * 80)
    display(summary_df[display_cols].sort_values(['Dataset', 'Model']))
    summary_df.to_csv(
        CRC_RESULTS_DIR / 'tables' / 'ml_full_cohort_metrics_summary.csv', index=False
    )
    print('Saved: tables/ml_full_cohort_metrics_summary.csv')

    # -- 2. Grouped bar chart ------------------------------------------------
    datasets_fc  = fc_auc_df['Dataset'].unique()
    models_fc    = fc_auc_df['Model'].unique()
    n_ds_fc      = len(datasets_fc)
    bar_metrics  = ['Mean_AUC', 'Mean_AUPRC', 'Mean_F1', 'Mean_Accuracy']
    bar_stds     = ['Std_AUC',  'Std_AUPRC',  'Std_F1',  'Std_Accuracy']
    bar_labels_b = ['AUC-ROC', 'AUPRC', 'F1', 'Accuracy']
    palette_fc   = ['#2980b9', '#27ae60', '#e74c3c', '#8e44ad', '#f39c12']

    fig, axes = plt.subplots(1, n_ds_fc, figsize=(7 * n_ds_fc, 6), sharey=True)
    if n_ds_fc == 1:
        axes = [axes]

    x     = np.arange(len(bar_metrics))
    width = 0.8 / max(len(models_fc), 1)

    for ax, ds in zip(axes, datasets_fc):
        ds_data = fc_auc_df[fc_auc_df['Dataset'] == ds]
        for j, model in enumerate(models_fc):
            m_data = ds_data[ds_data['Model'] == model]
            if m_data.empty:
                continue
            means  = [m_data[mc].mean() for mc in bar_metrics]
            stds   = [m_data[sc].mean() for sc in bar_stds]
            offset = (j - len(models_fc) / 2 + 0.5) * width
            bars   = ax.bar(x + offset, means, width * 0.9, yerr=stds,
                            label=model, color=palette_fc[j % len(palette_fc)],
                            capsize=3, alpha=0.85, edgecolor='white')
            for bar, v in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=7)

        ax.axhline(0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels_b, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title(ds.replace('-', '\n'), fontsize=9)
        ax.legend(fontsize=7, loc='lower right', framealpha=0.7)
        ax.grid(axis='y', alpha=0.25)

    fig.suptitle(
        'Full-Cohort ML Classification Metrics (Disease + Healthy)\n'
        'Top metabolite targets per dataset (Species -> Metabolite direction), 5-fold StratifiedKFold OOF',
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    fig_path = CRC_RESULTS_DIR / 'figures' / 'ml' / 'full_cohort_ml_metrics_summary.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f'Saved: {fig_path.name}')
