

# --------------------------------------------------------------------------"""
Differential abundance — ERAWIJANTARI: Healthy vs Gastrectomy; YACHIDA: Healthy vs NonHealthy. Mann-Whitney U + BH-FDR; volcano plots with polyamines highlighted.
"""


# ============================================================
# DIFFERENTIAL ABUNDANCE — SPECIES & METABOLITES
# ERAWIJANTARI: Healthy vs Gastrectomy
# YACHIDA:      Healthy vs NonHealthy (Stage_0/I_II/III_IV pooled)
# ============================================================

CRC_DA_COMPARISONS = {
    'ERAWIJANTARI-GASTRIC-CANCER-2020': [('Healthy', 'Gastrectomy')],
    'YACHIDA-CRC-2019':                 [('Healthy', 'NonHealthy')],
}

# da_meta: binarised copy of harmonized_meta (Yachida stages → NonHealthy)
da_meta = {}
for ds in CRC_DA_COMPARISONS:
    mc = harmonized_meta[ds].copy()
    if ds == 'YACHIDA-CRC-2019':
        mc['Study.Group'] = mc['Study.Group'].apply(
            lambda g: g if g == 'Healthy' else 'NonHealthy'
        )
    da_meta[ds] = mc

# Run DA for species and metabolites
da_results = {}
for ds, comparisons in CRC_DA_COMPARISONS.items():
    da_results[ds] = {}
    for group1, group2 in comparisons:
        print(f'\n--- {ds}: {group1} vs {group2} ---')
        spc_da = differential_abundance(transformed_species[ds], da_meta[ds], group1, group2)
        mtb_da = differential_abundance(transformed_mtb[ds],     da_meta[ds], group1, group2)
        da_results[ds][(group1, group2)] = {'species': spc_da, 'metabolites': mtb_da}
        print(f'  Species:     {spc_da["Significant"].sum():3d} sig / {len(spc_da)} total')
        print(f'  Metabolites: {mtb_da["Significant"].sum():3d} sig / {len(mtb_da)} total')

print('\nDA complete.')

# ============================================================
# VOLCANO PLOTS — species (left) | metabolites (right)
# Polyamines highlighted in orange in the metabolite panel.
# ============================================================
import os
os.makedirs(CRC_RESULTS_DIR / 'figures' / 'da', exist_ok=True)

FC_THRESH = 0.5
Q_THRESH  = 0.1
TOP_N_ANN = 10

def _volcano_ax(ax, res_df, title, fc_thresh=1.0, q_thresh=0.05,
                highlight=None, top_n=10):
    """Draw a volcano plot on an existing Axes (embeddable in subplot grids)."""
    if res_df is None or res_df.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title(title, fontsize=10); return

    lfc  = res_df['Log2FC'].values
    logq = -np.log10(res_df['QValue'].clip(lower=1e-300).values)
    feat = res_df['Feature'].values

    sig_both = (np.abs(lfc) >= fc_thresh) & (res_df['QValue'].values < q_thresh)
    sig_fc   = (np.abs(lfc) >= fc_thresh) & ~sig_both
    sig_q    = (res_df['QValue'].values < q_thresh) & (np.abs(lfc) < fc_thresh)
    not_sig  = ~(sig_both | sig_fc | sig_q)

    ax.scatter(lfc[not_sig],  logq[not_sig],  c='#cccccc', s=15, alpha=0.6, linewidths=0, rasterized=True)
    ax.scatter(lfc[sig_fc],   logq[sig_fc],   c='#5B9BD5', s=20, alpha=0.75, linewidths=0)
    ax.scatter(lfc[sig_q],    logq[sig_q],    c='#70AD47', s=20, alpha=0.75, linewidths=0)
    ax.scatter(lfc[sig_both], logq[sig_both], c='#C00000', s=25, alpha=0.85, linewidths=0)

    if highlight:
        hl = np.isin(feat, highlight)
        ax.scatter(lfc[hl], logq[hl], c='#FF8C00', s=55, zorder=5,
                   linewidths=0.6, edgecolors='black')

    ax.axvline( fc_thresh, color='gray', lw=0.8, ls='--')
    ax.axvline(-fc_thresh, color='gray', lw=0.8, ls='--')
    ax.axhline(-np.log10(q_thresh), color='gray', lw=0.8, ls='--')

    ann_idx = np.where(sig_both)[0]
    if len(ann_idx):
        ann_idx = ann_idx[np.argsort(logq[ann_idx])[::-1][:top_n]]
        for i in ann_idx:
            ax.annotate(feat[i][:35], (lfc[i], logq[i]), fontsize=6,
                        xytext=(3, 3), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6, lw=0))

    ax.set_xlabel('Log\u2082 Fold Change', fontsize=9)
    ax.set_ylabel('\u2212log\u2081\u2080(q)', fontsize=9)
    ax.set_title(title, fontsize=10, pad=6)
    ax.grid(axis='both', alpha=0.2)
    ax.text(0.02, 0.98, f'{sig_both.sum()} significant\n(|FC|>{fc_thresh}, q<{q_thresh})',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', fc='white', alpha=0.7, lw=0.5))


from matplotlib.lines import Line2D

for ds, comparisons in CRC_DA_COMPARISONS.items():
    pa_feats = list(polyamine_columns.get(ds, {}).values())
    for (g1, g2), res in da_results[ds].items():
        fig, (ax_spc, ax_mtb) = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(f'{ds}  \u2014  {g1} vs {g2}  (Differential Abundance)',
                     fontsize=13, fontweight='bold', y=1.01)

        _volcano_ax(ax_spc, res['species'],     title='Species',
                    fc_thresh=FC_THRESH, q_thresh=Q_THRESH, top_n=TOP_N_ANN)
        _volcano_ax(ax_mtb, res['metabolites'], title='Metabolites',
                    fc_thresh=FC_THRESH, q_thresh=Q_THRESH,
                    highlight=pa_feats, top_n=TOP_N_ANN)

        legend_els = [
            Line2D([0],[0], marker='o', color='w', markerfacecolor='#C00000', ms=7, label='Sig. FC & q'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='#5B9BD5', ms=7, label='FC only'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='#70AD47', ms=7, label='q only'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='#cccccc', ms=7, label='Not sig.'),
        ]
        if pa_feats:
            legend_els.append(
                Line2D([0],[0], marker='o', color='w', markerfacecolor='#FF8C00',
                       ms=9, markeredgecolor='black', label='Polyamine')
            )
        fig.legend(handles=legend_els, loc='lower center', ncol=len(legend_els),
                   fontsize=9, bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

        plt.tight_layout()
        save_path = CRC_RESULTS_DIR / 'figures' / 'da' / f'{ds}_{g1}_vs_{g2}_volcano.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show(); plt.close()
        print(f'Saved: {save_path.name}')