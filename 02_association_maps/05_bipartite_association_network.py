

# --------------------------------------------------------------------------"""
Bipartite species–metabolite association network — OLS linear models per study group; bezier arc plots coloured by group (Healthy=blue, Gastrectomy=yellow, HS=grey, other=red).
"""


# ============================================================
# BIPARTITE SPECIES–METABOLITE ASSOCIATION NETWORK
# OLS linear model per study group, per dataset
# Top 20 species × top 50 metabolites (by significant association count)
# Lines coloured by study group: Healthy=blue, Gastrectomy=yellow,
#   HS=grey, all other stages=red
# ============================================================
from scipy.stats import linregress
from statsmodels.stats.multitest import multipletests
import matplotlib.path as mpath
import matplotlib.patches as mpatches

_BIPARTITE_PALETTE = {
    'Healthy':     '#1565C0',   # blue
    'Gastrectomy': '#FDD835',   # yellow
    'HS':          '#9E9E9E',   # grey
}
_BIPARTITE_DEFAULT = '#C62828'  # red — Stage_0, Stage_I_II, Stage_III_IV, MP

def _bp_color(group):
    return _BIPARTITE_PALETTE.get(group, _BIPARTITE_DEFAULT)


def compute_lm_by_group(spc_df, mtb_df, meta_df, min_samples=10):
    """OLS regression (metabolite ~ species) per study group.

    Returns
    -------
    dict : {group_name: DataFrame(Species, Metabolite, Beta, PValue, QValue, R2)}
    """
    spc_cols = [c for c in spc_df.columns if c != 'Sample']
    mtb_cols = [c for c in mtb_df.columns if c != 'Sample']
    results = {}

    for group in sorted(meta_df['Study.Group'].unique()):
        samples = meta_df.loc[meta_df['Study.Group'] == group, 'Sample'].values
        if len(samples) < min_samples:
            print(f"  [{group}] skip — only {len(samples)} samples (< {min_samples})")
            continue

        merged = (
            spc_df[spc_df['Sample'].isin(samples)]
            .merge(mtb_df[mtb_df['Sample'].isin(samples)],
                   on='Sample', suffixes=('_spc', '_mtb'))
        )

        rows = []
        for spc in spc_cols:
            sk = spc + '_spc' if spc + '_spc' in merged.columns else spc
            x = merged[sk].values
            if x.std() == 0:
                continue
            for mtb in mtb_cols:
                mk = mtb + '_mtb' if mtb + '_mtb' in merged.columns else mtb
                y = merged[mk].values
                if y.std() == 0:
                    continue
                slope, _, r, p, _ = linregress(x, y)
                rows.append({'Species': spc, 'Metabolite': mtb,
                             'Beta': slope, 'PValue': p, 'R2': r ** 2})

        if not rows:
            results[group] = pd.DataFrame()
            continue

        df = pd.DataFrame(rows)
        _, qvals, _, _ = multipletests(df['PValue'], method='fdr_bh')
        df['QValue'] = qvals
        results[group] = df
        n_sig = int((df['QValue'] < 0.05).sum())
        print(f"  [{group}] n={len(samples)},  {n_sig}/{len(df)} pairs significant (q<0.05)")

    return results


def plot_bipartite_lm(lm_by_group, top_species=20, top_mtb=50,
                      title='', save_path=None):
    """Bipartite arc plot: species (left) ↔ metabolites (right).

    Draws bezier curves for each significant association (q<0.05).
    Line colour = study group; alpha ∝ −log10(q); width ∝ |Beta|.
    """
    sig_frames = []
    for group, df in lm_by_group.items():
        if df is None or df.empty:
            continue
        sig = df[df['QValue'] < 0.05].copy()
        sig['Group'] = group
        sig_frames.append(sig)

    if not sig_frames:
        print("  No significant associations (q<0.05) — plot skipped.")
        return None, None

    all_sig = pd.concat(sig_frames, ignore_index=True)

    # Select top nodes by frequency of significant appearances
    top_spc_list = (all_sig['Species']
                    .value_counts().head(top_species).index.tolist())
    top_mtb_list = (all_sig['Metabolite']
                    .value_counts().head(top_mtb).index.tolist())

    plot_df = all_sig[
        all_sig['Species'].isin(top_spc_list) &
        all_sig['Metabolite'].isin(top_mtb_list)
    ]

    if plot_df.empty:
        print("  No edges after filtering to top species/metabolites — plot skipped.")
        return None, None

    # Y-axis positions (top-to-bottom order)
    n_spc = len(top_spc_list)
    n_mtb = len(top_mtb_list)
    spc_ys = {s: i for i, s in enumerate(reversed(top_spc_list))}

    if n_mtb > 1:
        mtb_ys = {m: i * (n_spc - 1) / (n_mtb - 1)
                  for i, m in enumerate(reversed(top_mtb_list))}
    else:
        mtb_ys = {top_mtb_list[0]: (n_spc - 1) / 2.0}

    fig_h = max(14, max(n_spc, n_mtb) * 0.32)
    fig, ax = plt.subplots(figsize=(16, fig_h))

    # Draw bezier arcs
    for _, row in plot_df.iterrows():
        y0  = spc_ys[row['Species']]
        y1  = mtb_ys[row['Metabolite']]
        col = _bp_color(row['Group'])
        q   = max(row['QValue'], 1e-10)
        alpha = min(0.80, 0.10 + 0.70 * min(1.0, -np.log10(q) / 6.0))
        lw    = 0.4 + 1.4 * min(1.0, abs(row['Beta']) / 2.0)

        verts = [(0.0, y0), (0.5, y0), (0.5, y1), (1.0, y1)]
        codes = [mpath.Path.MOVETO,
                 mpath.Path.CURVE4,
                 mpath.Path.CURVE4,
                 mpath.Path.CURVE4]
        patch = mpatches.PathPatch(
            mpath.Path(verts, codes),
            facecolor='none', edgecolor=col,
            lw=lw, alpha=alpha, zorder=2)
        ax.add_patch(patch)

    # Species nodes (left column)
    for spc, y in spc_ys.items():
        ax.scatter(0, y, s=70, color='#1976D2', zorder=5)
        ax.text(-0.03, y, spc, ha='right', va='center', fontsize=7)

    # Metabolite nodes (right column)
    for mtb, y in mtb_ys.items():
        ax.scatter(1, y, s=70, color='#388E3C', zorder=5)
        ax.text(1.03, y, mtb, ha='left', va='center', fontsize=7)

    # Column labels
    y_top = max(n_spc, n_mtb) - 0.5
    ax.text(0, y_top + 0.6, 'Species\n(top 20)', ha='center',
            fontsize=10, fontweight='bold', color='#1976D2')
    ax.text(1, y_top + 0.6, 'Metabolites\n(top 50)', ha='center',
            fontsize=10, fontweight='bold', color='#388E3C')

    # Legend — one patch per group present in lm_by_group
    groups_present = [g for g in sorted(lm_by_group.keys()) if g in lm_by_group]
    handles = [mpatches.Patch(color=_bp_color(g), label=g) for g in groups_present]
    ax.legend(handles=handles, loc='upper center',
              bbox_to_anchor=(0.5, -0.01),
              ncol=min(6, len(groups_present)), fontsize=9,
              title='Study group (line colour)')

    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.5, max(n_spc, n_mtb) + 1.2)
    ax.axis('off')
    ax.set_title(title, fontsize=12, pad=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


# ── Main loop ─────────────────────────────────────────────────────────────────
lm_assoc = {}
(CRC_RESULTS_DIR / 'figures' / 'network').mkdir(parents=True, exist_ok=True)

for ds in CRC_DATASETS:
    print(f"\n{'='*60}\n{ds}")
    lm_assoc[ds] = compute_lm_by_group(
        transformed_species[ds], transformed_mtb[ds], harmonized_meta[ds]
    )

    fig, ax = plot_bipartite_lm(
        lm_assoc[ds],
        top_species=20,
        top_mtb=50,
        title=(f'{ds}\n'
               f'Species–Metabolite Associations  (OLS per group, q<0.05)\n'
               f'Top 20 species  ×  Top 50 metabolites'),
        save_path=(CRC_RESULTS_DIR / 'figures' / 'network'
                   / f'{ds}_bipartite_lm.png')
    )
    if fig:
        plt.show()
        plt.close()
        print(f"  Saved: figures/network/{ds}_bipartite_lm.png")
