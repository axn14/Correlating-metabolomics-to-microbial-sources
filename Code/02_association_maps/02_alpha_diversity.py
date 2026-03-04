# --------------------------------------------------------------------------
"""
Alpha diversity — Shannon H' for species and metabolites per disease stage. Mann-Whitney U (2 groups) or Kruskal-Wallis + BH-FDR pairwise (≥3 groups).
"""


# ============================================================
# ALPHA DIVERSITY — SHANNON INDEX BY DISEASE STAGE
# Species AND Metabolites shown side-by-side per dataset
# ============================================================
from scipy.stats import entropy, mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import seaborn as sns

shannon_diversity_results  = {}
shannon_diversity_mtb      = {}
shannon_stats              = {}
shannon_stats_mtb          = {}

# ── Helper: compute Shannon H' per sample row ────────────────────────────────
def _shannon_from_df(df):
    feat_cols = [c for c in df.columns if c != 'Sample']
    return df[feat_cols].apply(
        lambda row: float(entropy(row[row > 0])) if (row > 0).any() else 0.0,
        axis=1
    ).values

def _rank_biserial(u, n1, n2):
    return 1 - (2 * u) / (n1 * n2)

def _eta_squared_kw(h, k, n):
    return (h - k + 1) / (n - k) if n > k else 0.0

def _sig_label(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

# ── Helper: run KW / MWU statistics ─────────────────────────────────────────
def _run_stats(h_df, groups, ds_label):
    k       = len(groups)
    n       = len(h_df)
    group_h = {g: h_df.loc[h_df['Study.Group'] == g, 'Shannon_H'].values
               for g in groups}
    stat_rows = []

    if k < 2:
        print(f"  [{ds_label}] Skipping statistics (only 1 group: "
              f"{groups[0] if groups else 'none'})")
        return stat_rows, 1.0

    if k == 2:
        g1, g2      = groups
        u_stat, p_v = mannwhitneyu(group_h[g1], group_h[g2], alternative='two-sided')
        r_rb        = _rank_biserial(u_stat, len(group_h[g1]), len(group_h[g2]))
        print(f"  [{ds_label}] Mann-Whitney U={u_stat:.1f}, p={p_v:.4f} "
              f"{_sig_label(p_v)}, r={r_rb:.3f}")
        stat_rows.append({
            'Test': 'Mann-Whitney U',
            'Groups': f'{g1} vs {g2}', 'Statistic': round(u_stat, 3),
            'P_value': round(p_v, 4), 'Q_value': round(p_v, 4),
            'Effect_size': round(r_rb, 3), 'Effect_metric': 'rank-biserial r',
        })
        return stat_rows, p_v

    # k >= 3: Kruskal-Wallis + pairwise BH-FDR
    h_stat, kw_p = kruskal(*[group_h[g] for g in groups])
    eta2         = _eta_squared_kw(h_stat, k, n)
    print(f"  [{ds_label}] Kruskal-Wallis H={h_stat:.3f}, p={kw_p:.4f} "
          f"{_sig_label(kw_p)}, eta2={eta2:.3f}")
    stat_rows.append({
        'Test': 'Kruskal-Wallis',
        'Groups': ' vs '.join(groups), 'Statistic': round(h_stat, 3),
        'P_value': round(kw_p, 4), 'Q_value': round(kw_p, 4),
        'Effect_size': round(eta2, 3), 'Effect_metric': 'eta_squared',
    })
    pairs    = list(combinations(groups, 2))
    pw_stats_list = []
    pw_pvals_list = []
    for g1, g2 in pairs:
        u_s, p_s = mannwhitneyu(group_h[g1], group_h[g2], alternative='two-sided')
        pw_stats_list.append(u_s)
        pw_pvals_list.append(p_s)
    _, qvals, _, _ = multipletests(pw_pvals_list, method='fdr_bh')
    print(f"  [{ds_label}] Pairwise post-hoc (BH-FDR):")
    for (g1, g2), u_s, p_s, q_s in zip(pairs, pw_stats_list, pw_pvals_list, qvals):
        r_rb  = _rank_biserial(u_s, len(group_h[g1]), len(group_h[g2]))
        print(f"    {g1:18s} vs {g2:18s}  U={u_s:.1f}  p={p_s:.4f}  "
              f"q={float(q_s):.4f} {_sig_label(float(q_s))}  r={r_rb:.3f}")
        stat_rows.append({
            'Test': 'Mann-Whitney U (post-hoc)',
            'Groups': f'{g1} vs {g2}', 'Statistic': round(u_s, 3),
            'P_value': round(p_s, 4), 'Q_value': round(float(q_s), 4),
            'Effect_size': round(r_rb, 3), 'Effect_metric': 'rank-biserial r',
        })
    return stat_rows, kw_p

# ── Helper: draw one box+strip panel ────────────────────────────────────────
def _draw_shannon_panel(ax, h_df, groups, stat_rows, kw_p, title, ylabel, palette):
    if h_df['Shannon_H'].isna().all() or len(groups) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='grey')
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_ylabel(ylabel, fontsize=11)
        return

    pal_dict = {g: palette.get(g, '#888888') for g in groups}
    group_h  = {g: h_df.loc[h_df['Study.Group'] == g, 'Shannon_H'].values
                for g in groups}

    sns.boxplot(data=h_df, x='Study.Group', y='Shannon_H', order=groups,
                palette=pal_dict, width=0.45, linewidth=1.2,
                showfliers=False, ax=ax)
    sns.stripplot(data=h_df, x='Study.Group', y='Shannon_H', order=groups,
                  palette=pal_dict, size=4, alpha=0.55, jitter=True, ax=ax)

    all_vals = np.concatenate(list(group_h.values()))
    y_range  = all_vals.max() - all_vals.min() if len(all_vals) > 1 else 1.0

    for xi, g in enumerate(groups):
        med = float(np.median(group_h[g]))
        ax.text(xi, med + y_range * 0.01, f'{med:.2f}',
                ha='center', va='bottom', fontsize=8,
                color='#333333', fontweight='bold')

    sig_pairs = [r for r in stat_rows
                 if r['Test'] != 'Kruskal-Wallis' and r['Q_value'] < 0.05]
    y_max   = all_vals.max()
    bar_h   = y_max + y_range * 0.06
    bar_gap = y_range * 0.09
    for rank_i, row in enumerate(sig_pairs):
        g1_str, g2_str = [s.strip() for s in row['Groups'].split(' vs ')]
        if g1_str not in groups or g2_str not in groups:
            continue
        xi, xj = groups.index(g1_str), groups.index(g2_str)
        y_bar  = bar_h + rank_i * bar_gap
        ax.plot([xi, xi, xj, xj],
                [y_bar, y_bar + bar_gap * 0.3, y_bar + bar_gap * 0.3, y_bar],
                lw=1.2, color='black')
        ax.text((xi + xj) / 2, y_bar + bar_gap * 0.35,
                _sig_label(row['Q_value']),
                ha='center', va='bottom', fontsize=11)

    ax.set_title(title, fontsize=11, pad=8)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel('')
    ax.set_xticklabels(
        [g.replace('_', ' ') for g in groups], fontsize=9, rotation=15, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── Main loop ────────────────────────────────────────────────────────────────
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.3)
palette = GROUP_PALETTE if 'GROUP_PALETTE' in dir() else {}

(CRC_RESULTS_DIR / 'figures' / 'alpha_diversity').mkdir(parents=True, exist_ok=True)
(CRC_RESULTS_DIR / 'tables').mkdir(parents=True, exist_ok=True)

for ds in CRC_DATASETS:
    print(f"\n{'='*60}\n{ds}")

    meta_sub = harmonized_meta[ds][['Sample', 'Study.Group']].copy()

    # ── 1a. Species Shannon H ─────────────────────────────────────────────────
    if ds in sample_qc_species and 'shannon_diversity' in sample_qc_species[ds].columns:
        h_df = sample_qc_species[ds][['Sample', 'shannon_diversity']].copy()
        h_df = h_df.rename(columns={'shannon_diversity': 'Shannon_H'})
        print(f"  Species: reusing sample_qc_species (n={len(h_df)})")
    else:
        h_vals = _shannon_from_df(species_reduced[ds])
        h_df   = pd.DataFrame({'Sample': species_reduced[ds]['Sample'].values,
                               'Shannon_H': h_vals})
        print(f"  Species: recomputed from species_reduced (n={len(h_df)})")

    h_df = h_df.merge(meta_sub, on='Sample', how='inner').dropna(
        subset=['Shannon_H', 'Study.Group'])
    shannon_diversity_results[ds] = h_df

    groups = sorted(h_df['Study.Group'].unique())
    k      = len(groups)
    print(f"  Groups: {groups}  (n={len(h_df)})")

    spc_stat_rows, spc_kw_p = _run_stats(h_df, groups, 'species')
    shannon_stats[ds] = [{'Dataset': ds, **r} for r in spc_stat_rows]

    # ── 1b. Metabolite Shannon H ──────────────────────────────────────────────
    if 'transformed_mtb' in dir() and ds in transformed_mtb:
        m_vals = _shannon_from_df(transformed_mtb[ds])
        m_df   = pd.DataFrame({'Sample': transformed_mtb[ds]['Sample'].values,
                               'Shannon_H': m_vals})
        m_df   = m_df.merge(meta_sub, on='Sample', how='inner').dropna(
            subset=['Shannon_H', 'Study.Group'])
        print(f"  Metabolites: computed from transformed_mtb (n={len(m_df)})")
    else:
        m_df = h_df[['Sample', 'Study.Group']].copy()
        m_df['Shannon_H'] = np.nan
        print(f"  WARNING: transformed_mtb['{ds}'] not found — metabolite panel empty")

    m_groups = (sorted(m_df['Study.Group'].unique())
                if not m_df['Shannon_H'].isna().all() else groups)
    mtb_stat_rows, mtb_kw_p = _run_stats(m_df, m_groups, 'metabolites')
    shannon_stats_mtb[ds] = [{'Dataset': ds, **r} for r in mtb_stat_rows]
    shannon_diversity_mtb[ds] = m_df

    # ── 2. Figure: 1 row x 2 cols ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(max(14, k * 4.0), 6))
    fig.suptitle(f"Shannon Diversity — {ds}", fontsize=13, y=1.01)

    if k < 2:
        spc_title = f"Species\nShannon H'  (single group — stats N/A)"
    else:
        spc_title = (f"Species\n"
                     f"Shannon H'  (p={spc_kw_p:.4f} {_sig_label(spc_kw_p)})")

    mk = len(m_groups)
    if mk < 2 or m_df['Shannon_H'].isna().all():
        mtb_title = "Metabolites (log2)\nShannon H'  (single group — stats N/A)"
    else:
        mtb_title = (f"Metabolites (log2)\n"
                     f"Shannon H'  (p={mtb_kw_p:.4f} {_sig_label(mtb_kw_p)})")

    _draw_shannon_panel(axes[0], h_df,  groups,   spc_stat_rows,  spc_kw_p,
                        spc_title, "Shannon H' index", palette)
    _draw_shannon_panel(axes[1], m_df,  m_groups, mtb_stat_rows,  mtb_kw_p,
                        mtb_title, "Shannon H' index (log2 metabolites)", palette)

    plt.tight_layout()
    fig_path = (CRC_RESULTS_DIR / 'figures' / 'alpha_diversity'
                / f'{ds}_shannon_species_vs_metabolites.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Saved: figures/alpha_diversity/{ds}_shannon_species_vs_metabolites.png")

    # ── 3. Save CSVs ──────────────────────────────────────────────────────────
    if shannon_stats[ds]:
        pd.DataFrame(shannon_stats[ds]).to_csv(
            CRC_RESULTS_DIR / 'tables' / f'{ds}_shannon_diversity_species.csv',
            index=False)
    if shannon_stats_mtb[ds]:
        pd.DataFrame(shannon_stats_mtb[ds]).to_csv(
            CRC_RESULTS_DIR / 'tables' / f'{ds}_shannon_diversity_metabolites.csv',
            index=False)

sns.set_style('white')
sns.set_context('notebook')

# ── Combined species summary (backward-compatible) ────────────────────────────
all_stats = pd.concat(
    [pd.DataFrame(shannon_stats[d]) for d in CRC_DATASETS if shannon_stats.get(d)],
    ignore_index=True
)
if not all_stats.empty:
    all_stats.to_csv(
        CRC_RESULTS_DIR / 'tables' / 'shannon_diversity_all_datasets.csv', index=False)
    print("\nSaved: tables/shannon_diversity_all_datasets.csv")
