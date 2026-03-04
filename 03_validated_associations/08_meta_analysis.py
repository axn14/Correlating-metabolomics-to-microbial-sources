

# --------------------------------------------------------------------------"""
Random-effects meta-analysis — DerSimonian-Laird pooling of Spearman correlations (Fisher z-transformed) across Erawijantari and Yachida. Forest plots; heterogeneity I² and τ²; BH-FDR correction.
"""


# ============================================================
# RANDOM-EFFECTS META-ANALYSIS: Yachida vs Erawijantari
# DerSimonian-Laird (DL) random-effects pooling of Spearman
# correlations for each species-polyamine pair.
# ============================================================

from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs(CRC_RESULTS_DIR / 'figures' / 'meta_analysis', exist_ok=True)

META_DATASETS = ['ERAWIJANTARI-GASTRIC-CANCER-2020', 'YACHIDA-CRC-2019']
META_LABELS   = {
    'ERAWIJANTARI-GASTRIC-CANCER-2020': 'Erawijantari',
    'YACHIDA-CRC-2019':                 'Yachida',
}


def _fisher_z(rho):
    rho = np.clip(rho, -0.9999, 0.9999)
    return 0.5 * np.log((1 + rho) / (1 - rho))


def _fisher_z_inv(z):
    return np.tanh(z)


def _dl_meta(zs, ses):
    """DerSimonian-Laird random-effects meta-analysis."""
    k    = len(zs)
    wi   = 1.0 / ses**2
    W    = wi.sum()
    z_fe = (wi * zs).sum() / W
    Q    = (wi * (zs - z_fe)**2).sum()
    Q_p  = scipy_stats.chi2.sf(Q, df=k - 1)
    c    = W - (wi**2).sum() / W
    tau2 = max(0.0, (Q - (k - 1)) / c)
    wi_re = 1.0 / (ses**2 + tau2)
    W_re  = wi_re.sum()
    z_re  = (wi_re * zs).sum() / W_re
    se_re = 1.0 / np.sqrt(W_re)
    I2    = max(0.0, (Q - (k - 1)) / Q * 100.0) if Q > 0 else 0.0
    return z_re, se_re, tau2, I2, Q, Q_p


if 'polyamine_corr' not in dir():
    print('polyamine_corr not found -- run correlation cell first.')
else:
    # Build lookup: (species, metabolite) -> {ds: {rho, N}}
    pair_data = {}
    for ds in META_DATASETS:
        if ds not in polyamine_corr or polyamine_corr[ds].empty:
            continue
        pc   = polyamine_corr[ds]
        N_ds = len(transformed_species[ds])
        for _, row in pc.iterrows():
            key = (row['Species'], row['Metabolite'])
            if key not in pair_data:
                pair_data[key] = {}
            pair_data[key][ds] = {'rho': row['Rho'], 'N': N_ds}

    both_pairs = {k: v for k, v in pair_data.items() if len(v) == 2}
    print(f'Species-polyamine pairs in BOTH cohorts: {len(both_pairs)}')

    if not both_pairs:
        print('No overlapping pairs found. '
              'Ensure polyamine_corr has Species, Metabolite, Rho columns '
              'and both datasets were processed.')
    else:
        # Run meta-analysis for every pair
        meta_rows = []
        for (species, metabolite), ds_vals in both_pairs.items():
            study_zs   = []
            study_ses  = []
            study_rhos = {}
            study_ns   = {}

            for ds in META_DATASETS:
                if ds not in ds_vals:
                    continue
                rho = ds_vals[ds]['rho']
                N   = ds_vals[ds]['N']
                z   = _fisher_z(rho)
                se  = 1.0 / max(np.sqrt(N - 3), 1e-9)
                study_zs.append(z)
                study_ses.append(se)
                study_rhos[ds] = rho
                study_ns[ds]   = N

            if len(study_zs) < 2:
                continue

            z_re, se_re, tau2, I2, Q, Q_p = _dl_meta(
                np.array(study_zs), np.array(study_ses))
            rho_re = _fisher_z_inv(z_re)
            ci_lo  = _fisher_z_inv(z_re - 1.96 * se_re)
            ci_hi  = _fisher_z_inv(z_re + 1.96 * se_re)
            z_score = z_re / se_re
            p_val   = 2 * (1 - scipy_stats.norm.cdf(abs(z_score)))

            meta_rows.append({
                'Species':          species,
                'Metabolite':       metabolite,
                'Rho_Erawijantari': round(study_rhos.get('ERAWIJANTARI-GASTRIC-CANCER-2020', float('nan')), 4),
                'N_Erawijantari':   study_ns.get('ERAWIJANTARI-GASTRIC-CANCER-2020', 0),
                'Rho_Yachida':      round(study_rhos.get('YACHIDA-CRC-2019', float('nan')), 4),
                'N_Yachida':        study_ns.get('YACHIDA-CRC-2019', 0),
                'Pooled_Rho_RE':    round(float(rho_re), 4),
                'CI_lower':         round(float(ci_lo),  4),
                'CI_upper':         round(float(ci_hi),  4),
                'SE_pooled':        round(float(se_re),  4),
                'I2_pct':           round(float(I2),     2),
                'tau2':             round(float(tau2),   6),
                'Q_statistic':      round(float(Q),      4),
                'Q_pvalue':         round(float(Q_p),    4),
                'Z_score_pooled':   round(float(z_score),4),
                'PValue_pooled':    round(float(p_val),  6),
            })

        meta_df = pd.DataFrame(meta_rows)

        if not meta_df.empty:
            _, qvals, _, _ = multipletests(meta_df['PValue_pooled'], method='fdr_bh')
            meta_df['QValue_pooled'] = qvals.round(6)
            meta_df.sort_values('QValue_pooled', inplace=True, ignore_index=True)

            n_sig     = (meta_df['QValue_pooled'] < 0.05).sum()
            n_high_i2 = (meta_df['I2_pct'] > 50).sum()

            print('\n' + '=' * 80)
            print('RANDOM-EFFECTS META-ANALYSIS RESULTS')
            print('=' * 80)
            print(f'  Pairs tested:                 {len(meta_df)}')
            print(f'  Significant (q<0.05):         {n_sig}')
            print(f'  High heterogeneity (I\u00b2>50%):  {n_high_i2}')
            display(meta_df.head(20))
            meta_df.to_csv(
                CRC_RESULTS_DIR / 'tables' / 'random_effects_meta_analysis.csv',
                index=False
            )
            print('\nSaved: tables/random_effects_meta_analysis.csv')

        # Forest plots: top species (by pooled |rho|)
        top_species_me = meta_df['Species'].unique()[:min(6, meta_df['Species'].nunique())]

        for spc in top_species_me:
            spc_data = meta_df[meta_df['Species'] == spc]
            if spc_data.empty:
                continue

            n_pairs = len(spc_data)
            n_cols  = min(n_pairs, 3)
            n_rows  = (n_pairs + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(5 * n_cols, 6 * n_rows),
                                     squeeze=False)
            axes_flat = axes.flatten()
            fig.suptitle(
                f'Forest Plot \u2014 Random-Effects Meta-Analysis\n'
                f'Species: {spc[:65]}',
                fontsize=11, y=1.02
            )

            for ax_idx, (_, mrow) in enumerate(spc_data.iterrows()):
                ax  = axes_flat[ax_idx]
                mtb = mrow['Metabolite']

                studies_p  = [d for d in META_DATASETS if d in pair_data.get((spc, mtb), {})]
                rhos_stu   = [pair_data[(spc, mtb)][d]['rho'] for d in studies_p]
                ns_stu     = [pair_data[(spc, mtb)][d]['N']   for d in studies_p]
                labels_stu = [META_LABELS[d] for d in studies_p]

                stu_zs  = [_fisher_z(r) for r in rhos_stu]
                stu_ses = [1.0 / max(np.sqrt(n - 3), 1e-9) for n in ns_stu]
                stu_cil = [_fisher_z_inv(z - 1.96 * se) for z, se in zip(stu_zs, stu_ses)]
                stu_cih = [_fisher_z_inv(z + 1.96 * se) for z, se in zip(stu_zs, stu_ses)]

                pooled_rho = mrow['Pooled_Rho_RE']
                ci_lo_p    = mrow['CI_lower']
                ci_hi_p    = mrow['CI_upper']
                I2_v       = mrow['I2_pct']
                Q_p_v      = mrow['Q_pvalue']
                q_pool     = mrow['QValue_pooled']

                y_pos = list(range(len(studies_p), 0, -1))

                for yp, lbl, rho, cil, cih, n in zip(
                        y_pos, labels_stu, rhos_stu, stu_cil, stu_cih, ns_stu):
                    sq = np.clip(n / 60, 3, 12)
                    ax.errorbar(rho, yp, xerr=[[rho - cil], [cih - rho]],
                                fmt='s', color='#2980b9', markersize=sq,
                                elinewidth=1.5, capsize=3, zorder=3)
                    ax.text(-1.08, yp, f'{lbl} (N={n})',
                            ha='right', va='center', fontsize=9)
                    ax.text(1.08, yp, f'{rho:.3f}',
                            ha='left', va='center', fontsize=9)

                d_y = 0; d_h = 0.28
                ax.fill([ci_lo_p, pooled_rho, ci_hi_p, pooled_rho],
                        [d_y, d_y + d_h, d_y, d_y - d_h],
                        color='#e74c3c', zorder=4, alpha=0.85)
                ax.text(1.08, d_y, f'{pooled_rho:.3f}',
                        ha='left', va='center', fontsize=9,
                        color='#c0392b', fontweight='bold')
                ax.text(-1.08, d_y,
                        f'Pooled RE\n(q={q_pool:.3f})',
                        ha='right', va='center', fontsize=8.5, color='#c0392b')

                ax.axvline(0, color='black', linestyle='--', lw=0.8, alpha=0.6)
                ax.set_xlim(-1.35, 1.35)
                ax.set_ylim(-0.8, len(studies_p) + 0.8)
                ax.set_xlabel('Spearman \u03c1', fontsize=9)
                ax.set_yticks([])
                ax.set_title(
                    f'vs {mtb[:32]}\nI\u00b2={I2_v:.0f}%,  Q p={Q_p_v:.3f}',
                    fontsize=9
                )
                ax.grid(axis='x', alpha=0.25)

            for ax_extra in axes_flat[n_pairs:]:
                ax_extra.set_visible(False)

            plt.tight_layout()
            safe_spc = (spc[:25]
                        .replace('/', '_').replace(';', '_').replace(' ', '_'))
            fig_path = (CRC_RESULTS_DIR / 'figures' / 'meta_analysis' /
                        f'forest_plot_{safe_spc}.png')
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.show()
            plt.close()
            print(f'Saved forest plot: {fig_path.name}')

        print('\nRandom-effects meta-analysis complete.')
        print(f'  Total pairs tested: {len(meta_df)}')
        print(f'  Significant (q<0.05): {n_sig}')
        print(f'  High heterogeneity (I\u00b2>50%): {n_high_i2}')
