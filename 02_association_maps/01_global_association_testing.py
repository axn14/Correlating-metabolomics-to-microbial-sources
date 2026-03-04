

# --------------------------------------------------------------------------"""
Global association testing — Mantel test (distance correlation), RDA (constrained ordination), and PLS (cross-validated R²) between microbiome and metabolome.
"""


# ============================================================
# GLOBAL ASSOCIATION TESTING — MANTEL TEST
# H0: no correlation between species and metabolite distance matrices
# Method: Pearson r between upper triangles of Euclidean distance
#         matrices; 999-permutation p-value.
# ============================================================
import os
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

os.makedirs(CRC_RESULTS_DIR / 'figures' / 'global_association', exist_ok=True)
os.makedirs(CRC_RESULTS_DIR / 'tables', exist_ok=True)

global_assoc_rows = []

print("=" * 70)
print("MANTEL TEST: Microbiome–Metabolome Distance Correlation")
print("=" * 70)

N_PERM_MANTEL = 999
RNG_MANTEL    = np.random.RandomState(42)

GROUP_PALETTE = {
    'Healthy':     '#4CAF50',
    'Gastrectomy': '#FF9800',
    'MP':          '#CDDC39',   # lime  — mucosal polyp (earliest pre-cancerous)
    'HS':          '#FFC107',   # amber — high-grade squamous lesion
    'Stage_0':     '#2196F3',
    'Stage_I_II':  '#9C27B0',
    'Stage_III_IV':'#F44336',
    'NonHealthy':  '#E91E63',
}
DEFAULT_COLOR = '#999999'

for ds in CRC_DATASETS:
    spc_df  = transformed_species[ds]
    mtb_df  = transformed_mtb[ds]
    meta_df = harmonized_meta[ds]

    # --- align samples ---
    shared = list(set(spc_df['Sample']) & set(mtb_df['Sample']))
    shared.sort()
    spc_m  = spc_df.set_index('Sample').loc[shared]
    mtb_m  = mtb_df.set_index('Sample').loc[shared]
    n      = len(shared)

    # --- distance matrices ---
    D_spc  = squareform(pdist(spc_m.values, metric='euclidean'))
    D_mtb  = squareform(pdist(mtb_m.values, metric='euclidean'))

    # upper triangle (no diagonal)
    idx    = np.triu_indices(n, k=1)
    v_spc  = D_spc[idx]
    v_mtb  = D_mtb[idx]

    # --- observed Mantel statistic ---
    r_obs, _ = pearsonr(v_mtb, v_spc)

    # --- permutation test ---
    r_perm = np.empty(N_PERM_MANTEL)
    for p in range(N_PERM_MANTEL):
        perm_idx   = RNG_MANTEL.permutation(n)
        D_spc_perm = D_spc[np.ix_(perm_idx, perm_idx)]
        r_perm[p], _ = pearsonr(v_mtb, D_spc_perm[idx])

    p_val = ((r_perm >= r_obs).sum() + 1) / (N_PERM_MANTEL + 1)
    sig   = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

    print(f"\n{ds}")
    print(f"  n={n}  Mantel r={r_obs:.4f}  p={p_val:.4f}  {sig}")

    global_assoc_rows.append({
        'Dataset': ds, 'Test': 'Mantel',
        'Statistic': round(r_obs, 4), 'P_value': round(p_val, 4),
        'Additional_info': f'n={n}, 999 perms',
    })

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{ds} — Mantel Test (r={r_obs:.3f}, p={p_val:.4f} {sig})',
                 fontsize=13, fontweight='bold')

    # scatter: distances coloured by study group pair
    ax = axes[0]
    meta_idx = meta_df.set_index('Sample').loc[shared, 'Study.Group']
    i_idx, j_idx = idx
    for pi, pj in zip(i_idx[:3000], j_idx[:3000]):   # subsample for readability
        gi, gj = meta_idx.iloc[pi], meta_idx.iloc[pj]
        lab = gi if gi == gj else f'{gi}/{gj}'
        c   = GROUP_PALETTE.get(gi, DEFAULT_COLOR)
        ax.scatter(v_mtb[pi * (n - 1) - pi*(pi-1)//2 + pj - pi - 1]
                   if False else D_mtb[i_idx, j_idx][0],
                   D_spc[i_idx, j_idx][0], c=c, alpha=0.2, s=4)
    # simpler: flat scatter without per-pair colouring
    ax.scatter(v_mtb, v_spc, alpha=0.15, s=4, color='steelblue', rasterized=True)
    m_, b_ = np.polyfit(v_mtb, v_spc, 1)
    x_ = np.linspace(v_mtb.min(), v_mtb.max(), 100)
    ax.plot(x_, m_*x_ + b_, 'r-', linewidth=1.5)
    ax.set_xlabel('Metabolite Euclidean Distance', fontsize=10)
    ax.set_ylabel('Species Euclidean Distance (CLR)', fontsize=10)
    ax.set_title(f'Distance Scatter  r={r_obs:.3f}', fontsize=10)
    ax.text(0.05, 0.92, f'p={p_val:.4f} {sig}', transform=ax.transAxes,
            fontsize=10, color='darkred', fontweight='bold')

    # permutation distribution
    ax2 = axes[1]
    ax2.hist(r_perm, bins=40, color='lightgray', edgecolor='gray', label='Permuted r')
    ax2.axvline(r_obs, color='red', linewidth=2, label=f'Observed r={r_obs:.3f}')
    ax2.set_xlabel('Mantel r', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Permutation Distribution', fontsize=10)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(CRC_RESULTS_DIR / 'figures' / 'global_association' / f'mantel_{ds}.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

print("\nMantel test complete.")


# ============================================================
# GLOBAL ASSOCIATION TESTING — RDA (Redundancy Analysis)
# Constrained ordination: how much of species variation is
# linearly explained by metabolite features?
# SVD-based hat-matrix handles rank-deficiency (p >> n).
# Permutation F-test (999 perms).
# ============================================================
from sklearn.preprocessing import StandardScaler as _SS

N_PERM_RDA = 999
RNG_RDA    = np.random.RandomState(42)

print("=" * 70)
print("RDA: Species ~ Metabolites")
print("=" * 70)

for ds in CRC_DATASETS:
    spc_df  = transformed_species[ds]
    mtb_df  = transformed_mtb[ds]
    meta_df = harmonized_meta[ds]

    # --- align samples ---
    shared = list(set(spc_df['Sample']) & set(mtb_df['Sample']))
    shared.sort()
    spc_m  = spc_df.set_index('Sample').loc[shared].values.astype(float)
    mtb_m  = mtb_df.set_index('Sample').loc[shared].values.astype(float)
    groups = meta_df.set_index('Sample').loc[shared, 'Study.Group'].values
    n      = len(shared)

    # --- standardise X, centre Y ---
    X = _SS().fit_transform(mtb_m)         # metabolites → predictor matrix
    Y = spc_m - spc_m.mean(axis=0)         # centre species (CLR already centred, but explicit)

    # --- SVD-based projection Y_hat = H @ Y ---
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    tol      = max(X.shape) * np.finfo(float).eps * s.max()
    keep     = s > tol
    Y_hat    = U[:, keep] @ (U[:, keep].T @ Y)

    # --- constrained variation ---
    SS_tot  = (Y ** 2).sum()
    SS_cons = (Y_hat ** 2).sum()
    R2_obs  = SS_cons / SS_tot

    # --- permutation F-test ---
    p_rda   = (Y.shape[1] - 1)   # number of response variables used for df
    df_con  = keep.sum()
    df_res  = n - df_con - 1
    if df_res <= 0:
        df_res = 1
    F_obs   = (SS_cons / df_con) / ((SS_tot - SS_cons) / df_res)

    F_perm = np.empty(N_PERM_RDA)
    for p in range(N_PERM_RDA):
        perm_idx   = RNG_RDA.permutation(n)
        Y_p        = Y[perm_idx]
        Y_hat_p    = U[:, keep] @ (U[:, keep].T @ Y_p)
        ss_c       = (Y_hat_p ** 2).sum()
        ss_r       = (Y_p ** 2).sum() - ss_c
        if ss_r <= 0:
            ss_r = 1e-10
        F_perm[p]  = (ss_c / df_con) / (ss_r / df_res)

    p_val = ((F_perm >= F_obs).sum() + 1) / (N_PERM_RDA + 1)
    sig   = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

    print(f"\n{ds}")
    print(f"  n={n}  constrained R2={R2_obs:.4f} ({R2_obs*100:.1f}%)  F={F_obs:.3f}  p={p_val:.4f}  {sig}")

    global_assoc_rows.append({
        'Dataset': ds, 'Test': 'RDA',
        'Statistic': round(R2_obs, 4), 'P_value': round(p_val, 4),
        'Additional_info': f'n={n}, constrained={R2_obs*100:.1f}%, F={F_obs:.2f}, 999 perms',
    })

    # --- RDA biplot (first 2 constrained axes) ---
    _, S_rda, Vt_rda = np.linalg.svd(Y_hat, full_matrices=False)
    U_rda = Y_hat @ Vt_rda[:2].T / (S_rda[:2] + 1e-12)  # sample scores (n × 2)

    # Metabolite arrows: X.T @ U_rda  (p_mtb × 2), normalise to unit max length
    arrows = (X.T @ U_rda[:, :2])
    arrow_len = np.linalg.norm(arrows, axis=1)
    top10_idx = np.argsort(arrow_len)[-10:]
    scale_arrow = 0.8 * np.abs(U_rda[:, :2]).max() / (arrow_len[top10_idx].max() + 1e-12)

    # variance explained per RDA axis
    var_ax = S_rda[:2] ** 2 / SS_tot * 100
    mtb_cols = [c for c in mtb_df.columns if c != 'Sample']

    unique_groups = sorted(set(groups))
    palette = {g: GROUP_PALETTE.get(g, DEFAULT_COLOR) for g in unique_groups}
    colors  = [palette[g] for g in groups]

    fig, ax = plt.subplots(figsize=(9, 7))
    for g in unique_groups:
        mask = groups == g
        ax.scatter(U_rda[mask, 0], U_rda[mask, 1],
                   c=palette[g], label=g, alpha=0.8, s=50, edgecolors='white', linewidths=0.5)

    for i in top10_idx:
        ax.annotate('', xy=(arrows[i, 0]*scale_arrow, arrows[i, 1]*scale_arrow),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5))
        ax.text(arrows[i, 0]*scale_arrow*1.08, arrows[i, 1]*scale_arrow*1.08,
                mtb_cols[i][:25], fontsize=7, color='darkorange', ha='center')

    ax.axhline(0, color='lightgray', lw=0.8, ls='--')
    ax.axvline(0, color='lightgray', lw=0.8, ls='--')
    ax.set_xlabel(f'RDA1 ({var_ax[0]:.1f}% constrained var)', fontsize=11)
    ax.set_ylabel(f'RDA2 ({var_ax[1]:.1f}% constrained var)', fontsize=11)
    ax.set_title(
        f'{ds}\nRDA: Species ~ Metabolites\n'
        f'R²={R2_obs:.3f} ({R2_obs*100:.1f}%), F={F_obs:.2f}, p={p_val:.4f} {sig}',
        fontsize=11, fontweight='bold'
    )
    ax.legend(title='Study Group', bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=9, title_fontsize=9)
    plt.tight_layout()
    fig.savefig(CRC_RESULTS_DIR / 'figures' / 'global_association' / f'rda_{ds}.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

print("\nRDA complete.")


# ============================================================
# GLOBAL ASSOCIATION TESTING — PLS (Partial Least Squares)
# Cross-decomposition of shared covariance between metabolomes
# and microbiomes.  PLSRegression (NIPALS).
# Cross-validated R² (10-fold) + 499-permutation significance.
# ============================================================
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler as _SS2

N_PERM_PLS = 499
N_COMP_MAX = 10
RNG_PLS    = np.random.RandomState(42)

print("=" * 70)
print("PLS: Metabolites → Species (cross-validated R²)")
print("=" * 70)

for ds in CRC_DATASETS:
    spc_df  = transformed_species[ds]
    mtb_df  = transformed_mtb[ds]
    meta_df = harmonized_meta[ds]

    # --- align ---
    shared  = list(set(spc_df['Sample']) & set(mtb_df['Sample']))
    shared.sort()
    X_raw   = mtb_df.set_index('Sample').loc[shared].values.astype(float)
    Y_raw   = spc_df.set_index('Sample').loc[shared].values.astype(float)
    groups  = meta_df.set_index('Sample').loc[shared, 'Study.Group'].values
    n       = len(shared)

    X_sc    = _SS2().fit_transform(X_raw)
    n_comp  = min(N_COMP_MAX, X_sc.shape[1], Y_raw.shape[1], n - 1)

    # --- fit full model ---
    pls     = PLSRegression(n_components=n_comp, scale=False, max_iter=500)
    pls.fit(X_sc, Y_raw)
    x_scores = pls.x_scores_           # n × n_comp
    y_scores = pls.y_scores_
    x_loads  = pls.x_loadings_          # p_mtb × n_comp
    y_loads  = pls.y_loadings_          # p_spc × n_comp

    x_var   = np.var(x_scores, axis=0, ddof=1)
    x_var_pct = x_var / np.var(X_sc, axis=0, ddof=1).sum() * 100
    y_var   = np.var(y_scores, axis=0, ddof=1)
    y_var_pct = y_var / np.var(Y_raw, axis=0, ddof=1).sum() * 100

    # --- cross-validated R² (5-fold, Y prediction from X) ---
    cv_folds = min(10, n)
    cv       = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    pls_cv   = PLSRegression(n_components=n_comp, scale=False, max_iter=500)

    # average R² across all Y columns (multivariate)
    from sklearn.metrics import r2_score
    r2_cv_per_fold = []
    for tr, te in cv.split(X_sc):
        pls_cv.fit(X_sc[tr], Y_raw[tr])
        Y_pred = pls_cv.predict(X_sc[te])
        r2_cv_per_fold.append(r2_score(Y_raw[te], Y_pred, multioutput='uniform_average'))
    r2_cv_obs = np.mean(r2_cv_per_fold)

    # --- permutation test ---
    r2_perm = np.empty(N_PERM_PLS)
    for p in range(N_PERM_PLS):
        perm_idx = RNG_PLS.permutation(n)
        Y_p      = Y_raw[perm_idx]
        r2_pf    = []
        for tr, te in cv.split(X_sc):
            pls_cv.fit(X_sc[tr], Y_p[tr])
            Y_pred_p = pls_cv.predict(X_sc[te])
            r2_pf.append(r2_score(Y_p[te], Y_pred_p, multioutput='uniform_average'))
        r2_perm[p] = np.mean(r2_pf)

    p_val = ((r2_perm >= r2_cv_obs).sum() + 1) / (N_PERM_PLS + 1)
    sig   = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

    print(f"\n{ds}")
    print(f"  n={n}  n_comp={n_comp}  CV R²={r2_cv_obs:.4f}  p={p_val:.4f}  {sig}")
    for c in range(n_comp):
        print(f"  LV{c+1}: X_var={x_var_pct[c]:.1f}%  Y_var={y_var_pct[c]:.1f}%")

    global_assoc_rows.append({
        'Dataset': ds, 'Test': 'PLS',
        'Statistic': round(r2_cv_obs, 4), 'P_value': round(p_val, 4),
        'Additional_info': f'n={n}, n_comp={n_comp}, CV-R2={r2_cv_obs:.4f}, 499 perms',
    })

    # --- 3-panel figure ---
    mtb_cols = [c for c in mtb_df.columns if c != 'Sample']
    spc_cols = [c for c in spc_df.columns if c != 'Sample']
    unique_groups = sorted(set(groups))
    palette  = {g: GROUP_PALETTE.get(g, DEFAULT_COLOR) for g in unique_groups}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f'{ds} — PLS  (CV R²={r2_cv_obs:.3f}, p={p_val:.4f} {sig},  n_comp={n_comp})',
        fontsize=13, fontweight='bold'
    )

    # Panel 1: variance explained per LV
    ax1 = axes[0]
    x_pos = np.arange(n_comp)
    width = 0.35
    ax1.bar(x_pos - width/2, x_var_pct, width, label='Metabolites (X)', color='#4C72B0', alpha=0.85)
    ax1.bar(x_pos + width/2, y_var_pct, width, label='Species (Y)', color='#DD8452', alpha=0.85)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'LV{c+1}' for c in range(n_comp)])
    ax1.set_ylabel('Variance Explained (%)', fontsize=10)
    ax1.set_title('Variance per Latent Variable', fontsize=10)
    ax1.legend(fontsize=9)

    # Panel 2: score plot LV1 vs LV2
    ax2 = axes[1]
    for g in unique_groups:
        mask = groups == g
        ax2.scatter(x_scores[mask, 0], x_scores[mask, 1] if n_comp > 1 else np.zeros(mask.sum()),
                    c=palette[g], label=g, alpha=0.8, s=50, edgecolors='white', linewidths=0.5)
    ax2.axhline(0, color='lightgray', lw=0.8, ls='--')
    ax2.axvline(0, color='lightgray', lw=0.8, ls='--')
    ax2.set_xlabel(f'LV1 scores ({x_var_pct[0]:.1f}% X-var)', fontsize=10)
    y_label2 = f'LV2 scores ({x_var_pct[1]:.1f}% X-var)' if n_comp > 1 else 'LV2 (N/A)'
    ax2.set_ylabel(y_label2, fontsize=10)
    ax2.set_title('Score Plot (X scores, LV1 vs LV2)', fontsize=10)
    ax2.legend(title='Study Group', fontsize=8, title_fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')

    # Panel 3: top 15 loadings on LV1 (species + metabolites)
    ax3 = axes[2]
    spc_load1 = y_loads[:, 0]
    mtb_load1 = x_loads[:, 0]
    all_labels  = [f'[S] {s[:30]}' for s in spc_cols] + [f'[M] {m[:30]}' for m in mtb_cols]
    all_loads   = np.concatenate([spc_load1, mtb_load1])
    top15_idx   = np.argsort(np.abs(all_loads))[-15:]
    top_loads   = all_loads[top15_idx]
    top_labels  = [all_labels[i] for i in top15_idx]
    colors_bar  = ['#DD8452' if l.startswith('[S]') else '#4C72B0' for l in top_labels]

    ax3.barh(range(15), top_loads, color=colors_bar, alpha=0.85)
    ax3.set_yticks(range(15))
    ax3.set_yticklabels(top_labels, fontsize=7)
    ax3.axvline(0, color='black', lw=0.8)
    ax3.set_xlabel('Loading on LV1', fontsize=10)
    ax3.set_title('Top 15 Loadings (LV1)', fontsize=10)
    from matplotlib.patches import Patch
    ax3.legend(handles=[Patch(color='#DD8452', label='Species'),
                         Patch(color='#4C72B0', label='Metabolites')],
               fontsize=8, loc='lower right')

    plt.tight_layout()
    fig.savefig(CRC_RESULTS_DIR / 'figures' / 'global_association' / f'pls_{ds}.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

# --- Summary table ---
global_assoc_df = pd.DataFrame(global_assoc_rows)
print("\n" + "=" * 70)
print("GLOBAL ASSOCIATION SUMMARY")
print("=" * 70)
display(global_assoc_df)
global_assoc_df.to_csv(CRC_RESULTS_DIR / 'tables' / 'global_association_summary.csv', index=False)
print("\nResults saved to tables/global_association_summary.csv")
