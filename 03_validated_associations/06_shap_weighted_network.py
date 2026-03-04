

# --------------------------------------------------------------------------"""
SHAP-weighted bipartite network — Nodes: metabolite targets (purple), species (green), group dummies (orange), confounders (grey). Edge weight = SHAP importance × best AUC (composite importance score).
"""


# ============================================================
# FULL-COHORT WEIGHTED NETWORK ANALYSIS
# Nodes: metabolite targets (purple) + species features (green)
#        + Group features (orange) + confounders (gray)
# Edges: SHAP mean |SHAP| importance
# Composite weight: SHAP × best_AUC for that metabolite target
# ============================================================

import os as _os
_os.makedirs(CRC_RESULTS_DIR / 'figures' / 'network', exist_ok=True)


def build_fc_network(fc_shap_results, fc_auc_df, ds):
    """Build weighted bipartite graph from full-cohort SHAP + AUC results.
    fc_shap_results is keyed by (ds, metabolite_target).
    """
    G = nx.Graph()

    # Best AUC per metabolite target (max across models)
    if fc_auc_df is not None and not fc_auc_df.empty:
        ds_auc = fc_auc_df[fc_auc_df['Dataset'] == ds]
        best_auc = (ds_auc.groupby('Target_Metabolite')['Mean_AUC']
                          .max().to_dict())
    else:
        best_auc = {}

    edge_rows = []
    for (d, met), sr in fc_shap_results.items():
        if d != ds:
            continue
        met_short = met[:50]
        auc = best_auc.get(met_short, 0.5)   # default to random if no AUC

        # Add metabolite-target node
        G.add_node(met_short,
                   node_type='metabolite_target',
                   best_auc=auc,
                   size=auc)

        for feat_name, shap_imp in sr['top_features']:
            ftype = ('group'      if feat_name.startswith('Group_') else
                     'confounder' if feat_name.startswith('Conf_')  else
                     'species')
            composite = shap_imp * auc

            # Add feature node
            if feat_name not in G:
                G.add_node(feat_name, node_type=ftype, mean_shap=shap_imp)
            else:
                G.nodes[feat_name]['mean_shap'] = max(
                    G.nodes[feat_name].get('mean_shap', 0), shap_imp)

            # Add / update edge
            if G.has_edge(met_short, feat_name):
                prev = G[met_short][feat_name]
                G[met_short][feat_name]['shap_imp'] = max(prev['shap_imp'], shap_imp)
                G[met_short][feat_name]['composite_weight'] = max(
                    prev['composite_weight'], composite)
            else:
                G.add_edge(met_short, feat_name,
                           shap_imp=shap_imp,
                           composite_weight=composite,
                           feat_type=ftype)

            edge_rows.append({
                'Dataset':              ds,
                'Target_Metabolite':    met_short,
                'Feature':              feat_name,
                'Feature_Type':         ftype,
                'SHAP_Importance':      round(shap_imp, 5),
                'Target_Met_Best_AUC':  round(auc, 4),
                'Composite_Weight':     round(composite, 6),
            })

    return G, pd.DataFrame(edge_rows)


def visualise_fc_network(G, ds, edge_df, top_n_edges=50):
    """Spring-layout visualisation coloured by node type, edges by composite weight."""
    if len(G) == 0:
        print(f"  [{ds}] Empty graph — skipping visualisation.")
        return

    # Keep only top_n_edges by composite weight
    all_edges_sorted = sorted(G.edges(data=True),
                              key=lambda e: e[2].get('composite_weight', 0),
                              reverse=True)
    keep_edges = [(u, v) for u, v, _ in all_edges_sorted[:top_n_edges]]
    keep_nodes = set()
    for u, v in keep_edges:
        keep_nodes.update([u, v])
    Gsub = G.subgraph(keep_nodes).copy()
    rm = [(u, v) for u, v in Gsub.edges()
          if (u, v) not in keep_edges and (v, u) not in keep_edges]
    Gsub.remove_edges_from(rm)

    pos = nx.spring_layout(Gsub, seed=42, k=1.5 / max(1, len(Gsub) ** 0.5))

    node_colors, node_sizes = [], []
    for n, d in Gsub.nodes(data=True):
        nt = d.get('node_type', 'species')
        if nt == 'metabolite_target':
            node_colors.append('#9b59b6')   # purple — prediction target
            node_sizes.append(800 + 1200 * d.get('best_auc', 0.5))
        elif nt == 'species':
            node_colors.append('#2ecc71')   # green — species features
            node_sizes.append(300 + 600 * d.get('mean_shap', 0))
        elif nt == 'group':
            node_colors.append('#e67e22')   # orange — disease-status
            node_sizes.append(400)
        elif nt == 'confounder':
            node_colors.append('#95a5a6')   # gray — confounder
            node_sizes.append(200)
        else:
            node_colors.append('#3498db')
            node_sizes.append(300)

    weights = np.array([Gsub[u][v].get('composite_weight', 0) for u, v in Gsub.edges()])
    norm_w  = weights / weights.max() if weights.max() > 0 else weights
    edge_widths = 0.5 + 5.0 * norm_w
    edge_colors = ['#e74c3c' if Gsub[u][v].get('feat_type') == 'group' else
                   '#2ecc71' if Gsub[u][v].get('feat_type') == 'species' else '#95a5a6'
                   for u, v in Gsub.edges()]

    fig, ax = plt.subplots(figsize=(16, 14))
    nx.draw_networkx_edges(Gsub, pos, ax=ax,
                           width=edge_widths, edge_color=edge_colors, alpha=0.6)
    nx.draw_networkx_nodes(Gsub, pos, ax=ax,
                           node_color=node_colors, node_size=node_sizes, alpha=0.85)
    labels = {n: (n[:28] + '…' if len(n) > 30 else n) for n in Gsub.nodes()}
    nx.draw_networkx_labels(Gsub, pos, labels=labels, ax=ax, font_size=6.5)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#9b59b6', label='Target Metabolite (size = AUC)'),
        Patch(facecolor='#2ecc71', label='Species feature'),
        Patch(facecolor='#e67e22', label='Group / Disease-status'),
        Patch(facecolor='#95a5a6', label='Confounder'),
        Line2D([0], [0], color='#e74c3c', linewidth=2, label='Edge: Group feature'),
        Line2D([0], [0], color='#2ecc71', linewidth=2, label='Edge: Species feature'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.8)
    ax.set_title(
        f'Full-Cohort Weighted Network [{ds}]\n'
        f'Edge weight = SHAP × AUC  |  Top {top_n_edges} edges shown',
        fontsize=12
    )
    ax.axis('off')
    plt.tight_layout()
    out_path = CRC_RESULTS_DIR / 'figures' / 'network' / f'{ds}_FC_network_weighted.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Saved: {out_path.name}")


# ---- MAIN LOOP ----
if 'fc_shap_results' not in dir() or not fc_shap_results:
    print("fc_shap_results not found — run full-cohort SHAP cell first.")
else:
    all_edge_rows          = []
    fc_network_metrics_all = []

    for ds in CRC_DATASETS:
        print(f"\n{'='*60}")
        print(f"Full-Cohort Weighted Network [{ds}]")
        print(f"{'='*60}")

        fc_auc_local = (fc_auc_df if ('fc_auc_df' in dir()
                                      and fc_auc_df is not None)
                        else pd.DataFrame())
        G_fc, edge_df = build_fc_network(fc_shap_results, fc_auc_local, ds)
        if len(G_fc) == 0:
            print(f"  No data for {ds}")
            continue

        n_mt = sum(1 for _, d in G_fc.nodes(data=True) if d['node_type'] == 'metabolite_target')
        n_sp = sum(1 for _, d in G_fc.nodes(data=True) if d['node_type'] == 'species')
        n_gr = sum(1 for _, d in G_fc.nodes(data=True) if d['node_type'] == 'group')
        print(f"  Nodes: {G_fc.number_of_nodes()} "
              f"({n_mt} metabolite targets, {n_sp} species, {n_gr} group features)")
        print(f"  Edges: {G_fc.number_of_edges()}")

        # Top composite-weight edges
        top_e = edge_df.nlargest(15, 'Composite_Weight')
        print(f"\n  Top 15 edges by composite weight (SHAP x AUC):")
        for _, r in top_e.iterrows():
            ftype = r['Feature_Type'].upper()[:4]
            print(f"    [{ftype}] {r['Feature'][:40]:40s} -> {r['Target_Metabolite'][:30]:30s} "
                  f"w={r['Composite_Weight']:.4f}  "
                  f"(SHAP={r['SHAP_Importance']:.4f}, AUC={r['Target_Met_Best_AUC']:.3f})")

        # Fraction from disease-status features
        total_w = edge_df['Composite_Weight'].sum()
        group_w = edge_df[edge_df['Feature_Type'] == 'group']['Composite_Weight'].sum()
        frac_grp = group_w / total_w if total_w > 0 else 0
        print(f"\n  Disease-status (Group_) features account for "
              f"{frac_grp * 100:.1f}% of total composite network weight.")

        # Centrality
        metrics_df = compute_network_metrics(G_fc)
        print(f"\n  Top 10 nodes by degree:")
        print(metrics_df.head(10).to_string(index=False))

        # Community detection
        if HAS_COMMUNITY and len(G_fc) >= 3:
            try:
                partition = best_partition(G_fc, weight='composite_weight', random_state=42)
                nx.set_node_attributes(G_fc, partition, 'community')
                n_comm = len(set(partition.values()))
                print(f"\n  Louvain communities: {n_comm}")
            except Exception as ce:
                print(f"  Community detection skipped: {ce}")

        # Visualise
        visualise_fc_network(G_fc, ds, edge_df, top_n_edges=50)

        # Collect
        edge_df['Dataset'] = ds
        all_edge_rows.append(edge_df)

        for _, r in metrics_df.iterrows():
            auc_val = (G_fc.nodes[r['Node']].get('best_auc', None)
                       if r['Node'] in G_fc else None)
            fc_network_metrics_all.append({
                'Dataset':     ds,
                'Node':        r['Node'],
                'NodeType':    r['NodeType'],
                'Degree':      r['Degree'],
                'Betweenness': r['Betweenness'],
                'Best_AUC':    auc_val,
            })

    # Save edge table
    if all_edge_rows:
        all_edges_df = pd.concat(all_edge_rows, ignore_index=True)
        csv_path = CRC_RESULTS_DIR / 'tables' / 'ml_fc_network_weighted_edges.csv'
        all_edges_df.to_csv(csv_path, index=False)
        print(f"\nSaved edge table ({len(all_edges_df)} edges): {csv_path.name}")

    # Save node metrics
    if fc_network_metrics_all:
        nm_df   = pd.DataFrame(fc_network_metrics_all)
        nm_path = CRC_RESULTS_DIR / 'tables' / 'ml_fc_network_metrics.csv'
        nm_df.to_csv(nm_path, index=False)
        print(f"Saved node metrics ({len(nm_df)} nodes): {nm_path.name}")

    print("\nFull-cohort weighted network analysis complete.")
