

# --------------------------------------------------------------------------"""
Correlation networks — bipartite NetworkX graphs (ρ≥0.3, q<0.05); Louvain community detection; polyamine-centric 1-hop subnetwork extraction and visualization.
"""


import networkx as nx

try:
    from community import best_partition
    HAS_COMMUNITY = True
except ImportError:
    print("python-louvain not installed. Community detection will be skipped.")
    HAS_COMMUNITY = False

import os
os.makedirs(CRC_RESULTS_DIR / 'figures' / 'network', exist_ok=True)


def build_correlation_network(corr_df, rho_threshold=0.3, q_threshold=0.05):
    """Build a bipartite network from significant correlations."""
    G = nx.Graph()
    sig = corr_df[
        (corr_df['QValue'] < q_threshold) &
        (corr_df['Rho'].abs() >= rho_threshold)
    ]
    for _, row in sig.iterrows():
        G.add_node(row['Species'], bipartite=0, node_type='species')
        G.add_node(row['Metabolite'], bipartite=1, node_type='metabolite')
        G.add_edge(
            row['Species'], row['Metabolite'],
            weight=abs(row['Rho']),
            sign='positive' if row['Rho'] > 0 else 'negative',
            rho=row['Rho'],
            qvalue=row['QValue'],
        )
    return G


def compute_network_metrics(G):
    """Compute centrality metrics for network nodes."""
    if len(G) == 0:
        return pd.DataFrame()
    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    return pd.DataFrame({
        'Node': list(G.nodes()),
        'Degree': [degree[n] for n in G.nodes()],
        'Betweenness': [betweenness[n] for n in G.nodes()],
        'NodeType': [G.nodes[n].get('node_type', 'unknown') for n in G.nodes()],
    }).sort_values('Degree', ascending=False)


# Build networks per dataset
networks = {}
network_metrics = {}

for ds in CRC_DATASETS:
    if ds not in full_corr or full_corr[ds] is None or full_corr[ds].empty:
        continue

    G = build_correlation_network(full_corr[ds], rho_threshold=0.3, q_threshold=0.05)
    networks[ds] = G
    metrics = compute_network_metrics(G)
    network_metrics[ds] = metrics

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_species = sum(1 for _, d in G.nodes(data=True) if d.get('node_type') == 'species')
    n_mtb = sum(1 for _, d in G.nodes(data=True) if d.get('node_type') == 'metabolite')
    print(f"{ds}: {n_nodes} nodes ({n_species} species, {n_mtb} metabolites), {n_edges} edges")

# Community detection
if HAS_COMMUNITY:
    for ds, G in networks.items():
        if len(G) < 3:
            continue
        partition = best_partition(G)
        nx.set_node_attributes(G, partition, 'community')
        n_communities = len(set(partition.values()))
        modularity = nx.community.modularity(
            G, [{n for n, c in partition.items() if c == comm}
                for comm in set(partition.values())]
        )
        print(f"{ds}: {n_communities} communities, modularity={modularity:.3f}")

# Polyamine-centric subnetworks
polyamine_networks = {}

for ds in CRC_DATASETS:
    if ds not in networks or len(networks[ds]) == 0:
        continue

    G = networks[ds]
    pa_nodes = [col for col in polyamine_columns[ds].values() if col in G]

    if not pa_nodes:
        print(f"{ds}: No polyamine nodes in network.")
        continue

    # Extract 1-hop neighborhood
    neighbors = set(pa_nodes)
    for n in pa_nodes:
        neighbors.update(G.neighbors(n))

    subG = G.subgraph(neighbors).copy()
    polyamine_networks[ds] = subG
    print(f"{ds}: Polyamine subnetwork -- {subG.number_of_nodes()} nodes, {subG.number_of_edges()} edges")

# Visualize polyamine subnetworks
for ds, subG in polyamine_networks.items():
    if subG.number_of_nodes() < 2:
        continue

    pa_node_set = set(polyamine_columns[ds].values())

    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(subG, k=2, seed=42)

    colors, sizes = [], []
    for n in subG.nodes():
        if n in pa_node_set:
            colors.append('red'); sizes.append(400)
        elif subG.nodes[n].get('node_type') == 'species':
            colors.append('skyblue'); sizes.append(150)
        else:
            colors.append('lightgreen'); sizes.append(150)

    edge_colors = ['red' if subG[u][v].get('sign') == 'negative' else 'green'
                   for u, v in subG.edges()]

    nx.draw(subG, pos, node_color=colors, node_size=sizes,
            edge_color=edge_colors, alpha=0.7, ax=ax, with_labels=False)

    labels = {}
    for n in subG.nodes():
        if n in pa_node_set:
            labels[n] = n.split('_', 1)[-1] if '_' in n else n
        elif subG.nodes[n].get('node_type') == 'species':
            labels[n] = n[:25]
    nx.draw_networkx_labels(subG, pos, labels, font_size=7, ax=ax)

    ax.set_title(f'{ds} -- Polyamine-Centric Network\n'
                 '(red=polyamines, blue=species, green=other metabolites)', fontsize=12)
    plt.tight_layout()
    plt.savefig(CRC_RESULTS_DIR / 'figures' / 'network' / f'{ds}_polyamine_network.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

# Hub species connected to polyamines
print("\nHub species connected to polyamines:")
print("=" * 80)

hub_species_all = []
for ds, subG in polyamine_networks.items():
    pa_node_set = set(polyamine_columns[ds].values())
    species_nodes = [n for n in subG.nodes() if subG.nodes[n].get('node_type') == 'species']

    for spc in species_nodes:
        pa_neighbors = [nb for nb in subG.neighbors(spc) if nb in pa_node_set]
        if pa_neighbors:
            hub_species_all.append({
                'Dataset': ds,
                'Species': spc,
                'N_polyamine_connections': len(pa_neighbors),
                'Degree': subG.degree(spc),
                'Connected_polyamines': ', '.join(pa_neighbors),
            })

hub_df = pd.DataFrame(hub_species_all).sort_values('N_polyamine_connections', ascending=False)
display(hub_df.head(20))
hub_df.to_csv(CRC_RESULTS_DIR / 'tables' / 'polyamine_hub_species.csv', index=False)