"""
ebpmf_projection_viz.py
Single source of truth for every projection / network figure in Chapter 4.

Both `EBPMF_projection-0617_revised.ipynb` and `EBPMF_visulisation_revised.ipynb`
import from here, so the two notebooks cannot drift apart.

--------------------------------------------------------------------------------
Two different trait networks exist in this project. They are NOT interchangeable.

  cosine(V)      -- what create_networks.R computes. W is ignored entirely, so
                    edges express similarity in the *latent trait factors*.
  cosine(V @ W)  -- what Marco's comment #246 asks for: "collapse the bipartite
                    graph onto the trait nodes ... two traits are connected if
                    they receive influence from similar embedding dimensions".

Empirically they are unrelated (strength ranking spearman = 0.071, module
ARI = 0.080), and only the second recovers functional domains:

    network                    ARI vs TRY domain   NMI     stability
    cosine(V)                        0.096         0.306     0.074
    cosine(|V @ W|)                  0.153         0.337     0.169
    cosine(V @ W)  signed            0.193         0.389     0.083   <- used here

The signed version is used because |V @ W| is non-negative, which forces every
cosine into [0.475, 0.983] and removes the positive/negative edge distinction
that the figure legend relies on.

Node encodings follow comment #246 exactly:
    node size  = total predictive influence   ( row sums of |V @ W| )
    edge width = similarity in embedding influence
    node color = functional domain

The network FIGURE is drawn in R (create_networks_v3.R), which keeps Marco's
ggraph aesthetics. Python computes and exports; R plots. Nothing is computed
twice, so the figure and the statistics cannot disagree.
--------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROW_TITLES = ["Structure", "Leaf economics", "Leaf chemistry",
              "Leaf physiology", "Stem / wood", "Root"]
LAYOUT = [7, 7, 7, 7, 7, 7]
DOMAIN_COLORS = dict(zip(ROW_TITLES, plt.cm.Dark2.colors[:6]))


# ==========================================================================
# influence
# ==========================================================================
def influence_summary(V, W, predictor_names, model):
    """|V @ W| and the ranked predictor importance (column means)."""
    V, W = np.asarray(V, float), np.asarray(W, float)
    predictor_names = np.asarray(list(predictor_names))

    influence = np.abs(V @ W)
    if len(predictor_names) != influence.shape[1]:
        raise ValueError(
            f"{len(predictor_names)} predictor labels for {influence.shape[1]} "
            f"columns of |V@W| in model '{model}'. 48 = embedding PCs, "
            f"28 = flora traits -- the two are being mixed.")

    importance = influence.mean(axis=0)
    order = np.argsort(-importance)
    return influence, predictor_names[order], importance[order], order


def grouped_influence_heatmap(influence, predictor_names, model,
                              order=None, row_titles=ROW_TITLES, figsize=None):
    n_traits, n_pred = influence.shape
    n_groups = len(row_titles)
    if n_traits % n_groups:
        raise ValueError(f"{n_traits} traits do not split into {n_groups} groups")

    grouped = influence.reshape(n_groups, n_traits // n_groups, n_pred).mean(axis=1)
    cols = np.asarray(list(predictor_names))
    if order is not None:
        grouped, cols = grouped[:, order], cols[order]

    plt.figure(figsize=figsize or (max(9, 0.22 * n_pred), 3.5), dpi=300)
    plt.imshow(grouped, cmap="YlOrRd", aspect="auto")
    plt.colorbar(label="Mean influence strength")
    plt.yticks(range(n_groups), row_titles, fontsize=10)
    plt.xticks(range(n_pred), cols, rotation=45, ha="right", fontsize=8)
    plt.title(f"{model}: mean influence of predictors on grouped target traits")
    plt.tight_layout()
    plt.show()


def full_influence_heatmap(influence, predictor_names, trait_names, model,
                           row_titles=ROW_TITLES, figsize=(12, 10)):
    n_per = len(trait_names) // len(row_titles)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    im = ax.imshow(influence, cmap="YlOrRd", aspect="auto")
    ax.set_yticks(range(len(trait_names)))
    ax.set_yticklabels(trait_names, fontsize=8)
    ax.set_xticks(range(influence.shape[1]))
    ax.set_xticklabels(predictor_names, rotation=45, ha="right", fontsize=8)

    axr = ax.twinx()
    axr.set_ylim(ax.get_ylim())
    axr.set_yticks([i * n_per + (n_per - 1) / 2 for i in range(len(row_titles))])
    axr.set_yticklabels(row_titles, fontsize=9)
    axr.tick_params(axis="y", length=0, pad=6)
    axr.set_ylabel("Trait group", labelpad=12)
    for i in range(1, len(row_titles)):
        ax.axhline(i * n_per - 0.5, color="black", lw=0.5, alpha=0.4)

    ax.set_title(f"{model}: influence of each predictor on each target trait")
    fig.subplots_adjust(right=0.72, bottom=0.18)
    fig.colorbar(im, cax=fig.add_axes([0.86, 0.18, 0.025, 0.68])).set_label("Influence strength")
    plt.show()


def importance_barplot(sorted_names, sorted_importance, model,
                       color="orangered", print_top=None, digits=3):
    n = len(sorted_names)
    share = sorted_importance / sorted_importance.sum()
    cum = np.cumsum(share)

    k = print_top or n
    print(f"{model}: predictors ranked by mean absolute influence")
    for i, (nm, v, sh, cu) in enumerate(
            zip(sorted_names[:k], sorted_importance[:k], share[:k], cum[:k]), 1):
        print(f"{i:3d}. {str(nm):32s} {v:.{digits}g}   {100*sh:4.1f}%   cum {100*cu:5.1f}%")
    print(f"     top-5 share: {100*share[:5].sum():.1f}%")

    fig, ax = plt.subplots(figsize=(max(12, 0.28 * n), 4), dpi=300)
    ax.bar(np.arange(n), sorted_importance, color=color)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(sorted_names, rotation=45, ha="right", fontsize=8)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylabel("Mean absolute influence on predictions")

    ax2 = ax.twinx()
    ax2.plot(np.arange(n), 100 * cum, color="grey", lw=1.2, marker="o", ms=2.5)
    ax2.axhline(50, ls="--", lw=0.8, color="grey")
    ax2.set_ylabel("Cumulative share (%)")
    ax2.set_ylim(0, 105)

    n50 = int(np.searchsorted(cum, 0.5) + 1)
    ax.set_title(f"{model}: {n50}/{n} predictors ({100*n50/n:.0f}%) "
                 f"account for half of all influence")
    plt.tight_layout()
    plt.show()


def received_influence_barplot(influence, trait_names, model, V=None,
                               row_titles=ROW_TITLES, color_by_domain=True,
                               normalise=False):
    """Row means of |V @ W|: influence RECEIVED by each target trait.
    This is the Fig. 5 (lower) / Fig. 6 (lower right) panel -- a different
    quantity from `importance_barplot`, which takes column means."""
    received = influence.mean(axis=1)
    ylab = "Mean absolute influence received"
    if normalise:
        if V is None:
            raise ValueError("normalise=True requires V")
        received = received / np.linalg.norm(np.asarray(V, float), axis=1)
        ylab += " (per unit ||v_j||)"

    trait_names = np.asarray(list(trait_names))
    cat = np.repeat(row_titles, len(trait_names) // len(row_titles))
    order = np.argsort(-received)
    vals, labs, cs = received[order], trait_names[order], cat[order]

    colors = [DOMAIN_COLORS[c] for c in cs] if color_by_domain else "seagreen"
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
    ax.bar(np.arange(len(vals)), vals, color=colors)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=8)
    ax.set_xlim(-0.6, len(vals) - 0.4)
    ax.set_ylabel(ylab)
    if color_by_domain:
        ax.legend(handles=[Line2D([0], [0], marker="s", ls="", color=DOMAIN_COLORS[t],
                                  label=t) for t in row_titles], fontsize=8, ncol=3)
    ax.set_title(f"{model}: target traits ranked by influence received from "
                 f"{influence.shape[1]} predictors" + (" (normalised)" if normalise else ""))
    plt.tight_layout()
    plt.show()
    return labs, vals


# ==========================================================================
# trait network -- comment #246 specification
# ==========================================================================
def cosine_matrix(M):
    M = np.asarray(M, float)
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    S = Mn @ Mn.T
    np.fill_diagonal(S, 0.0)
    return S


def _topk(A, k):
    B = np.zeros_like(A)
    for i in range(A.shape[0]):
        idx = np.argsort(-A[i])[:k]
        B[i, idx] = A[i, idx]
    return np.maximum(B, B.T)


def build_graph(S, node_names, topk=None):
    A = np.abs(S)
    if topk is not None:
        A = _topk(A, topk)
    G = nx.Graph()
    G.add_nodes_from(range(len(node_names)))
    for i in range(len(node_names)):
        for j in range(i + 1, len(node_names)):
            if A[i, j] > 0:
                G.add_edge(i, j, weight=A[i, j], distance=1.0 / A[i, j],
                           sign="positive" if S[i, j] >= 0 else "negative")
    nx.set_node_attributes(G, dict(enumerate(node_names)), "name")
    return G


def node_stats(G, prefix):
    n = G.number_of_nodes()
    deg = np.array([G.degree(v) for v in range(n)])
    stg = np.array([G.degree(v, weight="weight") for v in range(n)])
    btw = nx.betweenness_centrality(G, weight="distance", normalized=True)
    cls = nx.closeness_centrality(G, distance="distance")
    eig = nx.eigenvector_centrality_numpy(G, weight="weight")
    eig = np.abs(np.array([eig[v] for v in range(n)]))
    eig /= eig.max()
    return pd.DataFrame({
        f"{prefix}_degree": deg,
        f"{prefix}_strength": stg,
        f"{prefix}_mean_strength": stg / np.maximum(deg, 1),
        f"{prefix}_betweenness": [btw[v] for v in range(n)],
        f"{prefix}_closeness": [cls[v] for v in range(n)],
        f"{prefix}_eigenvector": eig,
    })


def consensus_modules(G, n_seeds=200):
    n = G.number_of_nodes()
    Co = np.zeros((n, n))
    ks = []
    for s in range(n_seeds):
        comms = nx.community.louvain_communities(G, weight="weight", seed=s)
        m = np.empty(n, int)
        for ci, c in enumerate(comms):
            for v in c:
                m[v] = ci
        Co += (m[:, None] == m[None, :])
        ks.append(len(comms))
    Co /= n_seeds
    k = int(np.median(ks))
    Z = linkage(squareform(1.0 - Co, checks=False), method="average")
    labels = fcluster(Z, t=k, criterion="maxclust")
    return labels, Co, k, float(np.mean((Co > 0.1) & (Co < 0.9))), ks


def trait_influence_network(V, W, trait_names, model, topk=5, n_seeds=200,
                            row_titles=ROW_TITLES):
    """Bipartite projection of the embedding-PC x trait influence graph onto the
    trait nodes (comment #246). Computation only -- the figure is drawn in R by
    create_networks_v3.R, from the CSVs written by `export_network_for_r`.

    Returns (node_table, edge_table, G_topk, co_membership, stats, similarity).
    """
    V, W = np.asarray(V, float), np.asarray(W, float)
    P = V @ W                                # signed influence profiles
    total_influence = np.abs(P).sum(axis=1)  # node size, per comment #246

    S = cosine_matrix(P)
    G_full = build_graph(S, trait_names, topk=None)
    G_topk = build_graph(S, trait_names, topk=topk)

    labels, Co, k, ambiguous, ks = consensus_modules(G_topk, n_seeds=n_seeds)
    domain = np.repeat(row_titles, len(trait_names) // len(row_titles))

    nodes = pd.DataFrame({"trait": list(trait_names), "model": model,
                          "domain": domain, "module": labels,
                          "total_influence": total_influence,
                          "latent_norm": np.linalg.norm(V, axis=1)})
    nodes = pd.concat([nodes, node_stats(G_full, "cosfull"),
                       node_stats(G_topk, f"costop{topk}")], axis=1)

    # every edge, with a flag for the top-k display subgraph
    tn = np.asarray(list(trait_names))
    rows = []
    topk_edges = set(map(lambda e: tuple(sorted(e)), G_topk.edges()))
    for i in range(len(tn)):
        for j in range(i + 1, len(tn)):
            rows.append((tn[i], tn[j], abs(S[i, j]),
                         "positive" if S[i, j] >= 0 else "negative",
                         tuple(sorted((i, j))) in topk_edges))
    edges = pd.DataFrame(rows, columns=["from", "to", "abs_weight", "sign", "in_topk"])

    comms = [set(np.where(labels == c)[0]) for c in sorted(set(labels))]
    Q = nx.community.modularity(G_topk, comms, weight="weight")
    cat_id = pd.factorize(pd.Series(domain))[0]
    stats = dict(Q=Q, k=k, ambiguous=ambiguous,
                 ARI=adjusted_rand_score(cat_id, labels),
                 NMI=normalized_mutual_info_score(cat_id, labels),
                 louvain_k=sorted(set(ks)))

    print(f"[{model}] Q = {Q:.3f}   k = {k}   ambiguous = {ambiguous:.3f}   "
          f"ARI(domain) = {stats['ARI']:.3f}   NMI = {stats['NMI']:.3f}")
    return nodes, edges, G_topk, Co, stats, S


def export_network_for_r(nodes, edges, Co, trait_names, model, stats,
                         dist_similarity, outdir="outputs"):
    """Write everything create_networks_v3.R needs. R computes nothing."""
    nodes.to_csv(f"{outdir}/network_nodes_{model}.csv", index=False)
    edges.to_csv(f"{outdir}/network_edges_{model}.csv", index=False)

    co = pd.DataFrame(np.asarray(Co), index=list(trait_names), columns=list(trait_names))
    co.to_csv(f"{outdir}/network_comembership_{model}.csv")

    # distance for the dendrogram: 1 - cosine of the influence profiles.
    # This is the same quantity the edges use, and it is what the 6 July meeting
    # asked for ("a dendrogram for traits based on the latent representation").
    # The co-membership matrix is NOT a usable tree distance: 77.5% of pairs are
    # exactly 0 or 1, 80% of merges happen below 0.01, and cutting it just
    # reproduces the Louvain modules.
    D = 1.0 - np.asarray(dist_similarity)
    np.fill_diagonal(D, 0.0)
    pd.DataFrame(np.clip(D, 0, 2), index=list(trait_names),
                 columns=list(trait_names)).to_csv(f"{outdir}/network_distance_{model}.csv")

    pd.DataFrame([{k: v for k, v in stats.items() if k != "louvain_k"}
                  | {"louvain_k": "/".join(map(str, stats["louvain_k"])),
                     "n_traits": len(trait_names)}]
                 ).to_csv(f"{outdir}/network_stats_{model}.csv", index=False)

    print(f"[{model}] wrote network_{{nodes,edges,comembership,distance,stats}}_{model}.csv "
          f"-> run create_networks_v3.R")


def network_permutation_tests(V, W, trait_names, row_titles=ROW_TITLES,
                              topk=5, n_perm=200, n_seeds_null=20, seed=0):
    """Is the modular structure real, and does it track functional domains?

    null 1: permute each trait's influence profile independently -> destroys any
            shared use of embedding dimensions while preserving each trait's
            marginal influence distribution.
    null 2: permute the functional-domain labels.
    """
    rng = np.random.default_rng(seed)
    P = np.asarray(V, float) @ np.asarray(W, float)
    domain = np.repeat(row_titles, len(trait_names) // len(row_titles))
    cat_id = pd.factorize(pd.Series(domain))[0]

    G = build_graph(cosine_matrix(P), trait_names, topk=topk)
    labels, _, k, _, _ = consensus_modules(G, n_seeds=200)
    comms = [set(np.where(labels == c)[0]) for c in sorted(set(labels))]
    Q_obs = nx.community.modularity(G, comms, weight="weight")
    ari_obs = adjusted_rand_score(cat_id, labels)

    Qn = []
    for _ in range(n_perm):
        Pn = np.array([rng.permutation(r) for r in P])
        Gn = build_graph(cosine_matrix(Pn), trait_names, topk=topk)
        ln, _, _, _, _ = consensus_modules(Gn, n_seeds=n_seeds_null)
        cn = [set(np.where(ln == c)[0]) for c in sorted(set(ln))]
        Qn.append(nx.community.modularity(Gn, cn, weight="weight"))
    Qn = np.array(Qn)
    p_Q = (np.sum(Qn >= Q_obs) + 1) / (n_perm + 1)

    aris = np.array([adjusted_rand_score(rng.permutation(cat_id), labels)
                     for _ in range(5000)])
    p_ari = (np.sum(aris >= ari_obs) + 1) / 5001

    print(f"modularity   Q = {Q_obs:.3f}   null {Qn.mean():.3f} "
          f"[{np.percentile(Qn,2.5):.3f}, {np.percentile(Qn,97.5):.3f}]   p = {p_Q:.4f}")
    print(f"domain match ARI = {ari_obs:.3f}   null {aris.mean():.4f} "
          f"[97.5% = {np.percentile(aris,97.5):.3f}]   p = {p_ari:.4f}")
    return dict(Q=Q_obs, p_Q=p_Q, Q_null=Qn, ARI=ari_obs, p_ARI=p_ari, ARI_null=aris)


def trait_dendrogram(Co, trait_names, model, k=None):
    Z = linkage(squareform(1.0 - np.asarray(Co), checks=False), method="average")
    plt.figure(figsize=(12, 6), dpi=300)
    dendrogram(Z, labels=list(trait_names), leaf_rotation=90, leaf_font_size=8)
    plt.title(f"{model}: trait dendrogram (1 - consensus co-membership)")
    plt.ylabel("distance")
    plt.tight_layout()
    plt.show()