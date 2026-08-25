################################################################################
##
## Marco's method, run on BOTH profile matrices, changing ONE thing.
##
##   basis "VV" : P = V         (42 x n_latent)   latent metric I       Marco, Teams 2:43
##   basis "VW" : P = V %*% W   (42 x n_pred)     latent metric W W'    Peter #246 / #247
##
## Everything downstream is identical between the two runs -- G = P %*% t(P),
## Louvain on |G| with the signs re-added, the top-k union fix, the layout, the
## plot styling, the node summary. So any difference between the two outputs is
## attributable to the profile matrix and to nothing else. That is the whole
## point: create_networks_v3.R used cos(V@W) and v4 used Gram(V), which differ in
## the metric AND in the normalisation at once, so their disagreement
## (strength spearman 0.071, module ARI 0.080) could not be pinned on either.
##
## Algebra:
##     (V W)(V W)' = V (W W') V'          vs          V V' = V (I) V'
## Choosing VV over VW is not "adding nothing"; it asserts W W' propto I.
## The notebook tests that assertion directly instead of leaving it implicit.
##
## Marco's method is untouched in both runs, in his own words:
##   * "use the Gram matrix directly"              -> edges = signed Gram P %*% t(P)
##   * "cosine ... is also circular" (do not use)  -> NO cosine
##   * "I didn't even do a Bray-Curtis"            -> NO Bray-Curtis
##   * "Negative weights have meaning. Do the Louvain on absolute values and
##      then use the original signs"               -> Louvain(|G|) + re-add sign
##   * "avoid ... the dendrogram"                  -> NO dendrogram
##
## Retains the v4 top-k fix (issue #1): the original group_by(from) after
## filter(from < to) silently drops a trait's strongest edges when its name sorts
## late, so the displayed neighbourhood depended on the trait's NAME. Here every
## node keeps its k strongest links over both endpoints.
##
## Reads   outputs/network_input_{MODEL}.xlsx   sheets "basis_VV" and "basis_VW"
##         (written by the notebook's export cell; sheet 1 col 1 = trait name)
## Writes  outputs/trait_network_gram_{MODEL}_{BASIS}.pdf
##         outputs/network_nodes_gram_{MODEL}_{BASIS}.csv
##         outputs/network_edges_gram_{MODEL}_{BASIS}.csv   <- new; lets the
##                 notebook recompute modularity and compare the two graphs
##                 without re-deriving anything in Python.
################################################################################

library(readxl)
library(igraph)
library(ggraph)
library(tidygraph)
library(dplyr)
library(purrr)
library(ggforce)
library(tibble)
library(readr)

################################################################################
## Configuration

setwd("C:/Users/cml/OneDrive - Universiteit Leiden/peng/flora_emb/S1_supplementary_code")

MODEL  <- "emb"                       # "emb" or "flora"
BASES  <- c("VV", "VW")               # run both; the comparison IS the result
TOP_K  <- 5
OUTDIR <- "outputs"
INFILE <- sprintf("%s/network_input_%s.xlsx", OUTDIR, MODEL)

dir.create(OUTDIR, showWarnings = FALSE)

################################################################################
## One function, two bases. Nothing here branches on BASIS except the sheet read.

build_network <- function(basis, model = MODEL, top_k = TOP_K) {

  message("\n================ ", model, " / basis ", basis, " ================")

  ## ---- profile matrix P: THE ONLY DIFFERENCE BETWEEN THE TWO RUNS ----------
  prof_df <- read_excel(INFILE, sheet = paste0("basis_", basis))
  prof <- as.data.frame(prof_df)
  rownames(prof) <- prof[[1]]
  prof[[1]] <- NULL
  prof[] <- lapply(prof, as.numeric)

  A <- as.matrix(prof)
  storage.mode(A) <- "numeric"
  message("  profiles: ", nrow(A), " traits x ", ncol(A), " columns")

  ## ---- signed Gram, Marco's way: no cosine, no Bray-Curtis, no row norm ----
  C <- A %*% t(A)
  diag(C) <- 0

  edges_signed <- as.data.frame(as.table(C)) |>
    rename(from = Var1, to = Var2, weight = Freq) |>
    mutate(from = as.character(from), to = as.character(to)) |>
    filter(from < to) |>
    mutate(
      sign       = ifelse(weight >= 0, "positive", "negative"),
      abs_weight = abs(weight),
      pair       = paste(from, to, sep = "\u0001")
    )

  ## ---- top-k union over BOTH endpoints (v4 fix, issue #1) ------------------
  keep_pairs <- bind_rows(
    edges_signed |> transmute(node = from, pair, abs_weight),
    edges_signed |> transmute(node = to,   pair, abs_weight)
  ) |>
    group_by(node) |>
    slice_max(abs_weight, n = top_k, with_ties = FALSE) |>
    ungroup() |>
    distinct(pair) |>
    pull(pair)

  edges_topk_signed <- edges_signed |>
    filter(pair %in% keep_pairs) |>
    select(from, to, weight, sign, abs_weight)

  g_signed <- graph_from_data_frame(
    edges_topk_signed,
    directed = FALSE,
    vertices = data.frame(name = rownames(C))
  )

  E(g_signed)$abs_weight     <- edges_topk_signed$abs_weight
  E(g_signed)$sign           <- edges_topk_signed$sign
  E(g_signed)$cluster_weight <- E(g_signed)$abs_weight

  ## ---- Louvain on |G|, signs restored for display only ---------------------
  set.seed(123)
  comm <- cluster_louvain(g_signed, weights = E(g_signed)$cluster_weight)

  V(g_signed)$community    <- as.factor(membership(comm))
  V(g_signed)$strength_abs <- strength(g_signed, weights = E(g_signed)$abs_weight)

  Q <- modularity(comm)
  message("  Louvain: ", length(unique(membership(comm))),
          " communities, Q = ", round(Q, 4))

  ## ---- cluster-separated layout -------------------------------------------
  clusters <- split(V(g_signed)$name, V(g_signed)$community)

  cluster_layouts <- imap_dfr(clusters, function(nodes, cl) {
    subg <- induced_subgraph(g_signed, vids = nodes)
    xy <- layout_with_fr(subg, weights = E(subg)$abs_weight)
    data.frame(name = V(subg)$name, community = cl,
               local_x = xy[, 1], local_y = xy[, 2])
  })

  n_clust <- length(clusters)
  radius  <- 11
  cluster_centers <- data.frame(
    community = names(clusters),
    angle = seq(0, 2 * pi, length.out = n_clust + 1)[-1]
  ) |>
    mutate(center_x = radius * cos(angle), center_y = radius * sin(angle))

  layout_separated <- cluster_layouts |>
    left_join(cluster_centers, by = "community") |>
    mutate(x = center_x + local_x, y = center_y + local_y)

  V(g_signed)$x <- layout_separated$x[match(V(g_signed)$name, layout_separated$name)]
  V(g_signed)$y <- layout_separated$y[match(V(g_signed)$name, layout_separated$name)]

  ## ---- plot ----------------------------------------------------------------
  legend_edges <- data.frame(x = -Inf, xend = -Inf, y = -Inf, yend = -Inf,
                             Association = c("positive", "negative"))

  p_net <- ggraph(g_signed, layout = "manual",
                  x = V(g_signed)$x, y = V(g_signed)$y) +
    ggforce::geom_mark_hull(
      data = layout_separated,
      aes(x = x, y = y, group = community, fill = community),
      inherit.aes = FALSE, concavity = 5, expand = unit(4, "mm"),
      alpha = 0.1, color = NA, show.legend = FALSE
    ) +
    geom_edge_link(aes(width = abs_weight, linetype = sign),
                   alpha = 0.50, color = "grey55", show.legend = FALSE) +
    geom_segment(data = legend_edges,
                 aes(x = x, xend = xend, y = y, yend = yend, linetype = Association),
                 inherit.aes = FALSE, linewidth = 1.2, color = "grey35",
                 show.legend = TRUE) +
    scale_edge_width(name = "Edge width: association", range = c(0.03, 1.5)) +
    scale_edge_linetype_manual(values = c(positive = "solid", negative = "22"),
                               guide = "none") +
    scale_linetype_manual(
      name = "Association",
      values = c(positive = "solid", negative = "22"),
      guide = guide_legend(override.aes = list(shape = NA, linewidth = 1.2,
                                               color = "grey35"))
    ) +
    geom_node_point(aes(size = strength_abs, color = community),
                    alpha = 0.95, show.legend = TRUE) +
    geom_node_text(aes(label = name), repel = TRUE, size = 4,
                   fontface = "bold", color = "black", show.legend = FALSE) +
    scale_size_continuous(name = "Node size: connectedness", range = c(3, 8)) +
    scale_color_discrete(name = "Cluster") +
    scale_fill_discrete(name = "Cluster") +
    ggtitle(sprintf("%s  --  basis %s  (G = P P', P = %s)", model, basis,
                    ifelse(basis == "VV", "V", "V W"))) +
    theme_void() +
    theme(legend.title = element_text(face = "bold"),
          legend.text = element_text(size = 9),
          plot.title = element_text(face = "bold", hjust = 0.5))

  print(p_net)
  ggsave(file.path(OUTDIR, sprintf("trait_network_gram_%s_%s.pdf", model, basis)),
         p_net, width = 12, height = 9)

  ## ---- node summary --------------------------------------------------------
  ## strength_full / mean_sim_full are threshold-free (no k, no cosine); the
  ## centralities below are defined on the top-k DISPLAY graph and inherit k.
  Cabs          <- abs(C)
  strength_full <- rowSums(Cabs)
  mean_sim_full <- strength_full / (nrow(C) - 1)
  own_norm      <- sqrt(rowSums(A^2))        # ||P_j||, the circularity diagnostic

  node_summary <- data.frame(
    trait         = V(g_signed)$name,
    basis         = basis,
    community     = V(g_signed)$community,
    degree        = degree(g_signed),
    strength      = strength(g_signed, weights = E(g_signed)$abs_weight),
    mean_strength = strength(g_signed, weights = E(g_signed)$abs_weight) /
                    pmax(degree(g_signed), 1),
    betweenness   = betweenness(g_signed, weights = 1 / E(g_signed)$abs_weight,
                                normalized = TRUE),
    closeness     = closeness(g_signed, weights = 1 / E(g_signed)$abs_weight,
                              normalized = TRUE),
    eigenvector   = eigen_centrality(g_signed, weights = E(g_signed)$abs_weight)$vector,
    strength_full = strength_full[match(V(g_signed)$name, names(strength_full))],
    mean_sim_full = mean_sim_full[match(V(g_signed)$name, names(mean_sim_full))],
    own_norm      = own_norm[match(V(g_signed)$name, names(own_norm))],
    row.names     = NULL
  ) |>
    arrange(desc(strength))

  write.csv(node_summary,
            file.path(OUTDIR, sprintf("network_nodes_gram_%s_%s.csv", model, basis)),
            row.names = FALSE)

  ## ---- edge list: full signed Gram, unthresholded --------------------------
  ## The notebook needs this to recompute Q and to compare the two graphs
  ## without re-deriving the profiles in Python.
  edges_out <- edges_signed |>
    mutate(basis = basis, in_topk = pair %in% keep_pairs) |>
    select(basis, from, to, weight, sign, abs_weight, in_topk)

  write.csv(edges_out,
            file.path(OUTDIR, sprintf("network_edges_gram_%s_%s.csv", model, basis)),
            row.names = FALSE)

  message("  wrote nodes / edges / pdf for basis ", basis)
  invisible(list(nodes = node_summary, edges = edges_out, Q = Q, graph = g_signed))
}

################################################################################
## Run both

res <- lapply(BASES, build_network)
names(res) <- BASES

################################################################################
## Side-by-side, so the comparison exists in R too and not only in the notebook

cmp <- res$VV$nodes |>
  select(trait, comm_VV = community, strength_VV = strength_full, own_VV = own_norm) |>
  inner_join(
    res$VW$nodes |>
      select(trait, comm_VW = community, strength_VW = strength_full, own_VW = own_norm),
    by = "trait"
  )

cat("\n================ VV vs VW ================\n")
cat(sprintf("node strength spearman   : %+.3f\n",
            cor(cmp$strength_VV, cmp$strength_VW, method = "spearman")))
cat(sprintf("module ARI               : %+.3f\n",
            igraph::compare(as.integer(cmp$comm_VV), as.integer(cmp$comm_VW),
                            method = "adjusted.rand")))
cat(sprintf("Q  VV = %.4f   VW = %.4f\n", res$VV$Q, res$VW$Q))
cat(sprintf("r(strength, ||P||)  VV = %+.3f   VW = %+.3f\n",
            cor(cmp$strength_VV, cmp$own_VV),
            cor(cmp$strength_VW, cmp$own_VW)))
cat("\n  r(strength, ||P||) near 1 means the node statistic is a relabelling of\n")
cat("  ||P_j||, i.e. of how much latent signal the model gave that trait.\n")

write.csv(cmp, file.path(OUTDIR, sprintf("network_compare_%s_VV_vs_VW.csv", MODEL)),
          row.names = FALSE)

cat("\ndone. Two networks written; the notebook reads the CSVs.\n")
