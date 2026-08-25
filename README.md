# Floristic Embeddings Encode Holistic Ecological Information Beyond Discrete Trait Measurements

Companion code and data-processing workflow for:

**Floristic embeddings encode holistic ecological information beyond discrete trait measurements.** Sun et al., 2026


---

## Models

Four model configurations are compared following the manuscript: EBPMF(EMB) EBPMF(traits)  BHPMF(R) and MICE 

---

## Repository structure

The notebooks read data with paths **relative to the repository root**, so run everything from the project root directory (the only exception is the BHPMF step, which is run from inside `R/`; see below).

```
S1_supplementary_code/
├── encoder.py                            # generate flora-description embeddings
├── EBPMF_run.ipynb                       # train EBPMF (emb + flora variants), MICE, prepare BHPMF inputs
├── EBPMF_run_species_holdout_dual.ipynb  # species-holdout robustness experiment (emb vs flora)
├── EBPMF_visulisation.ipynb              # main figures (model comparison, projection, RQ2, RQ3 networks)
├── README.md
├── python/                               # importable model / plotting modules
│   ├── bhpmf_torch.py                    #   EBPMF model code (EmbeddingPriorBPMF_GPU) used by the notebooks
│   ├── bhpmf.py                          #   CPU reference implementation (not used by the notebooks)
│   └── ebpmf_projection_viz.py           #   projection-figure helpers for the visualisation notebook
├── data/                                 # all inputs (csv / xlsx / npz)
├── outputs/                              # model outputs (npy / pkl / csv / xlsx / pdf)
└── R/                                    # BHPMF + trait-network workflow in R
    ├── run_bhpmf.R                       #   BHPMF gap filling (R 3.4.4 + BHPMF package)
    ├── create_networks.R                 #   RQ3 trait networks (signed Gram + Louvain, bases VV and VW)
    ├── extr_err.py                       #   parse the BHPMF console log into extracted_err_iter.csv
    ├── select_stop_iter.ipynb            #   choose the BHPMF early-stopping iteration
    ├── BHPMF_input/                      #   0617_X_matrix.csv, 0617_hierarchy_info.csv (written by EBPMF_run.ipynb)
    ├── your_log_0618.txt                 #   console log of the 1000-iteration long-chain run (provenance)
    └── extracted_err_iter.csv            #   per-iteration Test Err extracted from that log (provenance)
```

**Important — module location.** The importable modules live in `python/`: the model code (`bhpmf_torch.py`) and the projection-plot helpers (`ebpmf_projection_viz.py`). The notebooks reach them by putting `python/` on the import path in their first cell:

```python
import sys; sys.path.append("python")
```

so that `from bhpmf_torch import ...` and `from python.ebpmf_projection_viz import ...` both resolve. Run the notebooks from the repository root so `python/` is found.

Note that `sys.path` only affects `import`; it does **not** affect `pd.read_csv(...)` or `np.load(...)`, which resolve against the working directory. Data paths are therefore written out explicitly (the notebooks use the `data/` and `outputs/` prefixes).

---

## Main data files

All of these live in `data/`.

| File | Description |
| :---- | :---- |
| `floras_0609_clean.csv` | Cleaned description table (one row per species, `Description` and `scientificName` columns). This is the input of `encoder.py`, and its row order defines the row order of the embedding matrix. |
| `appendix4_flora_traits.xlsx` | Trait table extracted from the flora collection (the 28 flora-derived traits used as the EBPMF(traits) prior). |
| `selected_traits4.csv` | List of the 42 selected TRY target traits used in the analyses. |
| `species_all_traits_mean.csv` | TRY trait summary table with species-level mean trait values. |
| `flora_with_higher_taxaall.csv` | Higher-taxonomy lookup (order / family / genus) for species in the collection. |
| `flora_emb_multilingual_0617_with_metadata.npz` | Precomputed flora-description embeddings plus species names (output of `encoder.py`). Load with `np.load(path, allow_pickle=True)["embeddings"]`. |

`flora_3in1_1103_dirty.csv` and `appendix4_flora_traits.xlsx` originate from Sun et al., 2026.

On the word **`dirty`** in some filenames: this label is kept only for consistency with the original working files and does **not** mean the files are unusable — they are the expected inputs of the workflow.

---

## Main code files

| File | Description |
| :---- | :---- |
| `encoder.py` | Generates species embeddings from the flora descriptions (see details below). |
| `python/bhpmf_torch.py` | EBPMF model implementation imported by the notebooks (`EmbeddingPriorBPMF_GPU`; the file also contains `BPMF_GPU` / `BHPMF_GPU` and `run_*_gpu` convenience wrappers that are not used by the final workflow). |
| `python/bhpmf.py` | CPU (numpy/joblib) reference implementation of BPMF/BHPMF. Kept for reference; **not** imported by the notebooks. |
| `python/ebpmf_projection_viz.py` | Projection-figure helpers imported by `EBPMF_visulisation.ipynb` (`influence_summary`, `grouped_influence_heatmap`, `full_influence_heatmap`, `importance_barplot`, `received_influence_barplot`, plus `ROW_TITLES` / `LAYOUT`). The remaining network functions in this file belong to a superseded pipeline and are not used. |
| `EBPMF_run.ipynb` | Main training notebook. Prepares the master table and the random validation mask, trains both EBPMF variants (embedding prior and flora-trait prior, 10 seeds each), runs the MICE benchmark, saves the prediction pickles and the latent matrices (`V_*`, `W_*`), and writes the BHPMF input files to `R/BHPMF_input/`. |
| `EBPMF_visulisation.ipynb` | Main visualization notebook: model comparison, latent-space projection figures, RQ2 (sparsity / taxonomic conservatism), and the RQ3 trait-network analysis. Its network section exports a workbook for `R/create_networks.R` and reads that script's output back in (see the workflow below). |
| `R/run_bhpmf.R` | Runs BHPMF gap filling on `R/BHPMF_input/0617_*` and writes the imputed matrix. Final settings: `num.latent.feats = 26`, `num.samples = 482`, `burn = 200`, `gaps = 20`. The per-iteration internal Test Err is printed to the console, so the long-chain run must be captured to a log file. |
| `R/extr_err.py` | Parses the captured BHPMF console log into `extracted_err_iter.csv` (columns `iter`, `test_err`). |
| `R/select_stop_iter.ipynb` | Selects the BHPMF early-stopping iteration from `extracted_err_iter.csv` using one rule: burn-in 300, patience 100, `min_delta = 0`. On the final long-chain log this stops at iteration 481 (best iteration 381), i.e. `num.samples = 482`. Writes `early_stop_after_burnin.csv` (the full trace) next to the notebook. |
| `R/create_networks.R` | Builds the RQ3 trait networks from `outputs/network_input_emb.xlsx` (sheets `basis_VV` and `basis_VW`): signed Gram matrix G = P Pᵀ, Louvain on \|G\| with signs restored, top-k display graph. See the dedicated section below. |

---

## The embedding step (`encoder.py`) in detail

`encoder.py` turns each flora description into a dense multilingual embedding.

- **Model:** `intfloat/multilingual-e5-large-instruct`, loaded via `sentence-transformers`. On first run the weights are downloaded from the Hugging Face Hub, so this step needs internet access (or a pre-populated model cache).
- **Device:** automatically uses CUDA if a GPU is available, otherwise CPU.
- **Input:** reads the `Description` column from `data/floras_0609_clean.csv`; missing descriptions are encoded as empty strings. Rows are encoded **in file order and without deduplication**, because the downstream notebooks index the embedding matrix by row position.
- **Encoding:** `batch_size=32`, with `normalize_embeddings=True` (L2-normalized vectors).
- **Output:** saves the embedding matrix together with the species names as `data/flora_emb_multilingual_0617_with_metadata.npz` (keys `embeddings` and `scientificName`) — exactly the file and format the notebooks load.

**Dependencies:** `sentence-transformers`, `torch`, `pandas`, `numpy` (a CUDA-enabled `torch` build is recommended but not required for this step).

Run `encoder.py` from the repository root:

```
python encoder.py
```

---

## The RQ3 trait networks (`R/create_networks.R`) in detail

The script is built on the network code written by M. Baldo for this project and keeps his method unchanged: the edges are the **signed Gram matrix** of the trait profiles (no cosine, no Bray-Curtis, no row normalisation), Louvain is run on the absolute weights with the signs restored afterwards for display only, and no dendrogram is used. The layout, hull, edge and node styling, and the node-level statistics (`degree`, `strength`, `mean_strength`, `betweenness` with weights `1/|w|`, `closeness`, `eigenvector`) are taken over as written.

Three things are deliberately different from that original script, and are documented here because they affect the published numbers:

1. **Top-k selection over both endpoints.** The original kept, for each trait, its five strongest edges *among the partners whose name sorts after it* (`filter(from < to)` followed by `group_by(from)`), which makes a trait's displayed neighbourhood depend on its name. Here each node keeps its `TOP_K` strongest links regardless of endpoint order, so `degree`, `strength` and the centralities no longer depend on the alphabet.
2. **All 42 traits are retained as vertices**, including any that end up with no retained edge, so the node table always aligns row-for-row with the per-trait results.
3. **A second basis is computed.** The profile matrix P is the only quantity that differs between the two runs: `basis_VV` uses P = V (the original choice, latent metric I) and `basis_VW` uses P = V W (latent metric W Wᵀ). Everything downstream is identical, so any difference between the two outputs is attributable to the basis alone. **The figures reported in the chapter use `basis_VW`** (`BASIS = 'VW'` in the visualisation notebook).

Configuration lives at the top of the file: `MODEL <- "emb"`, `BASES <- c("VV", "VW")`, `TOP_K <- 5`, and a `setwd()` line that must be pointed at your local repository root. A complete run writes eight files to `outputs/`:

```
network_nodes_gram_emb_VV.csv     network_edges_gram_emb_VV.csv     trait_network_gram_emb_VV.pdf
network_nodes_gram_emb_VW.csv     network_edges_gram_emb_VW.csv     trait_network_gram_emb_VW.pdf
network_compare_emb_VV_vs_VW.csv
```

---

## Recommended workflow

Run the analysis in the following order.

### 1. Generate flora-description embeddings

```
python encoder.py
```

Produces `data/flora_emb_multilingual_0617_with_metadata.npz`, where the notebooks expect it. A precomputed copy is already provided in `data/`, so this step is optional if you only want to reproduce the downstream results.

### 2. Train EBPMF models and reproduce the core results

Open and run `EBPMF_run.ipynb` from the top. This trains and evaluates both EBPMF variants and the MICE benchmark, saves `outputs/0617_EBPMF_emb.pkl`, `outputs/0617_EBPMF_flora.pkl`, `outputs/0617_MICE.pkl`, the latent matrices `outputs/{V,W}_{emb,flora}_0706.npy`, and writes the BHPMF input files `R/BHPMF_input/0617_X_matrix.csv` and `R/BHPMF_input/0617_hierarchy_info.csv`.

Both training cells must save their latent matrices, since the visualisation notebook checks that `V_emb_0706.npy` and `V_flora_0706.npy` are not byte-identical before using them.


### 3. Run BHPMF with a long chain

`run_bhpmf.R` reads its inputs relative to `R/`, so run it **from inside `R/`**, and capture the console output — `extr_err.py` parses the per-iteration `Test Err` lines from it:

```
cd R
# in run_bhpmf.R set:  num.samples <- 1000
Rscript run_bhpmf.R 2>&1 | tee your_log.txt
```

The BHPMF package requires an old R (we used R 3.4.4 inside a Docker container submitted through Slurm, with `R_LIBS_USER` pointing at the container's library path). Any environment that provides R ≈3.4 with the `BHPMF` and `Matrix` packages will work.

### 4. Extract the per-iteration RMSE records

```
python R/extr_err.py
```

Produces `R/extracted_err_iter.csv` (columns `iter`, `test_err`). Check that the input log filename inside `extr_err.py` matches the log you captured.

### 5. Select the early-stopping iteration

Open and run `R/select_stop_iter.ipynb` (from inside `R/`). With burn-in 300 and patience 100, the final long-chain log (`your_log_0618.txt`) stops at iteration 481, hence `num.samples = 482`.

### 6. Re-run BHPMF at the selected iteration

Set `num.samples <- 482` in `R/run_bhpmf.R` and run it again (again from inside `R/`). Then copy the result to where the visualisation notebook reads it:

```
cp R/final_imputed_full.csv outputs/0617_final_imputed_full.csv
```

(The values are on the z-scored trait scale; `EBPMF_visulisation.ipynb` transforms them back.)

### 7. Reproduce the main visualizations

Open `EBPMF_visulisation.ipynb` and run it from the top. The model-comparison, projection, and RQ2 figures run straight through. The RQ3 trait-network section hands off to R partway through the notebook:

1. Run the notebook up to and including the RQ3 network-export cell. It writes `outputs/network_input_emb.xlsx` (sheets `basis_VV` and `basis_VW`, plus a README sheet with checksums).
2. Run `R/create_networks.R` with `MODEL <- "emb"` (the default), after pointing its `setwd()` line at your local repository root. It reads the workbook, builds both networks, runs Louvain clustering, and writes the node / edge / comparison tables and the network PDFs back to `outputs/`.
3. Return to the notebook and run the remaining RQ3 cells. They read `outputs/network_nodes_gram_emb_VW.csv` to produce the betweenness figure (`outputs/fig_rq3_betweenness_main_emb.pdf`) and the modularity / ARI statistics.

### The whole frame
```
encoder.py
    ↓
EBPMF_run.ipynb                       # models, MICE, BHPMF inputs
    ↓
R/run_bhpmf.R                         # long chain (num.samples = 1000), capture the log
    ↓
R/extr_err.py
    ↓
R/select_stop_iter.ipynb              # burn-in 300, patience 100 → stop iter 481
    ↓
R/run_bhpmf.R                         # rerun with num.samples = 482; copy result to outputs/
    ↓
EBPMF_visulisation.ipynb              # run through the RQ3 network-export cell
    ↓
R/create_networks.R                   # build the trait networks from the exported workbook
    ↓
EBPMF_visulisation.ipynb              # resume: read the R output, finish the RQ3 figures
```

## Environment

### Python

Used for embedding generation, EBPMF training, the MICE comparison, data processing, projection, and visualization. Core packages:

- `numpy`, `pandas` — numerical computing and data manipulation
- `torch` — model training (EBPMF) and embedding generation
- `sentence-transformers` — flora-description embeddings (`encoder.py`)
- `scikit-learn` — `IterativeImputer`, PCA, metrics
- `statsmodels` — the MICE benchmark (`MICEData`) and the partial regressions in the RQ3 figure
- `scipy` — statistics (`pearsonr`)
- `networkx` — graph statistics for the RQ3 permutation tests
- `matplotlib`, `seaborn` — plotting
- `openpyxl` — reading / writing the Excel workbooks exchanged with R
- `adjustText` — label placement in the RQ3 betweenness figure
- `jupyter` — notebook execution

**A CUDA GPU is required for the training notebooks** (`EBPMF_run.ipynb`, and the data-preparation cell of `EBPMF_visulisation.ipynb`): the model runs on `device='cuda'`, and the random validation mask is drawn with a CUDA generator, so a CPU-only run would also produce a different mask. `encoder.py` alone falls back to CPU.

Install any further dependencies reported by the notebooks at run time.

### R

R is used for the BHPMF comparison and for the RQ3 trait-network step.

- `R/run_bhpmf.R` needs R ≈3.4 with the `BHPMF` and `Matrix` packages. We ran it in a Docker container (R 3.4.4) through Slurm, with `R_LIBS_USER` set to the container's library path; the script picks that variable up automatically. Run it from inside `R/`.
- `R/create_networks.R` runs on a current R and needs `readxl`, `igraph`, `ggraph`, `tidygraph`, `dplyr`, `purrr` and `ggforce`. It sets its own working directory near the top of the file — point that at your local repository root before running it.


---

