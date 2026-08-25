# ============================================================
# run_bhpmf.R
# Run BHPMF gap filling with R 3.4.4
# ============================================================

options(stringsAsFactors = FALSE)

# Use user library set by the Slurm script.
# In the container this should be /r_libs.
if (nzchar(Sys.getenv("R_LIBS_USER"))) {
  .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
}

cat("============================================================\n")
cat("BHPMF run started\n")
cat("Time:", as.character(Sys.time()), "\n")
cat("R version:", R.version.string, "\n")
cat("Working directory:", getwd(), "\n")
cat("R_LIBS_USER:", Sys.getenv("R_LIBS_USER"), "\n")
cat("Library paths:\n")
print(.libPaths())
cat("============================================================\n\n")

suppressPackageStartupMessages({
  library(Matrix)
  library(BHPMF)
})

# ---------- User settings ----------
x_file <- "BHPMF_input/0617_X_matrix.csv"
hierarchy_file <- "BHPMF_input/0617_hierarchy_info.csv"

tmp.dir <- "bhpmf_tmp"

mean_out <- file.path(tmp.dir, "mean_gap_filled.txt")
std_out  <- file.path(tmp.dir, "std_gap_filled.txt")

# Main BHPMF parameters
prediction.level <- 4
used.num.hierarchy.levels <- 3
num.samples <- 482  # 1000
burn <- 200
gaps <- 20
num.latent.feats <- 24

tuning <- FALSE
num.folds.tuning <- 10
rmse.plot.test.data <- FALSE
verbose <- TRUE

# Filtering
min_obs_per_row <- 1

# Output filenames
final_imputed_file <- "final_imputed_full.csv"
final_imputed_with_id_file <- "final_imputed_full_with_id.csv"
final_std_file <- "final_std_full.csv"
final_std_with_id_file <- "final_std_full_with_id.csv"

# ---------- Safety checks ----------
if (!file.exists(x_file)) {
  stop("Input file not found: ", x_file)
}

if (!file.exists(hierarchy_file)) {
  stop("Input file not found: ", hierarchy_file)
}

cat("Input files found:\n")
cat("  X matrix:", x_file, "\n")
cat("  hierarchy:", hierarchy_file, "\n\n")

# ---------- 1) Read input ----------
cat("Reading input files...\n")

X_df <- read.csv(
  x_file,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

hierarchy_all <- read.csv(
  hierarchy_file,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

cat("Raw X dim:", paste(dim(X_df), collapse = " x "), "\n")
cat("Raw hierarchy dim:", paste(dim(hierarchy_all), collapse = " x "), "\n\n")

required_hierarchy_cols <- c("id", "genus", "family", "order")
missing_cols <- setdiff(required_hierarchy_cols, colnames(hierarchy_all))

if (length(missing_cols) > 0) {
  stop(
    "hierarchy_info is missing required columns: ",
    paste(missing_cols, collapse = ", ")
  )
}

# Hierarchy order: lower level -> higher level
hierarchy_all <- hierarchy_all[, c("id", "genus", "family", "order")]

X_all <- as.matrix(X_df)
storage.mode(X_all) <- "numeric"

if (nrow(X_all) != nrow(hierarchy_all)) {
  stop(
    "X and hierarchy_info have different numbers of rows: X = ",
    nrow(X_all),
    ", hierarchy = ",
    nrow(hierarchy_all)
  )
}

orig_nrow <- nrow(X_all)
orig_ncol <- ncol(X_all)
orig_colnames <- colnames(X_df)

cat("Original dim:", paste(dim(X_all), collapse = " x "), "\n\n")

# ---------- 2) Filter overly sparse rows and empty columns ----------
cat("Checking missingness...\n")

row_obs <- rowSums(!is.na(X_all))
col_obs <- colSums(!is.na(X_all))

cat("Rows with 0 obs:", sum(row_obs == 0), "\n")
cat("Rows with 1 obs:", sum(row_obs == 1), "\n")
cat("Rows with 2 obs:", sum(row_obs == 2), "\n")
cat("Rows with >=3 obs:", sum(row_obs >= 3), "\n")
cat("Columns with 0 obs:", sum(col_obs == 0), "\n")
cat("Columns with >0 obs:", sum(col_obs > 0), "\n\n")

keep_rows <- row_obs >= min_obs_per_row
keep_cols <- col_obs > 0

cat("Minimum observations per row:", min_obs_per_row, "\n")
cat("Keep rows:", sum(keep_rows), "/", length(keep_rows), "\n")
cat("Keep cols:", sum(keep_cols), "/", length(keep_cols), "\n\n")

if (sum(keep_rows) == 0) {
  stop("No rows left after filtering. Lower min_obs_per_row.")
}

if (sum(keep_cols) == 0) {
  stop("No columns left after filtering.")
}

X_run <- X_all[keep_rows, keep_cols, drop = FALSE]
hierarchy_run <- hierarchy_all[keep_rows, , drop = FALSE]

cat("Filtered dim for BHPMF:", paste(dim(X_run), collapse = " x "), "\n\n")

# ---------- 3) Re-check hierarchy conflicts ----------
cat("Checking hierarchy consistency...\n")

genus_to_family_n <- tapply(
  hierarchy_run$family,
  hierarchy_run$genus,
  function(x) length(unique(x))
)

family_to_order_n <- tapply(
  hierarchy_run$order,
  hierarchy_run$family,
  function(x) length(unique(x))
)

n_genus_conflicts <- sum(genus_to_family_n > 1, na.rm = TRUE)
n_family_conflicts <- sum(family_to_order_n > 1, na.rm = TRUE)

cat("Genus with >1 family:", n_genus_conflicts, "\n")
cat("Family with >1 order:", n_family_conflicts, "\n\n")

if (n_genus_conflicts > 0) {
  conflict_names <- names(genus_to_family_n)[genus_to_family_n > 1]
  cat("Example conflicting genera:\n")
  print(head(conflict_names, 20))
  stop("Hierarchy still contains genus -> family conflicts. Please clean the taxonomy first.")
}

if (n_family_conflicts > 0) {
  conflict_names <- names(family_to_order_n)[family_to_order_n > 1]
  cat("Example conflicting families:\n")
  print(head(conflict_names, 20))
  stop("Hierarchy still contains family -> order conflicts. Please clean the taxonomy first.")
}

# ---------- 4) Prepare tmp directory ----------
cat("Preparing temporary directory:", tmp.dir, "\n")

if (dir.exists(tmp.dir)) {
  unlink(tmp.dir, recursive = TRUE, force = TRUE)
}

dir.create(tmp.dir, recursive = TRUE, showWarnings = FALSE)

# ---------- 5) Run BHPMF ----------
cat("============================================================\n")
cat("Running GapFilling()\n")
cat("Parameters:\n")
cat("  prediction.level =", prediction.level, "\n")
cat("  used.num.hierarchy.levels =", used.num.hierarchy.levels, "\n")
cat("  num.samples =", num.samples, "\n")
cat("  burn =", burn, "\n")
cat("  gaps =", gaps, "\n")
cat("  num.latent.feats =", num.latent.feats, "\n")
cat("  tuning =", tuning, "\n")
cat("  num.folds.tuning =", num.folds.tuning, "\n")
cat("============================================================\n\n")

set.seed(123)

GapFilling(
  X = X_run,
  hierarchy.info = hierarchy_run,
  prediction.level = prediction.level,
  used.num.hierarchy.levels = used.num.hierarchy.levels,
  num.samples = num.samples,
  burn = burn,
  gaps = gaps,
  num.latent.feats = num.latent.feats,
  tuning = tuning,
  num.folds.tuning = num.folds.tuning,
  tmp.dir = tmp.dir,
  mean.gap.filled.output.path = mean_out,
  std.gap.filled.output.path = std_out,
  rmse.plot.test.data = rmse.plot.test.data,
  verbose = verbose
)

cat("\nGapFilling() finished.\n\n")

# ---------- 6) Read mean output ----------
if (!file.exists(mean_out)) {
  stop("BHPMF mean output file not found: ", mean_out)
}

cat("Reading mean output:", mean_out, "\n")

imputed_run_df <- read.table(
  mean_out,
  sep = "\t",
  header = TRUE,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

imputed_run <- as.matrix(imputed_run_df)
storage.mode(imputed_run) <- "numeric"

cat("BHPMF mean output dim:", paste(dim(imputed_run), collapse = " x "), "\n")

if (nrow(imputed_run) != nrow(X_run) || ncol(imputed_run) != ncol(X_run)) {
  stop(
    paste0(
      "Output dimension mismatch: imputed_run = ",
      paste(dim(imputed_run), collapse = "x"),
      ", X_run = ",
      paste(dim(X_run), collapse = "x")
    )
  )
}

# ---------- 7) Paste mean output back ----------
cat("Pasting mean output back to original matrix size...\n")

final_imputed <- X_all
final_imputed[keep_rows, keep_cols] <- imputed_run
colnames(final_imputed) <- orig_colnames

write.csv(
  final_imputed,
  final_imputed_file,
  row.names = FALSE
)

final_with_id <- data.frame(
  id = hierarchy_all$id,
  final_imputed,
  check.names = FALSE
)

write.csv(
  final_with_id,
  final_imputed_with_id_file,
  row.names = FALSE
)

cat("Saved:", final_imputed_file, "\n")
cat("Saved:", final_imputed_with_id_file, "\n\n")

# ---------- 8) Read std output ----------
if (file.exists(std_out)) {
  cat("Reading std output:", std_out, "\n")

  std_run_df <- read.table(
    std_out,
    sep = "\t",
    header = TRUE,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )

  std_run <- as.matrix(std_run_df)
  storage.mode(std_run) <- "numeric"

  cat("BHPMF std output dim:", paste(dim(std_run), collapse = " x "), "\n")

  if (nrow(std_run) != nrow(X_run) || ncol(std_run) != ncol(X_run)) {
    stop(
      paste0(
        "std output dimension mismatch: std_run = ",
        paste(dim(std_run), collapse = "x"),
        ", X_run = ",
        paste(dim(X_run), collapse = "x")
      )
    )
  }

  cat("Pasting std output back to original matrix size...\n")

  final_std <- matrix(
    NA_real_,
    nrow = orig_nrow,
    ncol = orig_ncol
  )

  final_std[keep_rows, keep_cols] <- std_run
  colnames(final_std) <- orig_colnames

  write.csv(
    final_std,
    final_std_file,
    row.names = FALSE
  )

  final_std_with_id <- data.frame(
    id = hierarchy_all$id,
    final_std,
    check.names = FALSE
  )

  write.csv(
    final_std_with_id,
    final_std_with_id_file,
    row.names = FALSE
  )

  cat("Saved:", final_std_file, "\n")
  cat("Saved:", final_std_with_id_file, "\n\n")
} else {
  cat("std output file not found, skipping std output:", std_out, "\n\n")
}

# ---------- 9) Final summary ----------
cat("============================================================\n")
cat("Done.\n")
cat("Time:", as.character(Sys.time()), "\n")
cat("Output files:\n")
cat("  ", final_imputed_file, "\n", sep = "")
cat("  ", final_imputed_with_id_file, "\n", sep = "")

if (file.exists(std_out)) {
  cat("  ", final_std_file, "\n", sep = "")
  cat("  ", final_std_with_id_file, "\n", sep = "")
}

cat("Session info:\n")
print(sessionInfo())
cat("============================================================\n")