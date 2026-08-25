"""
PyTorch GPU-accelerated BPMF / EmbeddingPriorBPMF / BHPMF
===========================================================
U, V stay on GPU. Only small (d,d) matrices go to CPU for Wishart sampling.

Modified: 9:1 internal train/validation split. Early stopping is driven by
validation-set RMSE instead of training-set RMSE, to avoid stopping on
training noise / overfitting to observed entries.
"""

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_random_state
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import time

from bhpmf import (
    sample_wishart, sample_normal, sample_gamma,
    _safe_inv, _safe_cho_inv,
    effective_sample_size,
)


# ============================================================
# Internal 9:1 validation split helper
# ============================================================

def _make_internal_val_split(observed, rng, val_frac=0.1):
    """
    Split the observed mask into (train_mask, val_mask), 9:1 by default.

    Parameters
    ----------
    observed : (N, M) bool ndarray
    rng      : numpy RandomState
    val_frac : fraction of *observed* entries held out for validation

    Returns
    -------
    train_mask, val_mask : (N, M) bool ndarrays, mutually exclusive,
                           with train_mask | val_mask == observed
    """
    idx = np.argwhere(observed)
    n_obs = idx.shape[0]
    n_val = max(1, int(round(val_frac * n_obs)))
    perm = rng.permutation(n_obs)
    val_sel = perm[:n_val]
    val_mask = np.zeros_like(observed)
    val_mask[idx[val_sel, 0], idx[val_sel, 1]] = True
    train_mask = observed & ~val_mask
    return train_mask, val_mask


# ============================================================
# Normal-Wishart / projection sampling (unchanged)
# ============================================================

def _sample_hyperparams_cpu(X_np, mu0, beta0, nu0, W0_inv, rng):
    """Normal-Wishart on CPU. X_np is (N, d) numpy."""
    N, d = X_np.shape
    X_bar = X_np.mean(axis=0)
    S = (X_np - X_bar).T @ (X_np - X_bar)
    beta_n = beta0 + N
    nu_n = nu0 + N
    mu_n = (beta0 * mu0 + N * X_bar) / beta_n
    W_n_inv = (W0_inv + S
               + (beta0 * N / beta_n) * np.outer(X_bar - mu0, X_bar - mu0))
    W_n = _safe_inv(W_n_inv)
    W_n = (W_n + W_n.T) / 2.0
    W_n += np.eye(d) * max(1e-10, np.abs(W_n).max() * 1e-12)
    Lambda = sample_wishart(nu_n, W_n, rng)
    mu_cov = _safe_cho_inv(beta_n * Lambda)
    mu = sample_normal(mu_n, mu_cov, rng)
    return mu, Lambda


def _sample_projection_cpu(X_np, E, Lambda_x, embed_prec, rng):
    """Sample W, b on CPU."""
    N, d = X_np.shape
    emb_dim = E.shape[1]
    E_aug = np.hstack([E, np.ones((N, 1))])
    p = emb_dim + 1
    prior_prec = embed_prec * np.eye(p)
    Sigma_post = _safe_cho_inv(E_aug.T @ E_aug + prior_prec)
    W_full = np.zeros((d, p))
    for k in range(d):
        mu_post = Sigma_post @ (E_aug.T @ X_np[:, k])
        W_full[k] = sample_normal(mu_post, Sigma_post, rng)
    return W_full[:, :emb_dim], W_full[:, emb_dim]


def _streaming_posterior_pred(U_samples, V_samples, return_std=False,
                              clip_range=None):
    """
    Welford online mean/variance over posterior predictive matrices.

    Avoids materializing the full (n_samples, N, M) tensor — memory is
    bounded to two (N, M) float64 accumulators regardless of n_samples.
    """
    n = len(U_samples)
    if n == 0:
        return None
    mean = (U_samples[0] @ V_samples[0].T).astype(np.float64, copy=True)
    if not return_std:
        for i in range(1, n):
            pred_i = U_samples[i] @ V_samples[i].T
            mean += (pred_i - mean) / (i + 1)
        if clip_range:
            mean = np.clip(mean, *clip_range)
        return mean
    M2 = np.zeros_like(mean)
    for i in range(1, n):
        pred_i = U_samples[i] @ V_samples[i].T
        delta = pred_i - mean
        mean += delta / (i + 1)
        delta2 = pred_i - mean
        M2 += delta * delta2
    var = M2 / (n - 1) if n > 1 else np.zeros_like(mean)
    std = np.sqrt(np.maximum(var, 0.0))
    if clip_range:
        mean = np.clip(mean, *clip_range)
    return mean, std


def _pick_samples(U_samples, V_samples, n_samples=None, sample_gap=1):
    """
    Select a subset of posterior samples for prediction.

    Parameters
    ----------
    U_samples, V_samples : lists of arrays (all collected samples)
    n_samples : int or None
        How many samples to use.  None → use all (original behaviour).
    sample_gap : int
        Step size when walking backwards from the last sample (default 1 = no gap).

    Returns
    -------
    U_sel, V_sel : lists of arrays
    """
    n_total = len(U_samples)
    if n_total == 0:
        return U_samples, V_samples

    if n_samples is None and sample_gap <= 1:
        # original mode: use everything
        return U_samples, V_samples

    gap = max(1, int(sample_gap))
    # indices from back to front with gap
    indices = list(range(n_total - 1, -1, -gap))

    if n_samples is not None:
        indices = indices[:int(n_samples)]

    # chronological order
    indices = sorted(indices)
    return [U_samples[i] for i in indices], [V_samples[i] for i in indices]


def test_sample_sweep(model, R_safe=None, val_mask=None,
                      sample_gap=10, n_min=5, n_max=30):
    """
    Sweep n_samples from n_min to n_max (with given gap) and report val RMSE
    for each configuration.  Useful for choosing the best prediction setting.

    Parameters
    ----------
    model : fitted BPMF_GPU / EmbeddingPriorBPMF_GPU / BHPMF_GPU
    R_safe : (N, M) array with NaN filled as 0.  If None, reconstructed from
             model.train_mask_ + model.val_mask_ + last U_/V_.
    val_mask : (N, M) bool array.  If None, uses model.val_mask_.
    sample_gap : int, default 10
    n_min, n_max : int, range of sample counts to test

    Returns
    -------
    results : list of dict with keys 'n_samples', 'actual_n', 'val_rmse'
    """
    if val_mask is None:
        val_mask = model.val_mask_
    n_val = int(val_mask.sum())
    if n_val == 0:
        raise ValueError("No validation entries to evaluate.")

    # Reconstruct R_safe if not provided
    if R_safe is None:
        # Use the last point estimate to get R values on observed entries
        # (works because R_safe = R with NaN→0, and we only measure on val_mask
        #  which are observed entries where R_safe == original R)
        # We need the original R values on val positions.  They were stored in
        # train_gpu / val_gpu during fit, but R_safe is not saved.  We can
        # recover them from pred + residual at the last sample, but the
        # simplest is to require the user to pass R_safe.
        raise ValueError(
            "Pass R_safe (the NaN→0 filled rating matrix) for val RMSE "
            "computation.  Example: R_safe = np.where(~np.isnan(R), R, 0.0)")

    n_total = len(model.U_samples_)
    print(f"\n{'='*60}")
    print(f"  Sample sweep: gap={sample_gap}, "
          f"n_samples={n_min}..{n_max}, "
          f"total collected={n_total}")
    print(f"{'='*60}")
    print(f"  {'n_samples':>10}  {'actual':>6}  {'val_RMSE':>10}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*10}")

    results = []
    for n in range(n_min, n_max + 1):
        U_sel, V_sel = _pick_samples(
            model.U_samples_, model.V_samples_,
            n_samples=n, sample_gap=sample_gap)
        actual_n = len(U_sel)
        if actual_n == 0:
            continue
        pred_avg = _streaming_posterior_pred(U_sel, V_sel, return_std=False)
        val_err = ((R_safe - pred_avg) ** 2) * val_mask
        val_rmse = float(np.sqrt(val_err.sum() / n_val))
        results.append({'n_samples': n, 'actual_n': actual_n,
                        'val_rmse': val_rmse})
        print(f"  {n:>10}  {actual_n:>6}  {val_rmse:>10.4f}")

    # highlight best
    if results:
        best = min(results, key=lambda x: x['val_rmse'])
        print(f"  {'-'*10}  {'-'*6}  {'-'*10}")
        print(f"  Best: n_samples={best['n_samples']} "
              f"(actual={best['actual_n']}) → val_RMSE={best['val_rmse']:.4f}")
    print(f"{'='*60}\n")

    return results


# ============================================================
# GPU factor sampling — U,V stay as tensors
# ============================================================

def _sample_factors_on_gpu(V_gpu, R_gpu, obs_gpu, mu_prior_gpu,
                            Lambda_gpu, alpha, device, CHUNK=20000):
    """
    Sample all latent factors on GPU.

    obs_gpu is the *training* mask (val entries are 0 here), so held-out
    entries do not contribute to the posterior.
    """
    N = R_gpu.shape[0]
    d = V_gpu.shape[1]
    M = V_gpu.shape[0]
    per_entity = (mu_prior_gpu.dim() == 2)

    VVt_flat = (V_gpu.unsqueeze(2) * V_gpu.unsqueeze(1)).reshape(M, d * d)
    data_terms = alpha * ((R_gpu * obs_gpu) @ V_gpu)

    if per_entity:
        prior_terms = mu_prior_gpu @ Lambda_gpu.T
    else:
        prior_term = Lambda_gpu @ mu_prior_gpu

    eye_d = torch.eye(d, device=device, dtype=torch.float64)
    X_new = torch.empty(N, d, device=device, dtype=torch.float64)

    for s in range(0, N, CHUNK):
        e = min(s + CHUNK, N)
        cs = e - s

        sum_VVt = (obs_gpu[s:e] @ VVt_flat).reshape(cs, d, d)
        Lam = Lambda_gpu.unsqueeze(0) + alpha * sum_VVt
        Lam = (Lam + Lam.transpose(1, 2)) * 0.5 + eye_d * 1e-10

        try:
            L = torch.linalg.cholesky(Lam)
        except Exception:
            Lam = Lam + eye_d * 1e-6
            L = torch.linalg.cholesky(Lam)

        I_batch = eye_d.unsqueeze(0).expand(cs, -1, -1)
        L_inv = torch.linalg.solve_triangular(L, I_batch, upper=False)
        Sigma = torch.bmm(L_inv.transpose(1, 2), L_inv)

        if per_entity:
            rhs = prior_terms[s:e] + data_terms[s:e]
        else:
            rhs = data_terms[s:e] + prior_term.unsqueeze(0)

        mu_star = torch.bmm(Sigma, rhs.unsqueeze(2)).squeeze(2)

        L_sigma = L_inv.transpose(1, 2)
        Z = torch.randn(cs, d, device=device, dtype=torch.float64)
        X_new[s:e] = mu_star + torch.bmm(L_sigma, Z.unsqueeze(2)).squeeze(2)

    return X_new


# ============================================================
# BPMF_GPU
# ============================================================

class BPMF_GPU(BaseEstimator, RegressorMixin):
    """GPU-accelerated BPMF with 9:1 internal validation-driven early stopping."""

    def __init__(self, n_latent=15, n_iterations=200, burn_in=100,
                 alpha_0=1.0, beta_alpha_0=1.0,
                 early_stop_patience=None, early_stop_min_samples=30,
                 val_frac=0.1, val_warmup=None,
                 max_samples=None,
                 clip_range=None, thin=1,
                 device='cuda', random_state=42, verbose=False):
        self.n_latent = n_latent
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.alpha_0 = alpha_0
        self.beta_alpha_0 = beta_alpha_0
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_samples = early_stop_min_samples
        self.val_frac = val_frac
        self.val_warmup = val_warmup
        self.max_samples = max_samples
        self.clip_range = clip_range
        self.thin = thin
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, R, y=None):
        rng = check_random_state(self.random_state)
        R = np.array(R, dtype=np.float64)
        n_rows, n_cols = R.shape
        d = self.n_latent
        dev = torch.device(self.device if torch.cuda.is_available() else 'cpu')

        if self.verbose:
            print(f"  Device: {dev}")

        observed = ~np.isnan(R)
        R_safe = np.where(observed, R, 0.0)

        # --- 9:1 internal split ---
        train_mask, val_mask = _make_internal_val_split(
            observed, rng, val_frac=self.val_frac)
        n_train = int(train_mask.sum())
        n_val = int(val_mask.sum())
        if self.verbose:
            print(f"  Internal split: {n_train} train / {n_val} val "
                  f"({self.val_frac:.0%} held out)")

        R_gpu = torch.tensor(R_safe, dtype=torch.float64, device=dev)
        train_gpu = torch.tensor(train_mask, dtype=torch.float64, device=dev)
        val_gpu = torch.tensor(val_mask, dtype=torch.float64, device=dev)
        R_gpu_T = R_gpu.T.contiguous()
        train_gpu_T = train_gpu.T.contiguous()

        mu0 = np.zeros(d)
        beta0 = 2.0
        nu0 = d
        W0_inv = np.eye(d)
        a0 = self.alpha_0
        b0 = self.beta_alpha_0

        U_gpu = torch.randn(n_rows, d, device=dev, dtype=torch.float64) * 0.1
        V_gpu = torch.randn(n_cols, d, device=dev, dtype=torch.float64) * 0.1
        alpha = 2.0

        self.U_samples_ = []
        self.V_samples_ = []
        self.alpha_samples_ = []
        self.train_rmse_ = []
        self.val_rmse_ = []

        best_val_rmse = float('inf')
        best_U_gpu = U_gpu.clone()
        best_V_gpu = V_gpu.clone()
        best_alpha = alpha

        t0 = time.time()

        for it in range(self.n_iterations):
            U_np = U_gpu.cpu().numpy()
            V_np = V_gpu.cpu().numpy()

            # alpha from TRAIN residuals only
            pred = U_np @ V_np.T
            res_sq = ((R_safe - pred) ** 2) * train_mask
            alpha = sample_gamma(a0 + n_train / 2.0,
                                 b0 + 0.5 * res_sq.sum(), rng)

            mu_U, Lambda_U = _sample_hyperparams_cpu(
                U_np, mu0, beta0, nu0, W0_inv, rng)
            mu_V, Lambda_V = _sample_hyperparams_cpu(
                V_np, mu0, beta0, nu0, W0_inv, rng)

            mu_U_gpu = torch.tensor(mu_U, dtype=torch.float64, device=dev)
            Lambda_U_gpu = torch.tensor(Lambda_U, dtype=torch.float64, device=dev)
            mu_V_gpu = torch.tensor(mu_V, dtype=torch.float64, device=dev)
            Lambda_V_gpu = torch.tensor(Lambda_V, dtype=torch.float64, device=dev)

            # factor updates use TRAIN mask so val entries are unseen
            U_gpu = _sample_factors_on_gpu(
                V_gpu, R_gpu, train_gpu, mu_U_gpu, Lambda_U_gpu, alpha, dev)
            V_gpu = _sample_factors_on_gpu(
                U_gpu, R_gpu_T, train_gpu_T, mu_V_gpu, Lambda_V_gpu, alpha, dev)

            pred_gpu = U_gpu @ V_gpu.T
            err_gpu = R_gpu - pred_gpu
            train_rmse = torch.sqrt(((err_gpu ** 2) * train_gpu).sum() / n_train).item()
            val_rmse = torch.sqrt(((err_gpu ** 2) * val_gpu).sum() / n_val).item()
            self.train_rmse_.append(train_rmse)
            self.val_rmse_.append(val_rmse)

            # val_warmup: iterations before which val RMSE is unreliable
            # (U/V not yet converged → spurious low val points). These are
            # excluded from both best-tracking and early-stop prior-min.
            warmup = (self.val_warmup if self.val_warmup is not None
                      else self.burn_in)

            # Track global best (for restoring final state), post-warmup only
            if it >= warmup and val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_U_gpu = U_gpu.clone()
                best_V_gpu = V_gpu.clone()
                best_alpha = alpha

            # Sample collection (post burn-in). Ring buffer if max_samples set.
            if it >= self.burn_in and (it - self.burn_in) % self.thin == 0:
                self.U_samples_.append(U_gpu.cpu().numpy().copy())
                self.V_samples_.append(V_gpu.cpu().numpy().copy())
                self.alpha_samples_.append(alpha)
                if self.max_samples is not None:
                    while len(self.U_samples_) > self.max_samples:
                        self.U_samples_.pop(0)
                        self.V_samples_.pop(0)
                        self.alpha_samples_.pop(0)

            # Early stop: no improvement within the last `patience` iters
            # (window-based; prior window starts at `warmup` to ignore
            # spurious early-iter val lows when U/V haven't converged)
            should_stop = False
            if self.early_stop_patience is not None and it > self.burn_in:
                p = self.early_stop_patience
                post_warmup = self.val_rmse_[warmup:]
                if (len(post_warmup) > p and
                        len(self.U_samples_) >= self.early_stop_min_samples):
                    recent = post_warmup[-p:]
                    older = post_warmup[:-p]
                    if len(older) > 0 and min(recent) >= min(older):
                        U_gpu = best_U_gpu.clone()
                        V_gpu = best_V_gpu.clone()
                        alpha = best_alpha
                        should_stop = True
                        if self.verbose:
                            print(f"  *** Early stop at iter {it} "
                                  f"(no val improvement in last {p} iters "
                                  f"after warmup={warmup}; "
                                  f"window min={min(recent):.4f}, "
                                  f"prior min={min(older):.4f}, "
                                  f"global best={best_val_rmse:.4f}, "
                                  f"{len(self.U_samples_)} samples) ***")

            if self.verbose and (it + 1) % 10 == 0:
                elapsed = time.time() - t0
                speed = (it + 1) / elapsed
                eta = (self.n_iterations - it - 1) / speed
                n_samp = len(self.U_samples_)
                print(f"  Iter {it+1}/{self.n_iterations}  "
                      f"train={train_rmse:.4f}  val={val_rmse:.4f}  "
                      f"alpha={alpha:.2f}  "
                      f"[{speed:.1f} it/s, ETA {eta:.0f}s, {n_samp} samples]")

            if should_stop:
                break

        self.U_ = U_gpu.cpu().numpy()
        self.V_ = V_gpu.cpu().numpy()
        self.alpha_ = alpha
        self.shape_ = (n_rows, n_cols)
        self.train_mask_ = train_mask
        self.val_mask_ = val_mask
        self.best_val_rmse_ = best_val_rmse
        return self

    def predict(self, return_std=False, n_samples=None, sample_gap=1):
        """
        Posterior predictive mean (and optionally std).

        Parameters
        ----------
        n_samples : int or None
            None → use all collected samples (original behaviour).
            int  → use this many samples, picked from the end.
        sample_gap : int
            Step size when walking backwards (default 1 = consecutive).
            E.g. sample_gap=10 picks every 10th sample from the tail.
        """
        if len(self.U_samples_) == 0:
            warnings.warn("No posterior samples. Using last sample.")
            pred = self.U_ @ self.V_.T
            if self.clip_range:
                pred = np.clip(pred, *self.clip_range)
            return pred
        U_sel, V_sel = _pick_samples(
            self.U_samples_, self.V_samples_,
            n_samples=n_samples, sample_gap=sample_gap)
        result = _streaming_posterior_pred(
            U_sel, V_sel,
            return_std=return_std, clip_range=self.clip_range)
        return result

    def score(self, R, y=None):
        R = np.array(R, dtype=np.float64)
        observed = ~np.isnan(R)
        pred = self.predict()
        return -np.sqrt(np.mean((R[observed] - pred[observed]) ** 2))

    def diagnostics(self):
        diag = {
            'alpha_trace': np.array(self.alpha_samples_),
            'rmse_trace': np.array(self.train_rmse_),
            'val_rmse_trace': np.array(self.val_rmse_),
        }
        if len(self.alpha_samples_) > 3:
            diag['alpha_ess'] = effective_sample_size(np.array(self.alpha_samples_))
        if len(self.U_samples_) > 0:
            diag['u_mean_trace'] = np.array([u.mean() for u in self.U_samples_])
        return diag


# ============================================================
# EmbeddingPriorBPMF_GPU
# ============================================================

class EmbeddingPriorBPMF_GPU(BaseEstimator, RegressorMixin):
    """GPU EmbeddingPriorBPMF with val-driven annealing, W-freeze, early stop."""

    def __init__(self, n_latent=15, n_iterations=200, burn_in=100,
                 alpha_0=1.0, beta_alpha_0=1.0,
                 embed_precision=1.0, learn_projection=True,
                 w_freeze_after=None, w_freeze_patience=None,
                 precision_anneal=True,
                 early_stop_patience=None, early_stop_min_samples=30,
                 val_frac=0.1, val_warmup=None,
                 max_samples=None,
                 clip_range=None, thin=1,
                 device='cuda', random_state=42, verbose=False):
        self.n_latent = n_latent
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.alpha_0 = alpha_0
        self.beta_alpha_0 = beta_alpha_0
        self.embed_precision = embed_precision
        self.learn_projection = learn_projection
        self.w_freeze_after = w_freeze_after
        self.w_freeze_patience = w_freeze_patience
        self.precision_anneal = precision_anneal
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_samples = early_stop_min_samples
        self.val_frac = val_frac
        self.val_warmup = val_warmup
        self.max_samples = max_samples
        self.clip_range = clip_range
        self.thin = thin
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, R, y=None, row_embeddings=None, col_embeddings=None):
        rng = check_random_state(self.random_state)
        R = np.array(R, dtype=np.float64)
        n_rows, n_cols = R.shape
        d = self.n_latent
        dev = torch.device(self.device if torch.cuda.is_available() else 'cpu')

        if self.verbose:
            print(f"  Device: {dev}")

        observed = ~np.isnan(R)
        R_safe = np.where(observed, R, 0.0)

        # --- 9:1 internal split ---
        train_mask, val_mask = _make_internal_val_split(
            observed, rng, val_frac=self.val_frac)
        n_train = int(train_mask.sum())
        n_val = int(val_mask.sum())
        if self.verbose:
            print(f"  Internal split: {n_train} train / {n_val} val "
                  f"({self.val_frac:.0%} held out)")

        R_gpu = torch.tensor(R_safe, dtype=torch.float64, device=dev)
        train_gpu = torch.tensor(train_mask, dtype=torch.float64, device=dev)
        val_gpu = torch.tensor(val_mask, dtype=torch.float64, device=dev)
        R_gpu_T = R_gpu.T.contiguous()
        train_gpu_T = train_gpu.T.contiguous()

        E_row = self._prepare_embeddings(row_embeddings, n_rows, d, "row")

        mu0 = np.zeros(d)
        beta0 = 2.0
        nu0 = d
        W0_inv = np.eye(d)
        a0 = self.alpha_0
        b0 = self.beta_alpha_0
        base_prec = self.embed_precision

        if E_row is not None:
            emb_dim = E_row.shape[1]
            W_row = rng.randn(d, emb_dim) * 0.01
            b_row = np.zeros(d)
            mu_U_per_np = (W_row @ E_row.T).T + b_row
        else:
            mu_U_per_np = None

        U_gpu = torch.randn(n_rows, d, device=dev, dtype=torch.float64) * 0.1
        V_gpu = torch.randn(n_cols, d, device=dev, dtype=torch.float64) * 0.1
        alpha = 2.0

        self.U_samples_ = []
        self.V_samples_ = []
        self.alpha_samples_ = []
        self.train_rmse_ = []
        self.val_rmse_ = []

        w_freeze = self.w_freeze_after
        do_anneal = self.precision_anneal and self.learn_projection
        best_val_rmse = float('inf')
        best_W = W_row.copy() if E_row is not None else None
        best_b = b_row.copy() if E_row is not None else None
        best_U_gpu = U_gpu.clone()
        best_V_gpu = V_gpu.clone()
        best_alpha = alpha

        t0 = time.time()

        for it in range(self.n_iterations):
            U_np = U_gpu.cpu().numpy()
            V_np = V_gpu.cpu().numpy()

            # 1. Alpha on TRAIN
            pred_np = U_np @ V_np.T
            res_sq = ((R_safe - pred_np) ** 2) * train_mask
            alpha = sample_gamma(a0 + n_train / 2.0,
                                 b0 + 0.5 * res_sq.sum(), rng)

            # 2-3. Hyperparams (CPU)
            mu_U_global, Lambda_U = _sample_hyperparams_cpu(
                U_np, mu0, beta0, nu0, W0_inv, rng)
            mu_V, Lambda_V = _sample_hyperparams_cpu(
                V_np, mu0, beta0, nu0, W0_inv, rng)

            # 3.5. W update
            w_is_frozen = w_freeze is not None and it >= w_freeze
            if do_anneal and not w_is_frozen:
                current_prec = base_prec * (1.0 + 9.0 * (it / self.n_iterations))
            else:
                current_prec = base_prec

            if E_row is not None and self.learn_projection and not w_is_frozen:
                W_row, b_row = _sample_projection_cpu(
                    U_np, E_row, Lambda_U, current_prec, rng)
                mu_U_per_np = (W_row @ E_row.T).T + b_row

            Lambda_U_gpu = torch.tensor(Lambda_U, dtype=torch.float64, device=dev)
            Lambda_V_gpu = torch.tensor(Lambda_V, dtype=torch.float64, device=dev)

            # 4. Sample U on TRAIN mask
            if mu_U_per_np is not None:
                mu_gpu = torch.tensor(mu_U_per_np, dtype=torch.float64, device=dev)
            else:
                mu_gpu = torch.tensor(mu_U_global, dtype=torch.float64, device=dev)

            U_gpu = _sample_factors_on_gpu(
                V_gpu, R_gpu, train_gpu, mu_gpu, Lambda_U_gpu, alpha, dev)

            # 5. Sample V on TRAIN mask
            mu_V_gpu = torch.tensor(mu_V, dtype=torch.float64, device=dev)
            V_gpu = _sample_factors_on_gpu(
                U_gpu, R_gpu_T, train_gpu_T, mu_V_gpu, Lambda_V_gpu, alpha, dev)

            # RMSE on GPU
            pred_gpu = U_gpu @ V_gpu.T
            err_gpu = R_gpu - pred_gpu
            train_rmse = torch.sqrt(((err_gpu ** 2) * train_gpu).sum() / n_train).item()
            val_rmse = torch.sqrt(((err_gpu ** 2) * val_gpu).sum() / n_val).item()
            self.train_rmse_.append(train_rmse)
            self.val_rmse_.append(val_rmse)

            # val_warmup: iterations before which val RMSE is unreliable
            warmup = (self.val_warmup if self.val_warmup is not None
                      else self.burn_in)

            # --- Track global best by VAL rmse (post-warmup) ---
            if it >= warmup and val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_U_gpu = U_gpu.clone()
                best_V_gpu = V_gpu.clone()
                best_alpha = alpha
                if E_row is not None:
                    best_W = W_row.copy()
                    best_b = b_row.copy()

            # --- W auto-freeze: no val improvement in a recent window ---
            # w_freeze_patience semantics:
            #   None    → auto (use early_stop_patience // 2, or 20 if also None)
            #   int>0   → use this window length
            #   0/False → disable adaptive freeze entirely
            adaptive_freeze_on = (self.w_freeze_patience is None
                                  or self.w_freeze_patience)
            if (not w_is_frozen and self.learn_projection
                    and adaptive_freeze_on and it > max(30, warmup)):
                if self.w_freeze_patience:
                    w_freeze_win = self.w_freeze_patience
                elif self.early_stop_patience is not None:
                    w_freeze_win = max(10, self.early_stop_patience // 2)
                else:
                    w_freeze_win = 20
                post_warmup = self.val_rmse_[warmup:]
                if w_freeze is None and len(post_warmup) > w_freeze_win:
                    recent = post_warmup[-w_freeze_win:]
                    older = post_warmup[:-w_freeze_win]
                    if len(older) > 0 and min(recent) >= min(older):
                        w_freeze = it
                        if E_row is not None:
                            W_row, b_row = best_W.copy(), best_b.copy()
                            mu_U_per_np = (W_row @ E_row.T).T + b_row
                        if self.verbose:
                            print(f"  *** W frozen at iter {it} "
                                  f"(window={w_freeze_win}, "
                                  f"best VAL RMSE={best_val_rmse:.4f}) ***")

            # --- Sample collection (post burn-in; ring buffer if max_samples). ---
            if it >= self.burn_in and (it - self.burn_in) % self.thin == 0:
                self.U_samples_.append(U_gpu.cpu().numpy().copy())
                self.V_samples_.append(V_gpu.cpu().numpy().copy())
                self.alpha_samples_.append(alpha)
                if self.max_samples is not None:
                    while len(self.U_samples_) > self.max_samples:
                        self.U_samples_.pop(0)
                        self.V_samples_.pop(0)
                        self.alpha_samples_.pop(0)

            # --- Early stop: post-warmup window-based ---
            es_patience = self.early_stop_patience
            es_min_samples = self.early_stop_min_samples
            should_stop = False

            if es_patience is not None and it > self.burn_in:
                post_warmup = self.val_rmse_[warmup:]
                if (len(post_warmup) > es_patience and
                        len(self.U_samples_) >= es_min_samples):
                    recent = post_warmup[-es_patience:]
                    older = post_warmup[:-es_patience]
                    if len(older) > 0 and min(recent) >= min(older):
                        U_gpu = best_U_gpu.clone()
                        V_gpu = best_V_gpu.clone()
                        alpha = best_alpha
                        if E_row is not None and best_W is not None:
                            W_row, b_row = best_W, best_b
                        should_stop = True
                        if self.verbose:
                            print(f"  *** Early stop at iter {it} "
                                  f"(no val improvement in last {es_patience} iters "
                                  f"after warmup={warmup}; "
                                  f"window min={min(recent):.4f}, "
                                  f"prior min={min(older):.4f}, "
                                  f"global best={best_val_rmse:.4f}, "
                                  f"{len(self.U_samples_)} samples) ***")

            # --- Logging ---
            if self.verbose and (it + 1) % 10 == 0:
                elapsed = time.time() - t0
                speed = (it + 1) / elapsed
                eta = (self.n_iterations - it - 1) / speed
                frozen_str = " [W frozen]" if (w_freeze is not None and it >= w_freeze) else ""
                prec_str = f"  prec={current_prec:.1f}" if do_anneal and not w_is_frozen else ""
                n_samp = len(self.U_samples_)
                print(f"  Iter {it+1}/{self.n_iterations}  "
                      f"train={train_rmse:.4f}  val={val_rmse:.4f}  "
                      f"alpha={alpha:.2f}"
                      f"{prec_str}{frozen_str}  "
                      f"[{speed:.1f} it/s, ETA {eta:.0f}s, {n_samp} samples]")

            if should_stop:
                break

        self.U_ = U_gpu.cpu().numpy()
        self.V_ = V_gpu.cpu().numpy()
        self.alpha_ = alpha
        self.shape_ = (n_rows, n_cols)
        self.train_mask_ = train_mask
        self.val_mask_ = val_mask
        self.best_val_rmse_ = best_val_rmse
        if E_row is not None:
            self.W_row_ = W_row
            self.b_row_ = b_row
        return self

    def _prepare_embeddings(self, E, n_entities, d, name):
        if E is None:
            return None
        E = np.array(E, dtype=np.float64)
        assert E.shape[0] == n_entities
        E = StandardScaler().fit_transform(E)
        if E.shape[1] > 3 * d:
            target = min(2 * d, E.shape[1])
            if self.verbose:
                print(f"  {name} embedding: {E.shape[1]}d → PCA to {target}d")
            E = PCA(n_components=target).fit_transform(E)
        if self.verbose:
            print(f"  {name} embedding: {E.shape}")
        return E

    def predict(self, return_std=False, n_samples=None, sample_gap=1):
        """
        Posterior predictive mean (and optionally std).

        Parameters
        ----------
        n_samples : int or None
            None → use all collected samples (original behaviour).
            int  → use this many samples, picked from the end.
        sample_gap : int
            Step size when walking backwards (default 1 = consecutive).
        """
        if len(self.U_samples_) == 0:
            warnings.warn("No posterior samples. Using last sample.")
            pred = self.U_ @ self.V_.T
            if self.clip_range:
                pred = np.clip(pred, *self.clip_range)
            return pred
        U_sel, V_sel = _pick_samples(
            self.U_samples_, self.V_samples_,
            n_samples=n_samples, sample_gap=sample_gap)
        result = _streaming_posterior_pred(
            U_sel, V_sel,
            return_std=return_std, clip_range=self.clip_range)
        return result

    def score(self, R, y=None):
        R = np.array(R, dtype=np.float64)
        observed = ~np.isnan(R)
        pred = self.predict()
        return -np.sqrt(np.mean((R[observed] - pred[observed]) ** 2))

    def diagnostics(self):
        diag = {
            'alpha_trace': np.array(self.alpha_samples_),
            'rmse_trace': np.array(self.train_rmse_),
            'val_rmse_trace': np.array(self.val_rmse_),
        }
        if len(self.alpha_samples_) > 3:
            diag['alpha_ess'] = effective_sample_size(np.array(self.alpha_samples_))
        if len(self.U_samples_) > 0:
            diag['u_mean_trace'] = np.array([u.mean() for u in self.U_samples_])
        return diag


# ============================================================
# BHPMF_GPU
# ============================================================

class BHPMF_GPU(BaseEstimator, RegressorMixin):
    """GPU BHPMF with 9:1 internal validation-driven early stopping."""

    def __init__(self, n_latent=15, n_iterations=200, burn_in=100,
                 alpha_0=1.0, beta_alpha_0=1.0, mu_0=0.0, beta_0=2.0,
                 nu_0=None, W_0=None,
                 early_stop_patience=None, early_stop_min_samples=30,
                 val_frac=0.1, val_warmup=None,
                 max_samples=None,
                 clip_range=None, thin=1,
                 device='cuda', random_state=42, verbose=False):
        self.n_latent = n_latent
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.alpha_0 = alpha_0
        self.beta_alpha_0 = beta_alpha_0
        self.mu_0 = mu_0
        self.beta_0 = beta_0
        self.nu_0 = nu_0
        self.W_0 = W_0
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_samples = early_stop_min_samples
        self.val_frac = val_frac
        self.val_warmup = val_warmup
        self.max_samples = max_samples
        self.clip_range = clip_range
        self.thin = thin
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, R, y=None, hierarchy_rows=None, hierarchy_cols=None):
        rng = check_random_state(self.random_state)
        R = np.array(R, dtype=np.float64)
        n_rows, n_cols = R.shape
        d = self.n_latent
        dev = torch.device(self.device if torch.cuda.is_available() else 'cpu')

        if self.verbose:
            print(f"  Device: {dev}")

        observed = ~np.isnan(R)
        R_safe = np.where(observed, R, 0.0)

        # --- 9:1 internal split ---
        train_mask, val_mask = _make_internal_val_split(
            observed, rng, val_frac=self.val_frac)
        n_train = int(train_mask.sum())
        n_val = int(val_mask.sum())
        if self.verbose:
            print(f"  Internal split: {n_train} train / {n_val} val "
                  f"({self.val_frac:.0%} held out)")

        row_levels = self._build_hierarchy(hierarchy_rows, n_rows)
        col_levels = self._build_hierarchy(hierarchy_cols, n_cols)

        row_level_indices = []
        for groups, n_groups in row_levels:
            indices = [np.where(groups == g)[0] for g in range(n_groups)]
            row_level_indices.append(indices)

        col_level_indices = []
        for groups, n_groups in col_levels:
            indices = [np.where(groups == g)[0] for g in range(n_groups)]
            col_level_indices.append(indices)

        R_gpu = torch.tensor(R_safe, dtype=torch.float64, device=dev)
        train_gpu = torch.tensor(train_mask, dtype=torch.float64, device=dev)
        val_gpu = torch.tensor(val_mask, dtype=torch.float64, device=dev)
        R_gpu_T = R_gpu.T.contiguous()
        train_gpu_T = train_gpu.T.contiguous()

        mu0 = np.full(d, self.mu_0)
        beta0 = self.beta_0
        nu0 = self.nu_0 if self.nu_0 is not None else d
        W0 = self.W_0 if self.W_0 is not None else np.eye(d)
        W0_inv = _safe_inv(W0)
        a0 = self.alpha_0
        b0 = self.beta_alpha_0

        U_gpu = torch.randn(n_rows, d, device=dev, dtype=torch.float64) * 0.1
        V_gpu = torch.randn(n_cols, d, device=dev, dtype=torch.float64) * 0.1
        alpha = 2.0

        self.U_samples_ = []
        self.V_samples_ = []
        self.alpha_samples_ = []
        self.train_rmse_ = []
        self.val_rmse_ = []

        best_val_rmse = float('inf')
        best_U_gpu = U_gpu.clone()
        best_V_gpu = V_gpu.clone()
        best_alpha = alpha

        t0 = time.time()

        for it in range(self.n_iterations):
            U_np = U_gpu.cpu().numpy()
            V_np = V_gpu.cpu().numpy()

            # alpha on TRAIN
            pred_np = U_np @ V_np.T
            res_sq = ((R_safe - pred_np) ** 2) * train_mask
            alpha = sample_gamma(a0 + n_train / 2.0,
                                 b0 + 0.5 * res_sq.sum(), rng)

            # Hierarchical row hyperparams
            mu_U_parent, Lambda_U_parent = None, None
            for lev_idx, ((_groups, n_groups), indices) in enumerate(
                    zip(row_levels, row_level_indices)):
                U_lev = np.zeros((n_groups, d))
                for g in range(n_groups):
                    if len(indices[g]) > 0:
                        U_lev[g] = U_np[indices[g]].mean(axis=0)

                if mu_U_parent is None:
                    mu_U_lev, Lambda_U_lev = _sample_hyperparams_cpu(
                        U_lev, mu0, beta0, nu0, W0_inv, rng)
                else:
                    mu_U_lev, Lambda_U_lev = self._sample_hyperparams_with_prior(
                        U_lev, mu_U_parent, Lambda_U_parent, beta0, rng)
                mu_U_parent = mu_U_lev
                Lambda_U_parent = Lambda_U_lev

            if mu_U_parent is not None:
                mu_U, Lambda_U = self._sample_hyperparams_with_prior(
                    U_np, mu_U_parent, Lambda_U_parent, beta0, rng)
            else:
                mu_U, Lambda_U = _sample_hyperparams_cpu(
                    U_np, mu0, beta0, nu0, W0_inv, rng)

            # Hierarchical col hyperparams
            mu_V_parent, Lambda_V_parent = None, None
            for lev_idx, ((_groups, n_groups), indices) in enumerate(
                    zip(col_levels, col_level_indices)):
                V_lev = np.zeros((n_groups, d))
                for g in range(n_groups):
                    if len(indices[g]) > 0:
                        V_lev[g] = V_np[indices[g]].mean(axis=0)

                if mu_V_parent is None:
                    mu_V_lev, Lambda_V_lev = _sample_hyperparams_cpu(
                        V_lev, mu0, beta0, nu0, W0_inv, rng)
                else:
                    mu_V_lev, Lambda_V_lev = self._sample_hyperparams_with_prior(
                        V_lev, mu_V_parent, Lambda_V_parent, beta0, rng)
                mu_V_parent = mu_V_lev
                Lambda_V_parent = Lambda_V_lev

            if mu_V_parent is not None:
                mu_V, Lambda_V = self._sample_hyperparams_with_prior(
                    V_np, mu_V_parent, Lambda_V_parent, beta0, rng)
            else:
                mu_V, Lambda_V = _sample_hyperparams_cpu(
                    V_np, mu0, beta0, nu0, W0_inv, rng)

            # Sample U on TRAIN
            mu_U_gpu = torch.tensor(mu_U, dtype=torch.float64, device=dev)
            Lambda_U_gpu = torch.tensor(Lambda_U, dtype=torch.float64, device=dev)
            U_gpu = _sample_factors_on_gpu(
                V_gpu, R_gpu, train_gpu, mu_U_gpu, Lambda_U_gpu, alpha, dev)

            # Sample V on TRAIN
            mu_V_gpu = torch.tensor(mu_V, dtype=torch.float64, device=dev)
            Lambda_V_gpu = torch.tensor(Lambda_V, dtype=torch.float64, device=dev)
            V_gpu = _sample_factors_on_gpu(
                U_gpu, R_gpu_T, train_gpu_T, mu_V_gpu, Lambda_V_gpu, alpha, dev)

            # RMSE
            pred_gpu = U_gpu @ V_gpu.T
            err_gpu = R_gpu - pred_gpu
            train_rmse = torch.sqrt(((err_gpu ** 2) * train_gpu).sum() / n_train).item()
            val_rmse = torch.sqrt(((err_gpu ** 2) * val_gpu).sum() / n_val).item()
            self.train_rmse_.append(train_rmse)
            self.val_rmse_.append(val_rmse)

            # val_warmup: iterations before which val RMSE is unreliable
            warmup = (self.val_warmup if self.val_warmup is not None
                      else self.burn_in)

            # Track global best by VAL (post-warmup)
            if it >= warmup and val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_U_gpu = U_gpu.clone()
                best_V_gpu = V_gpu.clone()
                best_alpha = alpha

            # Sample collection (post burn-in). Ring buffer if max_samples set.
            if it >= self.burn_in and (it - self.burn_in) % self.thin == 0:
                self.U_samples_.append(U_gpu.cpu().numpy().copy())
                self.V_samples_.append(V_gpu.cpu().numpy().copy())
                self.alpha_samples_.append(alpha)
                if self.max_samples is not None:
                    while len(self.U_samples_) > self.max_samples:
                        self.U_samples_.pop(0)
                        self.V_samples_.pop(0)
                        self.alpha_samples_.pop(0)

            # Early stop: post-warmup window-based
            should_stop = False
            if self.early_stop_patience is not None and it > self.burn_in:
                p = self.early_stop_patience
                post_warmup = self.val_rmse_[warmup:]
                if (len(post_warmup) > p and
                        len(self.U_samples_) >= self.early_stop_min_samples):
                    recent = post_warmup[-p:]
                    older = post_warmup[:-p]
                    if len(older) > 0 and min(recent) >= min(older):
                        U_gpu = best_U_gpu.clone()
                        V_gpu = best_V_gpu.clone()
                        alpha = best_alpha
                        should_stop = True
                        if self.verbose:
                            print(f"  *** Early stop at iter {it} "
                                  f"(no val improvement in last {p} iters "
                                  f"after warmup={warmup}; "
                                  f"window min={min(recent):.4f}, "
                                  f"prior min={min(older):.4f}, "
                                  f"global best={best_val_rmse:.4f}, "
                                  f"{len(self.U_samples_)} samples) ***")

            if self.verbose and (it + 1) % 10 == 0:
                elapsed = time.time() - t0
                speed = (it + 1) / elapsed
                eta = (self.n_iterations - it - 1) / speed
                n_samp = len(self.U_samples_)
                print(f"  Iter {it+1}/{self.n_iterations}  "
                      f"train={train_rmse:.4f}  val={val_rmse:.4f}  "
                      f"alpha={alpha:.2f}  "
                      f"[{speed:.1f} it/s, ETA {eta:.0f}s, {n_samp} samples]")

            if should_stop:
                break

        self.U_ = U_gpu.cpu().numpy()
        self.V_ = V_gpu.cpu().numpy()
        self.alpha_ = alpha
        self.shape_ = (n_rows, n_cols)
        self.train_mask_ = train_mask
        self.val_mask_ = val_mask
        self.best_val_rmse_ = best_val_rmse
        return self

    @staticmethod
    def _build_hierarchy(group_labels_list, n_entities):
        if group_labels_list is None:
            return []
        levels = []
        for labels in group_labels_list:
            labels = np.asarray(labels)
            assert len(labels) == n_entities
            unique = np.unique(labels)
            label_map = {old: new for new, old in enumerate(unique)}
            mapped = np.array([label_map[l] for l in labels])
            levels.append((mapped, len(unique)))
        return levels

    @staticmethod
    def _sample_hyperparams_with_prior(X, mu_parent, Lambda_parent, beta0, rng):
        """Normal-Wishart with parent posterior as prior."""
        N, d = X.shape
        X_bar = X.mean(axis=0)
        S = (X - X_bar).T @ (X - X_bar)
        beta_n = beta0 + N
        nu_n = d + N
        mu_n = (beta0 * mu_parent + N * X_bar) / beta_n
        Lambda_p = (Lambda_parent + Lambda_parent.T) / 2.0 + np.eye(d) * 1e-8
        W0_inv_hier = _safe_inv(Lambda_p)
        W_n_inv = (W0_inv_hier + S
                   + (beta0 * N / beta_n) * np.outer(X_bar - mu_parent, X_bar - mu_parent))
        W_n = _safe_inv(W_n_inv)
        Lambda = sample_wishart(nu_n, W_n, rng)
        mu_cov = _safe_cho_inv(beta_n * Lambda)
        mu = sample_normal(mu_cov @ (beta_n * Lambda @ mu_n), mu_cov, rng)
        return mu, Lambda

    def predict(self, return_std=False, n_samples=None, sample_gap=1):
        """
        Posterior predictive mean (and optionally std).

        Parameters
        ----------
        n_samples : int or None
            None → use all collected samples (original behaviour).
            int  → use this many samples, picked from the end.
        sample_gap : int
            Step size when walking backwards (default 1 = consecutive).
        """
        if len(self.U_samples_) == 0:
            warnings.warn("No posterior samples. Using last sample.")
            pred = self.U_ @ self.V_.T
            if self.clip_range:
                pred = np.clip(pred, *self.clip_range)
            return pred
        U_sel, V_sel = _pick_samples(
            self.U_samples_, self.V_samples_,
            n_samples=n_samples, sample_gap=sample_gap)
        result = _streaming_posterior_pred(
            U_sel, V_sel,
            return_std=return_std, clip_range=self.clip_range)
        return result

    def score(self, R, y=None):
        R = np.array(R, dtype=np.float64)
        observed = ~np.isnan(R)
        pred = self.predict()
        return -np.sqrt(np.mean((R[observed] - pred[observed]) ** 2))

    def diagnostics(self):
        diag = {
            'alpha_trace': np.array(self.alpha_samples_),
            'rmse_trace': np.array(self.train_rmse_),
            'val_rmse_trace': np.array(self.val_rmse_),
        }
        if len(self.alpha_samples_) > 3:
            diag['alpha_ess'] = effective_sample_size(np.array(self.alpha_samples_))
        if len(self.U_samples_) > 0:
            diag['u_mean_trace'] = np.array([u.mean() for u in self.U_samples_])
        return diag


# ============================================================
# Convenience runners
# ============================================================

def run_bhpmf_gpu(trait_matrix, taxa_info, n_latent=15, n_iterations=200,
                   burn_in=100, device='cuda', random_state=42, verbose=True,
                   val_frac=0.1, val_warmup=None, max_samples=None, early_stop_patience=None):
    """GPU BHPMF runner with taxonomy hierarchy + val-driven early stop."""
    from sklearn.preprocessing import LabelEncoder

    R = np.array(trait_matrix, dtype=np.float64)

    if len(taxa_info) != R.shape[0]:
        if len(taxa_info) > R.shape[0]:
            taxa_info = taxa_info.iloc[:R.shape[0]].reset_index(drop=True)
        else:
            raise ValueError(f"taxa_info ({len(taxa_info)}) < trait_matrix ({R.shape[0]})")

    le_order = LabelEncoder()
    le_family = LabelEncoder()
    le_genus = LabelEncoder()
    order_labels = le_order.fit_transform(taxa_info['order'].values)
    family_labels = le_family.fit_transform(taxa_info['family'].values)
    genus_labels = le_genus.fit_transform(taxa_info['genus'].values)
    hierarchy_rows = [order_labels, family_labels, genus_labels]

    n_species, n_traits = R.shape
    n_obs = np.sum(~np.isnan(R))
    n_total = n_species * n_traits
    print(f"\nMatrix: {n_species} x {n_traits}")
    print(f"Observed: {n_obs}/{n_total} ({100*n_obs/n_total:.1f}%)")
    print(f"Hierarchy: {len(le_order.classes_)} orders → "
          f"{len(le_family.classes_)} families → "
          f"{len(le_genus.classes_)} genera → {n_species} species")
    print(f"Device: {device}")

    col_mean = np.nanmean(R, axis=0)
    col_std = np.nanstd(R, axis=0)
    col_std[col_std < 1e-8] = 1.0
    R_norm = (R - col_mean) / col_std

    model = BHPMF_GPU(
        n_latent=n_latent, n_iterations=n_iterations, burn_in=burn_in,
        val_frac=val_frac, val_warmup=val_warmup, max_samples=max_samples, early_stop_patience=early_stop_patience,
        device=device, random_state=random_state, verbose=verbose,
    )
    model.fit(R_norm, hierarchy_rows=hierarchy_rows)

    pred_norm, pred_std_norm = model.predict(return_std=True)
    predictions = pred_norm * col_std + col_mean
    predictions_std = pred_std_norm * col_std
    model.col_mean_ = col_mean
    model.col_std_ = col_std
    return predictions, predictions_std, model


def run_ebpmf_gpu(trait_matrix, row_embeddings, n_latent=15, n_iterations=200,
                   burn_in=100, embed_precision=1.0, learn_projection=True,
                   precision_anneal=True, device='cuda',
                   random_state=42, verbose=True,
                   val_frac=0.1, val_warmup=None, max_samples=None, early_stop_patience=None):
    """GPU EmbeddingPriorBPMF runner with val-driven early stop."""
    R = np.array(trait_matrix, dtype=np.float64)
    row_embeddings = np.array(row_embeddings, dtype=np.float64)
    assert R.shape[0] == row_embeddings.shape[0]

    n_species, n_traits = R.shape
    n_obs = np.sum(~np.isnan(R))
    n_total = n_species * n_traits
    print(f"\nMatrix: {n_species} x {n_traits}")
    print(f"Observed: {n_obs}/{n_total} ({100*n_obs/n_total:.1f}%)")
    print(f"Embedding: {row_embeddings.shape[1]}d, Device: {device}")

    col_mean = np.nanmean(R, axis=0)
    col_std = np.nanstd(R, axis=0)
    col_std[col_std < 1e-8] = 1.0
    R_norm = (R - col_mean) / col_std

    model = EmbeddingPriorBPMF_GPU(
        n_latent=n_latent, n_iterations=n_iterations, burn_in=burn_in,
        embed_precision=embed_precision, learn_projection=learn_projection,
        precision_anneal=precision_anneal,
        val_frac=val_frac, val_warmup=val_warmup, max_samples=max_samples, early_stop_patience=early_stop_patience,
        device=device, random_state=random_state, verbose=verbose,
    )
    model.fit(R_norm, row_embeddings=row_embeddings)

    pred_norm, pred_std_norm = model.predict(return_std=True)
    predictions = pred_norm * col_std + col_mean
    predictions_std = pred_std_norm * col_std
    model.col_mean_ = col_mean
    model.col_std_ = col_std
    return predictions, predictions_std, model


def run_bpmf_gpu(trait_matrix, n_latent=15, n_iterations=200, burn_in=100,
                  device='cuda', random_state=42, verbose=True,
                  val_frac=0.1, val_warmup=None, max_samples=None, early_stop_patience=None):
    """GPU BPMF runner with val-driven early stop."""
    R = np.array(trait_matrix, dtype=np.float64)
    n_species, n_traits = R.shape
    n_obs = np.sum(~np.isnan(R))
    n_total = n_species * n_traits
    print(f"\nMatrix: {n_species} x {n_traits}")
    print(f"Observed: {n_obs}/{n_total} ({100*n_obs/n_total:.1f}%)")
    print(f"Device: {device}")

    col_mean = np.nanmean(R, axis=0)
    col_std = np.nanstd(R, axis=0)
    col_std[col_std < 1e-8] = 1.0
    R_norm = (R - col_mean) / col_std

    model = BPMF_GPU(
        n_latent=n_latent, n_iterations=n_iterations, burn_in=burn_in,
        val_frac=val_frac, val_warmup=val_warmup, max_samples=max_samples, early_stop_patience=early_stop_patience,
        device=device, random_state=random_state, verbose=verbose,
    )
    model.fit(R_norm)

    pred_norm, pred_std_norm = model.predict(return_std=True)
    predictions = pred_norm * col_std + col_mean
    predictions_std = pred_std_norm * col_std
    model.col_mean_ = col_mean
    model.col_std_ = col_std
    return predictions, predictions_std, model