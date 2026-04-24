"""
Cluster-robust standard errors and Benjamini-Hochberg correction
helpers used by the headline tables in the paper (Tables 4, 6, 7, 8,
8a) and by Figure 11.

References
----------
Petersen, M. A. (2009). Estimating Standard Errors in Finance Panel
    Data Sets: Comparing Approaches. Review of Financial Studies, 22(1),
    435-480.
Benjamini, Y. and Hochberg, Y. (1995). Controlling the False Discovery
    Rate. Journal of the Royal Statistical Society Series B, 57(1),
    289-300.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests


def cluster_robust_mean_p(values, clusters, min_g=3):
    """One-way cluster-robust SE on a constant.

    Equivalent to OLS of `values` on an intercept with
    `cov_type='cluster'` and `cov_kwds={'groups': clusters}`.

    Returns (n, mean, p_iid, p_cluster, n_clusters, t_cluster).
    p_cluster and t_cluster are NaN when the cell has fewer than
    `min_g` distinct clusters or fewer than 5 observations.
    """
    y = np.asarray(values, dtype=float)
    mask = ~np.isnan(y)
    y = y[mask]
    g = np.asarray(clusters)[mask]
    n = len(y)
    if n < 5:
        return n, float("nan"), float("nan"), float("nan"), 0, float("nan")
    mean = float(y.mean())
    iid_se = y.std(ddof=1) / np.sqrt(n)
    p_iid = 2 * (1 - stats.t.cdf(abs(mean / iid_se), df=n - 1)) if iid_se > 0 else float("nan")
    codes = pd.Categorical(g).codes.astype(np.int64)
    n_g = int(len(set(codes)))
    if n_g < min_g:
        return n, mean, p_iid, float("nan"), n_g, float("nan")
    X = np.ones((n, 1))
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": codes})
    cl_se = float(res.bse[0])
    if cl_se <= 0:
        return n, mean, p_iid, float("nan"), n_g, float("nan")
    t_cl = mean / cl_se
    p_cl = 2 * (1 - stats.t.cdf(abs(t_cl), df=n_g - 1))
    return n, mean, p_iid, p_cl, n_g, t_cl


def two_way_cluster_mean_p(values, clusters_a, clusters_b, min_g=3):
    """Two-way cluster-robust SE on a constant via the
    SE_a^2 + SE_b^2 - SE_ab^2 decomposition (Petersen 2009; Cameron,
    Gelbach, and Miller 2011).

    Returns (n, mean, p_two_way, n_a, n_b).
    """
    y = np.asarray(values, dtype=float)
    a = np.asarray(clusters_a)
    b = np.asarray(clusters_b)
    mask = ~np.isnan(y)
    y, a, b = y[mask], a[mask], b[mask]
    n = len(y)
    if n < 5:
        return n, float("nan"), float("nan"), 0, 0
    mean = float(y.mean())
    a_codes = pd.Categorical(a).codes.astype(np.int64)
    b_codes = pd.Categorical(b).codes.astype(np.int64)
    n_a = int(len(set(a_codes)))
    n_b = int(len(set(b_codes)))
    if n_a < min_g or n_b < min_g:
        return n, mean, float("nan"), n_a, n_b
    ab_codes = pd.Categorical(list(zip(a_codes, b_codes))).codes.astype(np.int64)
    X = np.ones((n, 1))
    se_a = float(sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": a_codes}).bse[0])
    se_b = float(sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": b_codes}).bse[0])
    se_ab = float(sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": ab_codes}).bse[0])
    var_two_way = se_a**2 + se_b**2 - se_ab**2
    if var_two_way <= 0:
        return n, mean, float("nan"), n_a, n_b
    se_two_way = np.sqrt(var_two_way)
    df_eff = max(min(n_a, n_b) - 1, 1)
    p = 2 * (1 - stats.t.cdf(abs(mean / se_two_way), df=df_eff))
    return n, mean, float(p), n_a, n_b


def panel_fe_diff_p(values, treat, clusters):
    """OLS of `values` on a treatment dummy plus cluster fixed effects,
    with one-way cluster-robust SEs on the cluster identifier.

    Used in §6.4a to test whether the trailing-30d return on portfolio
    stocks differs between days Cramer mentions the stock and days he
    is silent, with stock fixed effects absorbing cross-stock
    heterogeneity and cluster-robust SEs absorbing within-stock
    autocorrelation in the rolling-window response.

    Returns (n, n_clusters, beta, se_cluster, p_cluster).
    """
    y = np.asarray(values, dtype=float)
    t = np.asarray(treat, dtype=float)
    g = np.asarray(clusters)
    mask = ~np.isnan(y) & ~np.isnan(t)
    y, t, g = y[mask], t[mask], g[mask]
    n = len(y)
    codes = pd.Categorical(g).codes.astype(np.int64)
    n_g = int(len(set(codes)))
    if n_g < 3 or n < 10:
        return n, n_g, float("nan"), float("nan"), float("nan")
    fe = pd.get_dummies(pd.Categorical(g), drop_first=True).astype(float).to_numpy()
    X = np.column_stack([np.ones(n), t, fe])
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": codes})
    beta = float(res.params[1])
    se = float(res.bse[1])
    p = float(res.pvalues[1])
    return n, n_g, beta, se, p


def block_bootstrap_diff_p(values, treat, clusters, b_iters=1000, seed=42):
    """Cluster bootstrap: resample clusters (e.g., tickers) with
    replacement, recompute the (treat == 1) - (treat == 0) mean
    difference each iteration, return empirical two-sided p-value
    against the null of zero difference.

    Returns (n, n_clusters, diff_obs, p_two_sided, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(values, dtype=float)
    t = np.asarray(treat, dtype=int)
    g = np.asarray(clusters)
    mask = ~np.isnan(y)
    y, t, g = y[mask], t[mask], g[mask]
    n = len(y)
    codes = pd.Categorical(g).codes.astype(np.int64)
    n_g = int(len(set(codes)))
    if n_g < 3 or n < 10:
        return n, n_g, float("nan"), float("nan"), float("nan"), float("nan")
    diff_obs = float(y[t == 1].mean() - y[t == 0].mean())
    by_cluster = {c: (y[codes == c], t[codes == c]) for c in range(n_g)}
    diffs = np.empty(b_iters, dtype=float)
    for i in range(b_iters):
        picks = rng.integers(0, n_g, size=n_g)
        ys = []
        ts = []
        for c in picks:
            yc, tc = by_cluster[c]
            ys.append(yc)
            ts.append(tc)
        ys_arr = np.concatenate(ys)
        ts_arr = np.concatenate(ts)
        treated = ys_arr[ts_arr == 1]
        control = ys_arr[ts_arr == 0]
        if len(treated) == 0 or len(control) == 0:
            diffs[i] = np.nan
            continue
        diffs[i] = treated.mean() - control.mean()
    diffs = diffs[~np.isnan(diffs)]
    centered = diffs - diffs.mean()
    p = float((np.abs(centered) >= abs(diff_obs)).mean())
    ci_low, ci_high = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return n, n_g, diff_obs, p, ci_low, ci_high


def bh_correct(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction on a 1-D array of p-values.

    NaN inputs are passed through unchanged in the output and treated
    as non-rejected.
    """
    p = np.asarray(pvals, dtype=float)
    out_p = np.full_like(p, np.nan)
    out_r = np.zeros_like(p, dtype=bool)
    finite = ~np.isnan(p)
    if finite.sum() == 0:
        return out_r, out_p
    reject, p_adj, _, _ = multipletests(p[finite], alpha=alpha, method="fdr_bh")
    out_r[finite] = reject
    out_p[finite] = p_adj
    return out_r, out_p
