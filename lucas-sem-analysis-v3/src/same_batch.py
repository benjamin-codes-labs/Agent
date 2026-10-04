"""Are Batch_1 and Batch_2 distinguishable at all, or statistically the same population?

Four angles, each calibrated against the B1-vs-B3 and B2-vs-B3 comparisons with Batch_3 subsampled to 7 samples
(equal n = equal power, otherwise B3 comparisons would look stronger just because B3 has 17 samples):
  1. count test: how many of the ~77 features differ at p < 0.05, against a label-permutation null of that count
     (handles the correlation between features automatically);
  2. multivariate energy distance between the two groups (robust-standardised features), permutation p;
  3. neighbour mixing: share of each sample's 3 nearest neighbours (all 31 samples) that are from its own batch,
     against a permutation null;
  4. effect sizes with 95 % CIs and the minimum detectable difference (80 % power, n = 7 vs 7) for the key
     physical quantities, i.e. how large a B1-B2 difference the data can rule out.
Everything is run on all features and on the subset that does not track the acquisition probes (|rho| <= 0.5).

Writes outputs/metrics/same_batch.json.

Usage: python -m src.same_batch
"""
import json

import numpy as np
from scipy import stats

from . import config as C
from .b12 import PROBES, table

BATCHES = ["Batch_1", "Batch_2", "Batch_3"]
KEY = ["pore_all_frac", "pore_open_excess", "pore_deep_frac", "siox_frac", "siox_ecd_aw", "cbd_frac",
       "graphite_frac", "etd_graphite_roughness", "bse_siox_contrast_ratio"]


def robust_z(X):
    med = np.median(X, 0)
    iqr = np.subtract(*np.percentile(X, [75, 25], 0)) / 1.349
    iqr[iqr == 0] = X.std(0)[iqr == 0] + 1e-9
    return np.clip((X - med) / iqr, -5, 5)


def count_sig(Xa, Xb):
    return int(sum(stats.mannwhitneyu(Xa[:, j], Xb[:, j]).pvalue < 0.05 for j in range(Xa.shape[1])))


def energy(Z, a, b):
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
    return 2 * D[np.ix_(a, b)].mean() - D[np.ix_(a, a)].mean() - D[np.ix_(b, b)].mean()


def perm_test(stat, Z, a, b, rng, n):
    obs = stat(Z, a, b)
    idx = np.concatenate([a, b])
    hits = 0
    for _ in range(n):
        p = rng.permutation(idx)
        hits += stat(Z, p[:len(a)], p[len(a):]) >= obs - 1e-12
    return obs, (hits + 1) / (n + 1)


def pair_tests(X, Z, y, pa, pb, rng, n_sub=30, n_perm=300):
    """Count and energy tests for batches pa vs pb; a batch with > 7 samples is subsampled to 7 (n_sub draws)."""
    ia, ib = np.flatnonzero(y == pa), np.flatnonzero(y == pb)
    draws = []
    for _ in range(n_sub if max(len(ia), len(ib)) > 7 else 1):
        a = rng.choice(ia, 7, replace=False) if len(ia) > 7 else ia
        b = rng.choice(ib, 7, replace=False) if len(ib) > 7 else ib
        cnt, p_cnt = perm_test(lambda Z_, a_, b_: count_sig(X[a_], X[b_]), Z, a, b, rng, n_perm)
        e, p_e = perm_test(energy, Z, a, b, rng, n_perm * 3)
        draws.append((cnt, p_cnt, e, p_e))
    d = np.array(draws)
    return {"n_features_p_lt_0.05": float(np.median(d[:, 0])), "count_perm_p": round(float(np.median(d[:, 1])), 4),
            "energy_distance": round(float(np.median(d[:, 2])), 3), "energy_perm_p": round(float(np.median(d[:, 3])), 4),
            "subsamples": len(draws)}


def neighbour_mixing(Z, y, rng, k=3, n_perm=5000):
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, 1)[:, :k]
    def rates(lab):
        same = (lab[nn] == lab[:, None]).mean(1)
        return np.array([same[lab == c].mean() for c in range(3)])
    obs = rates(y)
    null = np.array([rates(rng.permutation(y)) for _ in range(n_perm)])
    out = {}
    for c in range(3):
        out[BATCHES[c]] = {"own_batch_neighbour_share": round(float(obs[c]), 3),
                           "chance_share": round(float(null[:, c].mean()), 3),
                           "perm_p": round(float((np.sum(null[:, c] >= obs[c]) + 1) / (n_perm + 1)), 4)}
    # within B1 + B2 only: are B1 samples' neighbours (among B1/B2) more often B1?
    m = y < 2
    Dm = D[np.ix_(m, m)]
    nnm = np.argsort(Dm, 1)[:, :k]
    ym = y[m]
    o = (ym[nnm] == ym[:, None]).mean()
    nl = np.array([(yp[nnm] == yp[:, None]).mean() for yp in (rng.permutation(ym) for _ in range(n_perm))])
    out["within_B1_B2_only"] = {"same_batch_neighbour_share": round(float(o), 3), "chance_share": round(float(nl.mean()), 3),
                                "perm_p": round(float((np.sum(nl >= o) + 1) / (n_perm + 1)), 4)}
    return out


def effect_sizes(X, names, y, rng):
    a_i, b_i = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    tcrit = stats.t.ppf(0.975, 12) + stats.t.ppf(0.80, 12)
    out = {}
    for k in KEY:
        if k not in names:
            continue
        j = names.index(k)
        a, b = X[a_i, j], X[b_i, j]
        diff = b.mean() - a.mean()
        boot = [rng.choice(b, 7).mean() - rng.choice(a, 7).mean() for _ in range(5000)]
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        out[k] = {"B1_mean": round(float(a.mean()), 4), "B2_mean": round(float(b.mean()), 4),
                  "B2_minus_B1": round(float(diff), 4), "ci95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
                  "cohens_d": round(float(diff / sd), 2) if sd > 0 else None,
                  "min_detectable_diff_80pct_power": round(float(tcrit * sd * np.sqrt(2 / 7)), 4),
                  "mw_p": round(float(stats.mannwhitneyu(a, b).pvalue), 4)}
    return out


def main():
    rng = np.random.default_rng(0)
    sids, names, X = table()
    batch = {s: b for b, s in C.sample_ids()}
    y = np.array([BATCHES.index(batch[s]) for s in sids])
    probes = {r["sample_id"]: r for r in C.read_csv(C.OUT_MET / "bid_session_probes.csv")}
    P = np.array([[float(probes[s][p]) for p in PROBES] for s in sids])
    sess = np.array([max(abs(stats.spearmanr(X[:, j], P[:, k]).statistic) for k in range(P.shape[1])) for j in range(X.shape[1])])
    out = {"n_features_all": len(names), "n_features_non_session": int((sess <= 0.5).sum())}
    for subset, cols in (("all_features", np.arange(len(names))), ("non_session_features", np.flatnonzero(sess <= 0.5))):
        Xs = X[:, cols]
        Z = robust_z(Xs)
        res = {"expected_count_under_null": round(0.05 * len(cols), 1)}
        for pa, pb in ((0, 1), (0, 2), (1, 2)):
            res[f"{BATCHES[pa]}_vs_{BATCHES[pb]}"] = pair_tests(Xs, Z, y, pa, pb, rng)
            print(subset, BATCHES[pa], BATCHES[pb], res[f"{BATCHES[pa]}_vs_{BATCHES[pb]}"], flush=True)
        res["neighbour_mixing"] = neighbour_mixing(Z, y, rng)
        out[subset] = res
    out["B1_vs_B2_effect_sizes"] = effect_sizes(X, names, y, rng)
    C.write_json(C.OUT_MET / "same_batch.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
