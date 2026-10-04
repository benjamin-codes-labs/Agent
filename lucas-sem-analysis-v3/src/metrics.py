"""Stage 6a: phase fractions and batch statistics.

Writes outputs/metrics/phase_fractions.csv  per sample: 4 classes (% of valid pixels), excluded %, n_valid_px
       outputs/metrics/batch_stats.csv      per batch x class: mean, SD, n; Kruskal-Wallis p; pairwise
                                            Mann-Whitney p (Holm-corrected); session-stratified permutation p
Session control: samples sharing an image height were probably imaged in one session. The stratified test
permutes batch labels only within those height groups, so a batch effect must show up between samples of the
same session to count.

Usage: python -m src.metrics
"""
import itertools

import numpy as np
from scipy import stats

from . import config as C
from .teacher import siox_thrlike

N_PERM = 20000


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def kw_stat(values, groups):
    return stats.kruskal(*[values[groups == g] for g in np.unique(groups)]).statistic


def stratified_perm_p(values, groups, strata, rng):
    obs = kw_stat(values, groups)
    hits = 0
    idx_by_stratum = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    for _ in range(N_PERM):
        g = groups.copy()
        for idx in idx_by_stratum:
            g[idx] = rng.permutation(g[idx])
        hits += kw_stat(values, g) >= obs - 1e-12
    return (hits + 1) / (N_PERM + 1)


def main():
    qc = {r["sample_id"]: r for r in C.read_csv(C.QC)}
    t_low = {r["sample_id"]: float(r["bse_t_low"]) for r in C.read_csv(C.OUT_MET / "seed_coverage.csv")}
    rows = []
    for batch, sid in C.sample_ids():
        lab = np.load(C.proc_dir(sid) / "final_labels.npy")
        valid = lab != C.EXCLUDED
        n = int(valid.sum())
        # deep pores = pore pixels that are also in the dark BSE multi-Otsu class (what a BSE threshold sees);
        # open pores = the rest of the pore class (grey-floored, unfilled pores)
        deep = (lab == C.PORE) & (C.load_channels(sid, ["BSE"])["BSE"] < t_low[sid])
        rows.append({"batch": batch, "sample_id": sid, "height_group": qc[sid]["height_group"],
                     **{f"{c}_pct": round(100 * float((lab[valid] == k).sum()) / n, 3) for k, c in enumerate(C.CLASSES)},
                     "siox_thrlike_pct": round(siox_thrlike(lab), 3),
                     "pore_deep_pct": round(100 * float(deep.sum()) / n, 3),
                     "pore_open_pct": round(100 * float(((lab == C.PORE) & ~deep).sum()) / n, 3),
                     "excluded_pct": round(100 * float((~valid).mean()), 3), "n_valid_px": n})
    C.write_csv(C.OUT_MET / "phase_fractions.csv", rows)

    rng = np.random.default_rng(0)
    batches = np.array([r["batch"] for r in rows])
    strata = np.array([r["height_group"] for r in rows])
    out = []
    for c in C.CLASSES + ["pore_deep", "pore_open"]:
        v = np.array([r[f"{c}_pct"] for r in rows])
        kw_p = stats.kruskal(*[v[batches == b] for b in sorted(set(batches))]).pvalue
        pairs = list(itertools.combinations(sorted(set(batches)), 2))
        raw = np.array([stats.mannwhitneyu(v[batches == a], v[batches == b]).pvalue for a, b in pairs])
        adj = holm(raw)
        perm_p = stratified_perm_p(v, batches, strata, rng)
        for b in sorted(set(batches)):
            x = v[batches == b]
            out.append({"class": c, "batch": b, "n": len(x), "mean_pct": round(x.mean(), 2),
                        "sd_pct": round(x.std(ddof=1), 2), "min_pct": round(x.min(), 2), "max_pct": round(x.max(), 2),
                        "kruskal_p": round(kw_p, 4), "session_stratified_perm_p": round(perm_p, 4),
                        **{f"mw_holm_{a[-1]}v{bb[-1]}": round(p, 4) for (a, bb), p in zip(pairs, adj)}})
        print(f"{c:9s} " + "  ".join(f"{b[-1]}: {v[batches == b].mean():5.2f}±{v[batches == b].std(ddof=1):4.2f}" for b in sorted(set(batches)))
              + f" | KW p={kw_p:.3f} | session-stratified p={perm_p:.3f} | MW-Holm " +
              ", ".join(f"{a[-1]}v{bb[-1]}={p:.3f}" for (a, bb), p in zip(pairs, adj)))
    C.write_csv(C.OUT_MET / "batch_stats.csv", out)


if __name__ == "__main__":
    main()
