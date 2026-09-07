#!/usr/bin/env python3
"""E3 (pre-registered, conditional branch): difficulty-only ceiling computed from a
public leaderboard's pass^k row alone.

tau2-bench submissions publish pass_1..pass_4 per domain (>=4 trials). pass_k = E[p^k]
gives the first four moments of the per-task success probability p. A beta fit by method
of moments to m1, m2 yields the q = 1-p distribution; the Gini mean difference MD and the
S2.3 ceiling 1/2 + MD/(4 mu (1-mu)) follow, plus a finite-r (r=4) leave-one-out oracle
prediction by Monte Carlo. Fit quality is checked against the published m3, m4.
Data transcribed from github.com/sierra-research/tau2-bench leaderboard submissions.
"""
import numpy as np

ROWS = {  # domain: {model: (pass1..pass4 in %)}
 "telecom": {
   "gpt-5-2 (2026-02)":        (89.69, 82.46, 76.75, 71.93),
   "gemini-3-flash (2026-03)": (91.23, 83.48, 76.54, 70.18),
   "glm-5-think (2026-03)":    (86.84, 76.32, 68.20, 62.28),
 },
 "retail": {
   "gpt-5-2 (2026-02)":        (81.58, 69.59, 59.87, 51.75),
   "gemini-3-flash (2026-03)": (76.75, 65.94, 57.89, 51.75),
   "glm-5-think (2026-03)":    (73.68, 60.38, 51.10, 43.86),
 },
}
rng = np.random.default_rng(0)
print(f"{'domain':8s} {'model':26s} {'muF':>6s} {'mixed@4':>8s} {'a':>6s} {'b':>6s} "
      f"{'m3fit':>6s} {'m4fit':>6s} {'MD':>6s} {'ceil':>6s} {'LOO@4':>6s}")
for dom, models in ROWS.items():
    for name, ps in models.items():
        m1, m2, m3, m4 = [p / 100 for p in ps]
        muF = 1 - m1
        mixed4 = 4*m1 - 6*m2 + 4*m3 - 2*m4
        v = m2 - m1**2
        ab = m1*(1-m1)/v - 1
        al, be = m1*ab, (1-m1)*ab           # beta for p; q=1-p is Beta(be,al)
        p = rng.beta(al, be, 2_000_000)
        m3f, m4f = float((p**3).mean()), float((p**4).mean())
        q = 1 - rng.beta(al, be, 1_000_000); q2 = 1 - rng.beta(al, be, 1_000_000)
        MD = float(np.abs(q - q2).mean())
        ceil = 0.5 + MD / (4 * muF * (1 - muF))
        # finite-r LOO oracle at r=4: score runs by unit's leave-one-out fail rate
        n = 3000; qs = 1 - rng.beta(al, be, n)
        fails = rng.binomial(4, qs)
        scores, labels = [], []
        for qi, f in zip(qs, fails):
            for j in range(4):
                yj = 1 if j < f else 0
                loo = (f - yj) / 3.0
                scores.append(loo); labels.append(yj)
        scores = np.array(scores); labels = np.array(labels)
        P, N = scores[labels == 1], scores[labels == 0]
        idx = rng.integers(0, len(P), 400_000); jdx = rng.integers(0, len(N), 400_000)
        loo_auc = float(((P[idx] > N[jdx]) + 0.5 * (P[idx] == N[jdx])).mean())
        print(f"{dom:8s} {name:26s} {muF:6.3f} {mixed4:8.3f} {al:6.2f} {be:6.2f} "
              f"{m3f:6.3f} {m4f:6.3f} {MD:6.3f} {ceil:6.3f} {loo_auc:6.3f}"
              f"   (pub m3={m3:.3f} m4={m4:.3f})")
