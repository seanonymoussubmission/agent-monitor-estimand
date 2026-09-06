#!/usr/bin/env python3
"""Unit tests for the paper's core metrics (Tests A-D of the artifact).

Self-contained: synthetic data only, no downloads. Run: python3 test_metrics.py
"""
import numpy as np
rng = np.random.default_rng(0)

def pooled_auroc(y, s):
    P, N = s[y == 1], s[y == 0]
    return float(np.mean([[(p > q) + 0.5*(p == q) for q in N] for p in P]))

def within_auroc(task, y, s):
    num = den = 0.0
    for u in set(task):
        m = task == u; P, N = s[m & (y == 1)], s[m & (y == 0)]
        if not len(P) or not len(N): continue
        num += sum((p > q) + 0.5*(p == q) for p in P for q in N); den += len(P)*len(N)
    return (num/den if den else float("nan")), den

def decomposition(task, y, s):
    P, N = np.where(y == 1)[0], np.where(y == 0)[0]
    win_n = win_d = cro_n = cro_d = 0.0
    for i in P:
        same = task[N] == task[i]
        c = (s[i] > s[N]) + 0.5*(s[i] == s[N])
        win_n += c[same].sum(); win_d += same.sum()
        cro_n += c[~same].sum(); cro_d += (~same).sum()
    tot = win_d + cro_d
    return (win_n+cro_n)/tot, (win_n/win_d if win_d else np.nan), \
           (cro_n/cro_d if cro_d else np.nan), win_d/tot

# ---- Test A: task-only predictor -> pooled high, within exactly 0.5
task = np.repeat([f"t{i}" for i in range(50)], 10)
diff = np.repeat(rng.uniform(0.1, 0.9, 50), 10)
y = (rng.random(500) < diff).astype(int)                     # 1 = failure
s_task = diff.copy()                                          # knows only the task
w, npairs = within_auroc(task, y, s_task)
p = pooled_auroc(y, s_task)
assert p > 0.65, f"A: pooled {p}"
assert abs(w - 0.5) < 1e-12, f"A: within {w}"
print(f"A  task-only:      pooled={p:.3f} (>0.65)   within={w:.3f} (=0.500)   OK")

# ---- Test B: perfect within-task predictor -> within = 1.0
s_perfect = y + rng.normal(0, 1e-9, len(y))
w, _ = within_auroc(task, y, s_perfect)
assert abs(w - 1.0) < 1e-9, f"B: {w}"
print(f"B  perfect:        within={w:.3f} (=1.000)                             OK")

# ---- Test C: single-outcome tasks contribute no pairs
task_c = np.array(["a"]*4 + ["b"]*4)
y_c    = np.array([1,1,1,1, 0,0,0,0])                        # each task uniform
w, npairs = within_auroc(task_c, y_c, rng.random(8))
assert npairs == 0 and np.isnan(w), f"C: pairs={npairs}"
print(f"C  uniform tasks:  0 within pairs, estimand undefined                 OK")

# ---- Test D: exact decomposition A_pool = w*A_within + (1-w)*A_cross
s_mixed = 0.5*diff + 0.5*rng.random(500)
Ap, Aw, Ac, wgt = decomposition(task, y, s_mixed)
lhs, rhs = Ap, wgt*Aw + (1-wgt)*Ac
assert abs(lhs - rhs) < 1e-12, f"D: {lhs} vs {rhs}"
print(f"D  decomposition:  {Ap:.6f} = {wgt:.4f}*{Aw:.4f} + {1-wgt:.4f}*{Ac:.4f}   exact  OK")
print("\nall metric tests passed")

# ---- Test E: prefix protocols and look-ahead leakage
def fixed_turn_prefix(traj, k):      return traj[:k]                # no future info
def fractional_prefix(traj, frac):   return traj[:max(1,int(round(len(traj)*frac)))]
T_long, T_short = list(range(40)), list(range(8))
assert len(fixed_turn_prefix(T_long,5)) == len(fixed_turn_prefix(T_short,5)) == 5
fl, fs = len(fractional_prefix(T_long,0.25)), len(fractional_prefix(T_short,0.25))
assert fl == 10 and fs == 2 and fl != fs, "fractional prefix must depend on final T"
print("E  prefixes:       fixed-turn independent of T; fractional reveals T          OK")
