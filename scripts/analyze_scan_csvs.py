import pandas as pd
import numpy as np

def merge_peaks(positions, merge_gap=200):
    if len(positions) == 0:
        return []
    positions = sorted(positions)
    merged = [positions[0]]
    last = positions[0]
    for p in positions[1:]:
        if p - last > merge_gap:
            merged.append(p)
            last = p
    return merged

def evaluate_peaks(merged_peaks, known_positions, tol=100):
    tp = 0
    for k in known_positions:
        if any(abs(k - p) <= tol for p in merged_peaks):
            tp += 1
    fp = len(merged_peaks) - tp
    fn = len(known_positions) - tp
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return tp, fp, fn, prec, rec, f1

base_csv = r"x"
ft_csv   = r"x"
promoter_tsv = r"x"

base = pd.read_csv(base_csv)
ft   = pd.read_csv(ft_csv)

prom = pd.read_csv(promoter_tsv, sep="\t", skiprows=33)
pos_col = [c for c in prom.columns if "pos" in c.lower()][0]
known = prom[pos_col].dropna().astype(int).tolist()

thresholds = [0.5, 0.7, 0.8, 0.9]

for thr in thresholds:
    base_peaks = merge_peaks(base.loc[base["prob"] >= thr, "center"].tolist(), merge_gap=200)
    ft_peaks   = merge_peaks(ft.loc[ft["prob"] >= thr, "center"].tolist(), merge_gap=200)

    b = evaluate_peaks(base_peaks, known, tol=300)
    f = evaluate_peaks(ft_peaks, known, tol=300)

    print(f"Base: peaks={len(base_peaks)}  TP={b[0]} FP={b[1]} FN={b[2]}  P={b[3]:.3f} R={b[4]:.3f} F1={b[5]:.3f}")
    print(f"Fine: peaks={len(ft_peaks)}    TP={f[0]} FP={f[1]} FN={f[2]}  P={f[3]:.3f} R={f[4]:.3f} F1={f[5]:.3f}")
