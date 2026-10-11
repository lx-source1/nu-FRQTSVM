"""Leak-free benchmark for nu-FRQTSVM (nested CV): Accuracy / Macro-F1 / Balanced Accuracy / CPU time.

Per outer fold (stratified 5-fold, shuffle, seed 42): Z-score and SMOTE are fitted on the training part
only; nu-ratios are chosen by an inner 5-fold CV on that training part; the test fold is never used
for preprocessing or tuning.  CSV format: header row, features..., label in the LAST column.

  python run_experiment.py --data-dir data/ --out-dir results/
"""
import argparse, os, time, warnings
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from frqtsvm import NuFRQTSVM

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

RATIOS = (0.1, 0.3, 0.5, 0.7, 0.9)        # five candidates in (0, nu_max]  (Appendix A)


def load(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].to_numpy(float); y = df.iloc[:, -1].to_numpy()
    ok = np.isfinite(X).all(1) & pd.notna(y)
    return X[ok], LabelEncoder().fit_transform(y[ok])


def prepare(Xtr, ytr, seed, use_smote):
    sc = StandardScaler().fit(Xtr); Xs = sc.transform(Xtr)
    if use_smote and SMOTE is not None:
        k = min(5, np.bincount(ytr).min() - 1)
        if k >= 1:
            Xs, ytr = SMOTE(k_neighbors=k, random_state=seed).fit_resample(Xs, ytr)
    return sc, Xs, ytr


def fit_predict(Xtr, ytr, Xte, params, seed, use_smote, **kw):
    sc, Xs, ys = prepare(Xtr, ytr, seed, use_smote)
    t0 = time.process_time()
    model = NuFRQTSVM(nu_ratio=params, **kw).fit(Xs, ys)
    return model.predict(sc.transform(Xte)), time.process_time() - t0


def grid(decision):
    if decision == "eq30":                       # one surface per class -> a single nu ratio
        return [(r, r) for r in RATIOS]
    return [(a, b) for a in RATIOS for b in RATIOS]   # nu_1 (class surface) x nu_2 (rest surface)


def evaluate(X, y, seed=42, use_smote=True, **kw):
    rows = []
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
        best, best_acc = None, -1.0
        for params in grid(kw.get("decision", "twin")):                  # inner CV, training part only
            accs = [accuracy_score(y[tr][vi], fit_predict(X[tr][ti], y[tr][ti], X[tr][vi], params, seed, use_smote, **kw)[0])
                    for ti, vi in StratifiedKFold(5, shuffle=True, random_state=seed).split(X[tr], y[tr])]
            if np.mean(accs) > best_acc:
                best, best_acc = params, float(np.mean(accs))
        yp, cpu = fit_predict(X[tr], y[tr], X[te], best, seed, use_smote, **kw)
        rows.append({"fold": fold + 1, "nu_ratio_1": best[0], "nu_ratio_2": best[1],
                     "accuracy": 100 * accuracy_score(y[te], yp),
                     "macro_f1": 100 * f1_score(y[te], yp, average="macro"),
                     "balanced_accuracy": 100 * balanced_accuracy_score(y[te], yp), "cpu_time_s": cpu})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--out-dir", default="results")
    ap.add_argument("--decision", default="twin", choices=["twin", "eq30"])
    ap.add_argument("--solver", default="auto", choices=["auto", "cvxopt", "pgd"])
    ap.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS), help="nu/nu_max candidates")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--no-smote", action="store_true")
    a = ap.parse_args()
    RATIOS = tuple(a.ratios)
    if SMOTE is None and not a.no_smote:
        warnings.warn("imbalanced-learn not installed: running WITHOUT SMOTE")
    os.makedirs(a.out_dir, exist_ok=True); summary, folds = [], []
    for fn in sorted(f for f in os.listdir(a.data_dir) if f.endswith(".csv")):
        X, y = load(os.path.join(a.data_dir, fn)); name = fn[:-4]
        df = evaluate(X, y, a.seed, not a.no_smote, decision=a.decision, solver=a.solver)
        folds.append(df.assign(dataset=name))
        summary.append({"dataset": name, **{f"{m}_mean": df[m].mean() for m in ("accuracy", "macro_f1", "balanced_accuracy")},
                        **{f"{m}_std": df[m].std(ddof=1) for m in ("accuracy", "macro_f1", "balanced_accuracy")},
                        "cpu_time_s": df["cpu_time_s"].sum()})
        print(name, {k: round(v, 2) for k, v in summary[-1].items() if k != "dataset"}, flush=True)
    pd.DataFrame(summary).to_csv(os.path.join(a.out_dir, "summary.csv"), index=False)
    pd.concat(folds).to_csv(os.path.join(a.out_dir, "foldwise.csv"), index=False)
