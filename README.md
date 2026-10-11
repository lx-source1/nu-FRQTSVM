# nu-FRQTSVM
Source code for the nu-FRQTSVM paper
Reference implementation of the kernel-free ν-reduced quadratic-surface twin SVM with fuzzy
memberships ("A novel robust Quadratic ν-Twin Support Vector Machine for Multi-class
Classification with Applications").

* `frqtsvm.py` – model (scikit-learn API): diagonal quadratic surface, fuzzy weights (Eqs. 20-24),
  dual QP with ν = ratio · ν_max (Eq. 31), one-vs-all.
* `run_experiment.py` – leak-free nested-CV benchmark (Accuracy, Macro-F1, Balanced Accuracy, CPU time).
* `tests/test_model.py` – QP optimality, ν feasibility and the support-vector bound (Prop. 3.3.1).

```bash
pip install -r requirements.txt
python -m pytest -q tests          # or: python -c "import tests.test_model"
python run_experiment.py --data-dir data/ --out-dir results/     # CSVs: features..., label (last column)
```

```python
from frqtsvm import NuFRQTSVM
clf = NuFRQTSVM(nu_ratio=(0.3, 0.7)).fit(X_scaled, y)   # ν_1 = 0.3·ν_max (class surface), ν_2 = 0.7·ν_max
```

Options: `decision="twin" | "eq30"`, `fuzzy=False` (non-fuzzy ν-RQTSVM), `use_local_prob`,
`use_imbalance`, `tau`, `k_nn`, `solver="auto" | "cvxopt" | "pgd"`.

Datasets are not redistributed (public benchmarks: Kaggle; the steelmaking data is proprietary).

The proprietary steelmaking data cannot be shared.
