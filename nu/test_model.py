import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from frqtsvm import NuFRQTSVM, _qp_pgd, fuzzy_weights


def test_qp_vs_reference():
    rng = np.random.default_rng(0); W, u = rng.normal(size=(7, 40)), rng.uniform(.01, .05, 40); s = .5 * u.sum()
    a = _qp_pgd(W, u, s); f = lambda v: .5 * np.sum((W @ v) ** 2)
    ref = minimize(f, u * s / u.sum(), method="SLSQP", bounds=list(zip(np.zeros(40), u)),
                   constraints=[{"type": "eq", "fun": lambda v: v.sum() - s}], options={"ftol": 1e-14, "maxiter": 1000})
    assert abs(a.sum() - s) < 1e-8 and f(a) <= ref.fun + 1e-7


def test_models_and_bounds():
    X, y = make_classification(450, 6, n_informative=4, n_classes=3, n_clusters_per_class=1, random_state=0)
    X = StandardScaler().fit_transform(X)
    for dec in ("twin", "eq30"):
        m = NuFRQTSVM(decision=dec, nu_ratio=(.3, .6)).fit(X, y)
        assert (m.predict(X) == y).mean() > .7
        for infos in m.info_.values():
            for i in filter(None, infos):
                assert 0 < i["nu"] <= i["nu_max"] and abs(i["alpha"].sum() - i["nu"]) < 1e-6
                assert i["nu"] <= i["box"][i["alpha"] > 1e-9].sum() + 1e-8        # Prop. 3.3.1
    Xb, yb = make_classification(200, 4, random_state=1)
    assert len(NuFRQTSVM().fit(Xb, yb).surfaces_) == 1                             # binary twin pair


def test_weights_positive():
    X = np.random.default_rng(0).normal(size=(60, 3)); m = np.arange(60) < 20
    assert (fuzzy_weights(X, m) > 0).all()
