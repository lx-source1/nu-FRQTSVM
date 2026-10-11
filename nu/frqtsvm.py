"""nu-FRQTSVM: kernel-free nu-reduced quadratic-surface twin SVM with fuzzy memberships.

Surface:   f(x) = 0.5 x' diag(delta) x + b' x + c = phi(x)' z,   phi(x) = [x^2/2, x, 1]  (dim 2n+1)
Fuzzy:     theta = Gaussian membership x local-class probability x imbalance ratio (Eqs. 20-24)
Primal:    min 0.5 z'Gz - nu*rho + (1/N_opp) theta_opp' xi        (Eq. nu-FRQTSVM')
Dual:      min 0.5 a' D' G^-1 D a   s.t.  sum(a) >= nu,  0 <= a <= theta_opp / N_opp   (Eq. nu-DFRQTSVM')
           G = D_main diag(theta_main) D_main' + reg*I,  z = -G^-1 D_opp a
Bound:     0 < nu <= nu_max = sum(theta_opp) / N_opp    (Prop. 3.3.1)  ->  nu = ratio * nu_max

Multi-class by one-vs-all.  `decision` selects how the OvA surfaces are used:
  "twin"  : each OvA problem trains TWO surfaces (class-k surface with nu_1, rest surface with
            nu_2); score_k = |f_rest| - |f_k|, prediction = argmax_k   (binary: one twin pair)
  "eq30"  : one surface per class (nu_1 only); prediction = argmin_k |f_k|   (paper Eq. 30)
"""
import numpy as np
import scipy.linalg as sla
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

try:                                   # reference QP solver (open source)
    from cvxopt import matrix, solvers
    solvers.options["show_progress"] = False
    HAS_CVXOPT = True
except ImportError:
    HAS_CVXOPT = False


def phi(X):
    return np.hstack([0.5 * X ** 2, X, np.ones((len(X), 1))])


# ------------------------------------------------------------------ fuzzy weights (Sec. 2.2.5)
def fuzzy_weights(X, is_main, k_nn=5, tau=1.0, eps=1e-8, use_local_prob=True,
                  use_imbalance=True, min_weight=1e-6):
    """One weight per training sample for the OvA problem `is_main` (True = class k, False = rest)."""
    is_main = np.asarray(is_main, bool)
    theta = np.empty(len(X))
    for mask in (is_main, ~is_main):                       # Eqs. (20)-(21), own-group centre/variance
        d2 = ((X[mask] - X[mask].mean(0)) ** 2).sum(1)
        theta[mask] = np.exp(-d2 / (2.0 * (tau * d2.mean() + eps)))
    if use_local_prob:                                     # Eq. (23)
        k = int(min(k_nn, len(X) - 1))
        idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
        theta = theta * (is_main[idx] == is_main[:, None]).mean(1)
    if use_imbalance:                                      # Eq. (24): ir_k = N_-k / N_k,  ir_-k = N_k / N_-k
        n_m, n_r = is_main.sum(), (~is_main).sum()
        theta = theta * np.where(is_main, n_r / n_m, n_m / n_r)
    return np.maximum(theta, min_weight)                   # strictly positive (Prop. 3.1)


# ------------------------------------------------------------------ dual QP
def _project(v, u, s):
    """Projection onto {0 <= a <= u, sum a = s}."""
    if s >= u.sum():
        return u.copy()
    lo, hi = (v - u).min(), v.max()
    for _ in range(80):
        lam = 0.5 * (lo + hi)
        lo, hi = (lam, hi) if np.clip(v - lam, 0, u).sum() > s else (lo, lam)
    return np.clip(v - 0.5 * (lo + hi), 0, u)


def _qp_pgd(W, u, s, max_iter=5000, tol=1e-10):
    """min 0.5||W a||^2 s.t. sum a = s, 0<=a<=u   (accelerated projected gradient; Q = W'W is never formed)."""
    step = 1.0 / max(np.linalg.eigvalsh(W @ W.T)[-1], 1e-12)
    a = _project(u * (s / u.sum()), u, s); y, t = a.copy(), 1.0
    for _ in range(max_iter):
        g = W.T @ (W @ y)
        a_new = _project(y - step * g, u, s)
        if np.abs(a_new - a).max() <= tol:
            return a_new
        if g @ (a_new - a) > 0:
            t, y = 1.0, a_new.copy()
        else:
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t)); y = a_new + (t - 1) / t_new * (a_new - a); t = t_new
        a = a_new
    return a


def _qp_cvxopt(W, u, s):
    N = len(u)
    Q = W.T @ W; Q = 0.5 * (Q + Q.T) + 1e-8 * np.eye(N)
    sol = solvers.qp(matrix(Q), matrix(np.zeros(N)),
                     matrix(np.vstack([-np.eye(N), np.eye(N)])), matrix(np.hstack([np.zeros(N), u])),
                     matrix(np.ones((1, N))), matrix(np.array([s], dtype=float)))
    return np.clip(np.array(sol["x"]).ravel(), 0, u)


def solve_dual(W, u, ratio, solver="auto"):
    """Solve the dual for nu = ratio * nu_max.  The problem is rescaled so that nu_max = 1 and
    ||Q|| = 1 (the minimiser is scale-free), which keeps interior-point tolerances meaningful."""
    nu_max = u.sum()
    Ws = W * nu_max
    Ws = Ws / np.sqrt(max(np.linalg.eigvalsh(Ws @ Ws.T)[-1], 1e-300))
    b = u / nu_max
    use_cvx = HAS_CVXOPT if solver == "auto" else solver == "cvxopt"
    beta = _qp_cvxopt(Ws, b, ratio) if use_cvx else _qp_pgd(Ws, b, ratio)
    return nu_max * beta, nu_max


# ------------------------------------------------------------------ model
class NuFRQTSVM(ClassifierMixin, BaseEstimator):
    """nu_ratio: float or (r1, r2).  nu_1 = r1*nu_max for the class-k surface, nu_2 = r2*nu_max for the
    rest surface ("twin"); every nu is a fraction of its own data-dependent bound nu_max (Eq. 31)."""

    def __init__(self, nu_ratio=(0.5, 0.5), decision="twin", fuzzy=True, use_local_prob=True,
                 use_imbalance=True, tau=1.0, k_nn=5, reg=1e-6, solver="auto"):
        self.nu_ratio = nu_ratio; self.decision = decision; self.fuzzy = fuzzy
        self.use_local_prob = use_local_prob; self.use_imbalance = use_imbalance
        self.tau = tau; self.k_nn = k_nn; self.reg = reg; self.solver = solver

    def _surface(self, P_main, t_main, P_opp, t_opp, ratio):
        G = (P_main.T * t_main) @ P_main + self.reg * np.eye(P_main.shape[1])        # Eq. (26)
        C = sla.cholesky(G, lower=True)
        W = sla.solve_triangular(C, P_opp.T, lower=True)                              # Q = W'W
        alpha, nu_max = solve_dual(W, t_opp / len(t_opp), ratio, self.solver)
        z = -sla.solve_triangular(C.T, W @ alpha, lower=False)                        # z = -G^-1 D a
        return z, {"nu_max": nu_max, "nu": ratio * nu_max, "alpha": alpha, "box": t_opp / len(t_opp)}

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_, yi = np.unique(y, return_inverse=True)
        K = len(self.classes_)
        if K < 2:
            raise ValueError("need >= 2 classes")
        if self.decision not in ("twin", "eq30"):
            raise ValueError("decision must be 'twin' or 'eq30'")
        r1, r2 = np.broadcast_to(np.asarray(self.nu_ratio, float), (2,))
        P = phi(X)
        twin = self.decision == "twin"
        targets = [1] if (twin and K == 2) else list(range(K))
        self.surfaces_, self.info_ = {}, {}
        for k in targets:
            main = yi == k
            th = (fuzzy_weights(X, main, self.k_nn, self.tau, 1e-8, self.use_local_prob, self.use_imbalance)
                  if self.fuzzy else np.ones(len(X)))
            z1, i1 = self._surface(P[main], th[main], P[~main], th[~main], r1)
            z2 = i2 = None
            if twin:
                z2, i2 = self._surface(P[~main], th[~main], P[main], th[main], r2)
            self.surfaces_[k], self.info_[k] = (z1, z2), (i1, i2)
        return self

    def decision_function(self, X):
        check_is_fitted(self); P = phi(check_array(X)); ks = sorted(self.surfaces_)
        if self.decision == "eq30":                       # residual to each class surface (smaller = closer)
            return np.column_stack([np.abs(P @ self.surfaces_[k][0]) for k in ks])
        s = np.column_stack([np.abs(P @ self.surfaces_[k][1]) - np.abs(P @ self.surfaces_[k][0]) for k in ks])
        return s[:, 0] if len(ks) == 1 else s             # binary: >0 -> class 1

    def predict(self, X):
        s = self.decision_function(X)
        if self.decision == "eq30":
            return self.classes_[np.argmin(s, 1)]
        if s.ndim == 1:
            return self.classes_[(s > 0).astype(int)]
        return self.classes_[np.argmax(s, 1)]
