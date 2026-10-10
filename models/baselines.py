"""Pooling baselines (mean / max, same interface as ABMIL) and the cheap Phase 0 baselines."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.abmil import PoolMIL  # noqa: F401  (mean/max pooling heads)


def pooled_features(feats, how):
    """feats: list of [N,D] arrays -> [n_scans, D] (mean, max, or both concatenated)."""
    if how == "mean":
        return np.stack([f.mean(0) for f in feats])
    if how == "max":
        return np.stack([f.max(0) for f in feats])
    return np.concatenate([pooled_features(feats, "mean"), pooled_features(feats, "max")], 1)


def fit_per_class(model_fn, X_tr, Y_tr, X_eval):
    """One independent binary model per class; returns {name: probs [n, C]}."""
    out = {k: np.zeros((len(v), Y_tr.shape[1])) for k, v in X_eval.items()}
    for c in range(Y_tr.shape[1]):
        mdl = model_fn()
        mdl.fit(X_tr, Y_tr[:, c])
        for k, X in X_eval.items():
            out[k][:, c] = mdl.predict_proba(X)[:, 1]
    return out


def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced"))


def gbm():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, class_weight="balanced", random_state=0)
