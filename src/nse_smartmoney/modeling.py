"""Predictive modeling layer.

- ols_signal_test    : OLS with HAC (Newey-West) robust errors —
  do smart-money features explain forward returns?
- classification_suite: logistic regression + random forest predicting
  the sign of the 5-day forward return, time-series split, scored with
  accuracy / precision / recall / ROC-AUC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["deliv_z", "volume_z", "back_ret_5d", "dii_net_cr",
            "fii_net_cr", "smart_deal_net_qty", "dii_deal_net_qty",
            "accum_score"]
TARGET = "fwd_ret_5d"


def _design(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=FEATURES + [TARGET]).sort_values("date")
    # scale deal quantities to millions of shares for readable coefficients
    for c in ("smart_deal_net_qty", "dii_deal_net_qty"):
        df[c] = df[c] / 1e6
    for c in ("dii_net_cr", "fii_net_cr"):
        df[c] = df[c] / 1e3            # ₹ '000 crore
    return df


def ols_signal_test(features: pd.DataFrame):
    """OLS of fwd 5-day return on smart-money features, HAC errors."""
    import statsmodels.api as sm

    df = _design(features)
    X = sm.add_constant(df[FEATURES])
    model = sm.OLS(df[TARGET], X).fit(cov_type="HAC",
                                      cov_kwds={"maxlags": 5})
    summary = pd.DataFrame({"coef": model.params, "t": model.tvalues,
                            "p_value": model.pvalues})
    summary["significant"] = (summary["p_value"] < 0.05).astype(int)
    return model, summary.round(5)


def classification_suite(features: pd.DataFrame) -> pd.DataFrame:
    """Predict up/down 5-day forward move. Chronological 70/30 split
    (no shuffling — avoids look-ahead)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.preprocessing import StandardScaler

    df = _design(features)
    y = (df[TARGET] > 0).astype(int)
    X = df[FEATURES].values
    split = int(len(df) * 0.7)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=25,
            random_state=42, n_jobs=-1),
    }
    rows = []
    for name, m in models.items():
        Xtr = X_tr_s if name == "LogisticRegression" else X_tr
        Xte = X_te_s if name == "LogisticRegression" else X_te
        m.fit(Xtr, y_tr)
        proba = m.predict_proba(Xte)[:, 1]
        pred = (proba > 0.5).astype(int)
        rows.append({
            "model": name,
            "n_train": len(y_tr), "n_test": len(y_te),
            "base_rate": round(float(y_te.mean()), 4),
            "accuracy": round(accuracy_score(y_te, pred), 4),
            "precision": round(precision_score(y_te, pred,
                                               zero_division=0), 4),
            "recall": round(recall_score(y_te, pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_te, proba), 4)})
    res = pd.DataFrame(rows)

    # feature importances from the forest for interpretability
    rf = models["RandomForest"]
    imp = pd.DataFrame({"feature": FEATURES,
                        "importance": rf.feature_importances_}) \
        .sort_values("importance", ascending=False).reset_index(drop=True)
    res.attrs["importances"] = imp
    return res
