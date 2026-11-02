"""Fraud detection on the full Kaggle credit card dataset with a time based evaluation.

Usage: python analysis.py path/to/creditcard.csv
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
BOOTSTRAPS = 1000


def models():
    # Scaling sits inside each pipeline so it is fit on training data only
    return {
        "Majority class baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
        "Logistic regression (class weights)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
        "Random forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
        "Gradient boosting": HistGradientBoostingClassifier(random_state=SEED),
    }


def best_f1_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def pr_auc_interval(y_true, scores):
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(BOOTSTRAPS):
        sample = rng.integers(0, len(y_true), len(y_true))
        if y_true[sample].sum() > 0:
            values.append(average_precision_score(y_true[sample], scores[sample]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main(path):
    data = pd.read_csv(path).sort_values("Time").reset_index(drop=True)
    features = data.drop(columns="Class")
    target = data["Class"].to_numpy()
    amount = data["Amount"].to_numpy()
    train_end, valid_end = int(len(data) * 0.6), int(len(data) * 0.8)

    results = {
        "rows": len(data),
        "frauds": int(target.sum()),
        "fraud_rate": float(target.mean()),
        "split": {
            "train": {"rows": train_end, "frauds": int(target[:train_end].sum())},
            "validation": {"rows": valid_end - train_end, "frauds": int(target[train_end:valid_end].sum())},
            "test": {"rows": len(data) - valid_end, "frauds": int(target[valid_end:].sum())},
        },
        "models": {},
    }
    print(json.dumps({k: results[k] for k in ("rows", "frauds", "fraud_rate", "split")}, indent=2))

    x_train, y_train = features.iloc[:train_end], target[:train_end]
    x_valid, y_valid = features.iloc[train_end:valid_end], target[train_end:valid_end]
    x_test, y_test = features.iloc[valid_end:], target[valid_end:]
    test_amount = amount[valid_end:]

    for name, model in models().items():
        model.fit(x_train, y_train)
        valid_scores = model.predict_proba(x_valid)[:, 1]
        test_scores = model.predict_proba(x_test)[:, 1]
        # The threshold comes from the validation period, then is applied once to the test period
        threshold = best_f1_threshold(y_valid, valid_scores)
        flagged = test_scores >= threshold
        row = {
            "pr_auc": float(average_precision_score(y_test, test_scores)),
            "pr_auc_95ci": pr_auc_interval(y_test, test_scores),
            "roc_auc": float(roc_auc_score(y_test, test_scores)),
            "brier": float(brier_score_loss(y_test, test_scores)),
            "threshold": threshold,
            "precision": float(precision_score(y_test, flagged, zero_division=0)),
            "recall": float(recall_score(y_test, flagged, zero_division=0)),
            "flagged": int(flagged.sum()),
            "fraud_amount_caught_share": float(test_amount[flagged & (y_test == 1)].sum() / max(test_amount[y_test == 1].sum(), 1e-9)),
        }
        results["models"][name] = row
        print(name, json.dumps(row))

    # A random split can mix future transactions into training, so compare it with the time split
    random_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    gradient_boosting = HistGradientBoostingClassifier(random_state=SEED)
    random_cv = cross_val_score(gradient_boosting, features, target, cv=random_folds, scoring="average_precision", n_jobs=-1)
    results["gradient_boosting_random_5fold_pr_auc"] = {"mean": float(random_cv.mean()), "std": float(random_cv.std())}
    print("Random 5 fold PR-AUC for gradient boosting:", results["gradient_boosting_random_5fold_pr_auc"])

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1])
