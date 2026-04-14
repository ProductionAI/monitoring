"""
ML-Ensemble (mlens) Basic Example
==================================
Demonstrates a stacked SuperLearner ensemble on a classification task
using the Breast Cancer dataset from scikit-learn.

Architecture:
  Layer 1 (Base Learners): Random Forest, Gradient Boosting, KNN, SVM
  Layer 2 (Meta Learner):  Logistic Regression

Compatibility patches
---------------------
mlens 0.2.3 has two bugs on Python 3.12 + NumPy 2.x that are fixed
automatically below before the library is imported:

  1. mlens/externals/sklearn/type_of_target.py
       `from collections import Sequence`
     → `from collections.abc import Sequence`
     (collections.Sequence removed in Python 3.10)

  2. mlens/index/base.py
       `np.int`  →  `int`
     (np.int alias removed in NumPy 2.0)
"""

# ── Compatibility patches (must run before importing mlens) ───────────────────
import importlib, pathlib, re, site, sys

def _patch_file(rel_path: str, replacements: list) -> None:
    """Apply regex replacements to a file inside the mlens site-package."""
    search_paths = site.getsitepackages() + [site.getusersitepackages()] + sys.path
    for base in search_paths:
        candidate = pathlib.Path(base) / rel_path
        if candidate.exists():
            src = candidate.read_text()
            for old, new in replacements:
                src = re.sub(old, new, src)
            candidate.write_text(src)
            return

_patch_file(
    "mlens/externals/sklearn/type_of_target.py",
    [(r"from collections import Sequence",
      "from collections.abc import Sequence")],
)

_patch_file(
    "mlens/index/base.py",
    [(r"\bnp\.int\b", "int")],
)
# ─────────────────────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from mlens.ensemble import SuperLearner

# ── 1. Load & split data ──────────────────────────────────────────────────────
print("=" * 60)
print("  ML-Ensemble SuperLearner — Breast Cancer Classification")
print("=" * 60)

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDataset : Breast Cancer")
print(f"Features: {X.shape[1]}")
print(f"Train   : {X_train.shape[0]} samples")
print(f"Test    : {X_test.shape[0]} samples")

# ── 2. Define base learners with preprocessing pipelines ─────────────────────
# mlens supports per-pipeline preprocessing via dicts, so tree-based models
# and distance/kernel models can have different (or no) scaling applied.

preprocessing = {
    "scaled":   [StandardScaler()],  # for KNN and SVM
    "unscaled": [],                   # for tree-based models
}

base_learners = {
    "scaled": [
        KNeighborsClassifier(n_neighbors=7),
        SVC(kernel="rbf", probability=True, random_state=42),
    ],
    "unscaled": [
        RandomForestClassifier(n_estimators=100, random_state=42),
        GradientBoostingClassifier(n_estimators=100, random_state=42),
    ],
}

# ── 3. Build SuperLearner ─────────────────────────────────────────────────────
# SuperLearner generates out-of-fold (OOF) predictions in Layer 1,
# then trains the meta learner on those OOF predictions in Layer 2.

ensemble = SuperLearner(
    folds=5,
    random_state=42,
    verbose=1,
    backend="threading",
)

ensemble.add(base_learners, preprocessing)           # Layer 1: base learners
ensemble.add_meta(LogisticRegression(max_iter=500))  # Layer 2: meta learner

# ── 4. Train & evaluate ───────────────────────────────────────────────────────
print("\n--- Fitting SuperLearner ---")
ensemble.fit(X_train, y_train)

print("\n--- Generating Predictions ---")
y_pred = ensemble.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy : {acc:.4f} ({acc*100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# ── 5. Compare against individual base learners ───────────────────────────────
print("--- Individual Base Learner Comparison ---")
individual_models = {
    "KNN (k=7)":          KNeighborsClassifier(n_neighbors=7),
    "SVM (RBF)":          SVC(kernel="rbf", random_state=42),
    "Random Forest":      RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":  GradientBoostingClassifier(n_estimators=100, random_state=42),
}

scaler = StandardScaler().fit(X_train)
X_train_sc = scaler.transform(X_train)
X_test_sc  = scaler.transform(X_test)

results = {}
for name, model in individual_models.items():
    needs_scale = name in ("KNN (k=7)", "SVM (RBF)")
    Xtr = X_train_sc if needs_scale else X_train
    Xte = X_test_sc  if needs_scale else X_test
    model.fit(Xtr, y_train)
    acc_i = accuracy_score(y_test, model.predict(Xte))
    results[name] = acc_i
    print(f"  {name:<22} accuracy: {acc_i:.4f}")

print(f"\n  {'SuperLearner':<22} accuracy: {acc:.4f}  <- ensemble")
delta = acc - max(results.values())
print(f"\n  Improvement over best base learner: {delta:+.4f}")

# ── 6. Inspect the fitted ensemble ───────────────────────────────────────────
print("\n--- Ensemble Architecture ---")
print(ensemble)
print("\nDone.")
