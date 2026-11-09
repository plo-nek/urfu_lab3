from __future__ import annotations

import numpy as np

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)

def train_linear_regression(data: RegressionInput) -> RegressionResult:
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).ravel()
    X_test = np.asarray(data.test_features, dtype=float)

    n_samples, n_features = X_train.shape

    w = np.zeros(n_features, dtype=float)
    b = 0.0

    lr = data.learning_rate
    epochs = data.epochs
    loss_history = []

    for _ in range(epochs):
        y_pred = X_train @ w + b
        error = y_pred - y_train

        mse = float(np.mean(error ** 2))
        loss_history.append(mse)

        dw = (2.0 / n_samples) * (X_train.T @ error)
        db = (2.0 / n_samples) * np.sum(error)

        w -= lr * dw
        b -= lr * db

    predictions = X_test @ w + b

    return RegressionResult(
        weights=w,
        bias=float(b),
        loss_history=loss_history,
        predictions=predictions,
    )

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).ravel()
    X_test = np.asarray(data.test_features, dtype=float)

    n_samples, n_features = X_train.shape

    w = np.zeros(n_features, dtype=float)
    b = 0.0

    lr = data.learning_rate
    epochs = data.epochs
    loss_history: list[float] = []
    eps = 1e-12

    for _ in range(epochs):
        z = X_train @ w + b
        p = _sigmoid(z)

        bce = float(-np.mean(y_train * np.log(p + eps) + (1 - y_train) * np.log(1 - p + eps)))
        loss_history.append(bce)

        error = p - y_train
        dw = (1.0 / n_samples) * (X_train.T @ error)
        db = (1.0 / n_samples) * np.sum(error)

        w -= lr * dw
        b -= lr * db

    proba = _sigmoid(X_test @ w + b)
    preds = (proba >= 0.5).astype(int)

    return ClassificationResult(
        weights=w,
        bias=float(b),
        loss_history=loss_history,
        probabilities=proba,
        predictions=preds,
    )

def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    )

    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).ravel()
    X_test = np.asarray(data.test_features, dtype=float)
    y_test = np.asarray(data.test_target, dtype=float).ravel()

    def get_metrics(y_true, y_pred, y_proba) -> ClassificationMetrics:
        return ClassificationMetrics(
            roc_auc=float(roc_auc_score(y_true, y_proba)),
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
        )

    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_proba = dt.predict_proba(X_test)[:, 1]

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_proba = knn.predict_proba(X_test)[:, 1]

    return BaselineComparisonResult(
        logistic_regression=get_metrics(y_test, lr_pred, lr_proba),
        decision_tree=get_metrics(y_test, dt_pred, dt_proba),
        knn=get_metrics(y_test, knn_pred, knn_proba),
    )

if __name__ == "__main__":
    import json
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    )

    df_sp = pd.read_csv("Student_Performance.csv")
    df_sp["Extracurricular Activities"] = (
        df_sp["Extracurricular Activities"].map({"Yes": 1, "No": 0})
    )

    X = df_sp.drop(columns=["Performance Index"]).to_numpy(dtype=float)
    y = df_sp["Performance Index"].to_numpy(dtype=float)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    reg = train_linear_regression(
        RegressionInput(
            train_features=X_tr,
            train_target=y_tr,
            test_features=X_te,
            test_target=y_te,
            learning_rate=0.05,
            epochs=800,
        )
    )

    test_mse = float(np.mean((reg.predictions - y_te) ** 2))
    print(f"[Regression] test MSE = {test_mse:.4f}")
    print(f"[Regression] last train MSE = {reg.loss_history[-1]:.4f}")

    df_de = pd.read_csv("german.csv", sep=";")
    y_cls = df_de["Creditability"].to_numpy(dtype=float)
    X_cls = df_de.drop(columns=["Creditability"]).to_numpy(dtype=float)

    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_cls, y_cls, test_size=0.25, random_state=42, stratify=y_cls
    )

    scaler_c = StandardScaler()
    X_tr_c = scaler_c.fit_transform(X_tr_c)
    X_te_c = scaler_c.transform(X_te_c)

    clf = train_logistic_classifier(
        ClassificationInput(
            train_features=X_tr_c,
            train_target=y_tr_c,
            test_features=X_te_c,
            test_target=y_te_c,
            learning_rate=0.1,
            epochs=1000,
        )
    )

    def compute_metrics(y_true, y_pred, y_proba):
        return {
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

    own_metrics = compute_metrics(y_te_c, clf.predictions, clf.probabilities)
    print(f"[Classification/own_logistic] {own_metrics}")

    comp = compare_classical_models(
        ClassificationInput(
            train_features=X_tr_c,
            train_target=y_tr_c,
            test_features=X_te_c,
            test_target=y_te_c,
        )
    )

    def m_to_dict(m):
        return {
            "roc_auc": float(m.roc_auc),
            "accuracy": float(m.accuracy),
            "precision": float(m.precision),
            "recall": float(m.recall),
            "f1": float(m.f1),
        }

    lr_metrics = m_to_dict(comp.logistic_regression)
    dt_metrics = m_to_dict(comp.decision_tree)
    knn_metrics = m_to_dict(comp.knn)

    print(f"[Classification/logistic_regression] {lr_metrics}")
    print(f"[Classification/decision_tree]       {dt_metrics}")
    print(f"[Classification/knn]                 {knn_metrics}")

    results = {
        "linear_regression": {"test_mse": test_mse},
        "classification": {
            "own_logistic": own_metrics,
            "logistic_regression": lr_metrics,
            "decision_tree": dt_metrics,
            "knn": knn_metrics,
        },
    }

    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nСохранено в experiment_results.json")