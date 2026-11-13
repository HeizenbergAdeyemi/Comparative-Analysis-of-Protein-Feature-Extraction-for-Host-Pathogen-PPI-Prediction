"""Train classical ML and neural classifiers with leakage-safe preprocessing."""
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from models import get_classical_models, build_neural_model
from evaluate_model import save_evaluation

ALL_MODELS = ["RF", "SVM", "XGB", "DNN", "RNN", "CNN"]

def _scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
    return model.predict(X).astype(float)

def train_and_evaluate(X, samples, organism, feature_name, output_root="outputs",
                       models_to_run=None, test_size=0.2, random_state=42,
                       epochs=30, batch_size=32):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray([s["label"] for s in samples], dtype=np.int64)
    if X.ndim != 2 or X.shape[0] != len(y):
        raise ValueError(f"X/y mismatch: X={X.shape}, y={y.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("Feature matrix contains NaN/Inf values.")
    if len(np.unique(y)) != 2:
        raise ValueError("Binary PPI training requires both classes.")
    if min(np.bincount(y)) < 2:
        raise ValueError("Each class needs at least two samples for stratified splitting.")

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    run_dir = Path(output_root) / organism / feature_name
    run_dir.mkdir(parents=True, exist_ok=True)

    requested = models_to_run or ALL_MODELS
    classical = get_classical_models(random_state)
    results = {}

    for name in requested:
        name = name.upper()
        if name in classical:
            # Scaling is fitted on training data only. Tree models tolerate scaling too.
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", clone(classical[name]))
            ])
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
            score = _scores(pipeline, X_test)
            joblib.dump(pipeline, run_dir / f"{name}_model.joblib")
        elif name in ("DNN", "RNN", "CNN"):
            try:
                import tensorflow as tf
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(X_train).astype(np.float32)
                Xte = scaler.transform(X_test).astype(np.float32)
                neural = build_neural_model(name, X.shape[1], random_state)
                if name == "DNN":
                    train_input, test_input = Xtr, Xte
                else:
                    train_input, test_input = Xtr[..., None], Xte[..., None]
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=5, restore_best_weights=True
                    )
                ]
                neural.fit(
                    train_input, y_train, validation_split=0.15,
                    epochs=epochs, batch_size=batch_size, verbose=1,
                    callbacks=callbacks
                )
                score = neural.predict(test_input, verbose=0).ravel()
                pred = (score >= 0.5).astype(int)
                neural.save(run_dir / f"{name}_model.keras")
                joblib.dump(scaler, run_dir / f"{name}_scaler.joblib")
            except Exception as exc:
                print(f"[{organism}/{feature_name}/{name}] skipped: {exc}")
                continue
        else:
            print(f"Model {name} unavailable or not recognised; skipping.")
            continue

        metrics = save_evaluation(
            y_test, pred, score, run_dir, f"{feature_name}_{name}"
        )
        metrics["n_train"] = int(len(train_idx))
        metrics["n_test"] = int(len(test_idx))
        results[name] = metrics
        print(f"{organism} | {feature_name} | {name}: {metrics}")

    with (run_dir / "summary_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=True)
    return results
