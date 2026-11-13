"""Classical ML and neural network model definitions."""
import os
# Avoid known CUDA/PTX incompatibilities by default; set PPI_USE_GPU=1 to opt in.
if os.environ.get("PPI_USE_GPU", "0") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_classical_models(random_state=42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    models = {
        "RF": RandomForestClassifier(
            n_estimators=400, class_weight="balanced",
            random_state=random_state, n_jobs=-1
        ),
        "SVM": SVC(
            C=1.0, kernel="rbf", probability=True,
            class_weight="balanced", random_state=random_state
        ),
    }
    try:
        from xgboost import XGBClassifier
        models["XGB"] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            objective="binary:logistic", eval_metric="logloss",
            random_state=random_state, n_jobs=-1
        )
    except ImportError:
        print("xgboost is not installed; XGB will be omitted.")
    return models

def build_neural_model(model_name, input_dim, random_state=42):
    """Build a compact binary classifier. TensorFlow is imported lazily."""
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (
        Input, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D,
        GlobalMaxPooling1D, SimpleRNN
    )
    tf.keras.utils.set_random_seed(random_state)
    model_name = model_name.upper()
    model = Sequential(name=model_name)
    if model_name == "DNN":
        model.add(Input(shape=(input_dim,)))
        model.add(Dense(256, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.35))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation="sigmoid"))
    elif model_name in ("RNN", "CNN"):
        model.add(Input(shape=(input_dim, 1)))
        if model_name == "RNN":
            model.add(SimpleRNN(64, return_sequences=True))
            model.add(Dropout(0.3))
            model.add(SimpleRNN(32))
        else:
            model.add(Conv1D(64, 5, padding="same", activation="relu"))
            model.add(MaxPooling1D(2))
            model.add(Conv1D(32, 3, padding="same", activation="relu"))
            model.add(GlobalMaxPooling1D())
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation="sigmoid"))
    else:
        raise ValueError("Neural model must be DNN, RNN, or CNN.")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model
