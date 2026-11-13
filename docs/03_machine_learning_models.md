# Part 3 — Machine learning and deep learning

Six classifiers are defined in `models.py` and trained by `train_model.py`. All see the same pair feature vector.

## Training protocol (`train_model.train_and_evaluate`)

- **Split:** `train_test_split` on pair indices, `test_size=0.2`, `random_state=42`, **stratified by label**. This is a random split of *pairs*; it does not keep proteins apart (see Part 7).
- **Checks before training:** `X` is 2-D and finite, `X` and `y` have equal length, both classes are present, each class has at least 2 samples.
- **Classical models** run inside a scikit-learn `Pipeline`: median `SimpleImputer` → `StandardScaler` → classifier. The imputer and scaler are fitted on the training set only.
- **Neural models** use a `StandardScaler` fitted on the training set, then a 0.5 probability threshold to turn scores into class predictions.
- **GPU:** `models.py` hides all GPUs (`CUDA_VISIBLE_DEVICES=-1`) unless the environment variable `PPI_USE_GPU=1` is set (Part 6).

## Classical models

| Model | Library | Settings |
|---|---|---|
| Random Forest (RF) | scikit-learn | 400 trees, `class_weight="balanced"`, `random_state=42`, all cores |
| SVM | scikit-learn `SVC` | RBF kernel, `C=1.0`, `class_weight="balanced"`, `probability=True` (needed for ROC) |
| XGBoost (XGB) | xgboost | 400 trees, `max_depth=6`, `learning_rate=0.05`, `subsample=0.85`, `colsample_bytree=0.85`, logistic objective; **no class weighting** |

RF and SVM are class-balanced while XGBoost is not, so their precision/recall trade-offs differ partly for that reason. XGBoost is optional: if it is not installed it is omitted with a message.

## Neural models (TensorFlow/Keras)

All use Adam (learning rate 1e-3), binary cross-entropy, accuracy and AUC metrics, 30 epochs maximum, batch size 32, 15 % of the training data held out for validation, and early stopping on validation loss (patience 5, best weights restored).

| Model | Architecture |
|---|---|
| DNN | Dense 256 (ReLU) → BatchNorm → Dropout 0.35 → Dense 128 (ReLU) → Dropout 0.25 → Dense 1 (sigmoid) |
| RNN | Input as a sequence of length `input_dim` with 1 channel → SimpleRNN 64 (returns sequences) → Dropout 0.3 → SimpleRNN 32 → Dense 64 (ReLU) → Dropout 0.3 → Dense 1 (sigmoid) |
| CNN | Conv1D 64, kernel 5 → MaxPool 2 → Conv1D 32, kernel 3 → global max pool → Dense 64 (ReLU) → Dropout 0.3 → Dense 1 (sigmoid) |

The RNN and CNN treat the feature vector as an ordered sequence. For composition features that order is arbitrary (alphabetical residue order, host block then pathogen block), so any benefit from these architectures should be interpreted with that in mind. TensorFlow is imported lazily; if it is missing or fails, the neural models are skipped with a printed message and the run continues.

## Class imbalance

Test sets range from 50 % to 87.5 % positive (Part 1). RF and SVM use balanced class weights; XGBoost and the neural models do not. Always read results against the majority-class baseline (Part 5).
