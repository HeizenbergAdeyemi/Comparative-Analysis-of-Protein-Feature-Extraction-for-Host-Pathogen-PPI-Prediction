# Part 3 — Machine learning and deep learning

Six classifiers are defined. Classical models run on the pair feature vectors directly; the three neural architectures are defined in `models.py` and trained by `train_model.py`.

> **TODO (author):** add the hyperparameters actually used (below are the things to record, not values) and the train/test split protocol (ratio, stratification, random seed).

| Model | Family | Library | Hyperparameters to record |
|---|---|---|---|
| Random Forest (RF) | Bagged decision trees | scikit-learn | `n_estimators`, `max_depth`, `class_weight`, `random_state` |
| SVM | Maximum-margin kernel classifier | scikit-learn | kernel, `C`, `gamma`, `probability` (needed for ROC) |
| XGBoost | Gradient-boosted trees | xgboost | `n_estimators`, `learning_rate`, `max_depth`, `scale_pos_weight` |
| DNN | Fully connected network | TensorFlow/Keras | layers, units, dropout, optimiser, epochs, batch size |
| RNN | Recurrent network over the feature sequence | TensorFlow/Keras | cell type (LSTM/GRU), units, dropout |
| CNN | 1-D convolution over the feature vector | TensorFlow/Keras | filters, kernel size, pooling, dense head |

## Random Forest
An ensemble of decision trees trained on bootstrap samples with random feature subsets, averaging their votes. Robust to unscaled features and needs little tuning; probabilities come from the vote fraction.

## Support Vector Machine
Finds the maximum-margin separating surface, usually with an RBF kernel. Sensitive to feature scaling, so features should be standardised. A probability estimate (Platt scaling) is needed to draw an ROC curve.

## XGBoost
Sequentially boosted trees that correct previous errors, with regularisation. Usually the strongest tabular learner here; in this study it gave the top-ranked result (by AUC, then F1) in four of the five organism groups; Random Forest was top for SARS-CoV-2.

## Deep Neural Network (DNN)
Stacked dense layers with non-linear activations, dropout and a sigmoid output. Treats the vector as unordered.

## Recurrent Neural Network (RNN)
Reads the feature vector as a sequence of steps (LSTM/GRU). Only meaningful if the feature order carries structure, so its benefit on composition vectors should be justified rather than assumed.

## Convolutional Neural Network (CNN)
1-D convolutions detect local patterns along the vector, followed by pooling and a dense classifier. Same caveat on ordering as the RNN.

## Class imbalance
Test sets range from 50 % to 87.5 % positive (Part 1). Record whether class weights, resampling or neither were used, since this directly affects the precision/recall trade-offs in the results.
