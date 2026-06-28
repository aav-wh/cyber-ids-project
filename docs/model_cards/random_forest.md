# Model Card: Random Forest Classifier

COM668 Final Year Project | Abdulbosit Abdurazzakov | B00979380

## Model Details

| Property | Value |
|----------|-------|
| Type | RandomForestClassifier (sklearn) |
| Task | Binary classification (ATTACK / BENIGN) |
| n_estimators | 150 |
| class_weight | balanced |
| random_state | 42 |
| Training set | CICIDS2017 Mon–Wed (2,273,097 flows) |

## Performance (Thursday–Friday test set)

| Metric | Value |
|--------|-------|
| Accuracy | ~0.9994 |
| Precision (macro) | ~0.9987 |
| Recall (macro) | ~0.9963 |
| F1 (macro) | ~0.9975 |
| AUC-ROC | ~0.9999 |
| False Positive Rate | ~0.0003 |
| False Negative Rate | ~0.0074 |
| Avg inference latency | ~1.5ms |

## Intended Use

Supervised classification of pre-extracted CICIDS2017 network flow features.
The model outputs (a) a binary ATTACK/BENIGN label and (b) per-class probabilities
used by the ensemble and threshold optimisation modules.

## Limitations

- Trained on CICIDS2017 (2017 traffic); may not generalise to novel attack types.
- Temporal drift: performance degrades on Friday data (see notebook 05).
- Requires 78 pre-computed flow features — cannot operate on raw PCAP directly.

## Feature Importance (top 5 by MDI)

1. Destination Port
2. Flow Duration
3. Fwd Packet Length Max
4. Bwd Packet Length Max
5. Flow IAT Mean
