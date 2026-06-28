# Model Card: Isolation Forest

COM668 Final Year Project | Abdulbosit Abdurazzakov | B00979380

## Model Details

| Property | Value |
|----------|-------|
| Type | IsolationForest (sklearn) |
| Task | Unsupervised anomaly detection |
| n_estimators | 100 |
| contamination | 0.01 |
| Training mode | Semi-supervised (BENIGN-only) |
| Training set | CICIDS2017 Mon–Wed BENIGN flows only |

## Performance (Thursday–Friday test set)

| Metric | Value |
|--------|-------|
| Precision (macro) | ~0.88 |
| Recall (macro) | ~0.79 |
| F1 (macro) | ~0.83 |
| False Positive Rate | ~0.21 |

Note: IF alone has lower precision than RF because it has never seen
attack samples — it infers anomalousness purely from deviation from benign traffic.

## How It Works

Isolation Forest randomly isolates observations by recursively partitioning
the feature space. Anomalous points (attacks) require fewer partitions to isolate
than normal points (benign), resulting in a lower anomaly score.

The decision_function output is negative for anomalies and positive for inliers.
A threshold of 0 is used: score < 0 → ATTACK.

## Intended Use

Supplement to the Random Forest for detecting zero-day or novel attacks
that the supervised model may classify incorrectly due to absence of training labels.
Used in OR-rule ensemble — IF false positives are tolerated because RF corrects most.

## Advantages Over Supervised-Only

- Detects attack types not present in training labels
- Robust to label noise (no label dependency)
- Low contamination (0.01) tuned to minimise FPR while catching clear anomalies

## Contamination Tuning

Evaluated across [0.005, 0.01, 0.02, 0.05, 0.10] in notebook 03.
0.01 selected as the best F1/FPR trade-off (see results/if_contamination_tuning.csv).
