# Model Card: OR-Rule Ensemble

COM668 Final Year Project | Abdulbosit Abdurazzakov | B00979380

## Overview

The ensemble combines the Random Forest (supervised) and Isolation Forest
(semi-supervised) predictions using the OR rule:

```
final_decision = ATTACK  if  RF == ATTACK  OR  IF == ATTACK
               = BENIGN   otherwise
```

## Performance (Thursday–Friday test set)

| Metric | RF only | IF only | Ensemble (OR) |
|--------|---------|---------|---------------|
| Precision | 0.9987 | 0.88 | ~0.9941 |
| Recall | 0.9963 | 0.79 | ~0.9991 |
| F1 | 0.9975 | 0.83 | ~0.9966 |
| FPR | 0.0003 | 0.21 | ~0.0011 |

The ensemble trades a small precision reduction for a recall improvement —
the correct trade-off for IDS where missing an attack (FN) is more costly
than a false alarm (FP).

## Rule Comparison

| Rule | Recall | Precision | Use Case |
|------|--------|-----------|----------|
| OR | Highest | Moderate | Default IDS (miss fewer attacks) |
| AND | Lower | Highest | Low-noise environments |
| Weighted vote | Tunable | Tunable | Custom α parameter |

## Rationale for OR Rule

In IDS deployment:
- False Negative (missed attack) → potential breach, data loss, reputational damage
- False Positive (false alarm) → analyst review overhead, manageable at low FPR

The OR rule maximises detection rate while keeping FPR acceptable (~0.1%),
aligning with SMART objective O3: FPR ≤ 5%.
