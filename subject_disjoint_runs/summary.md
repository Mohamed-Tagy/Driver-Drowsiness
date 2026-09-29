# Subject-disjoint results

| Model / training set / online aug. | Seeds | Test acc. (mean ± SD) | Min–max | Macro F1 | Closed→open rate | Val acc. |
|---|---|---|---|---|---|---|
| Kaggle image-level / vit-base / base / strong | [42] | 99.19 ± nan | 99.19–99.19 | 99.19 ± nan | 0.83 ± nan | 99.17 ± nan |
| subject-disjoint / vit-base / base / strong | [42, 43, 44] | 98.41 ± 0.65 | 97.71–98.99 | 98.41 ± 0.65 | 1.81 ± 1.00 | 98.94 ± 0.04 |

## Paired McNemar tests (same split, same test images)

| A | B | Seed | A right / B wrong | A wrong / B right | p (exact) |
|---|---|---|---|---|---|
| (needs two configurations run on the same split) |||||| 
