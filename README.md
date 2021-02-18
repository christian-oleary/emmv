# EMMV

Implementation of EM/MV metrics based on N. Goix et al.

This is a means of evaluating anomaly detection models without anomaly labels

## Example Use

```python
test_scores = emmv_scores(model, features)
```

- Where 'model' is your **trained** scikit-learn model
- Where 'features' is a 2D dataframe of features (the *X* matrix)

Example resulting object:

```json
{ 
    "em": 0.77586,
    "mv": 0.25367
}
```

## Interpreting scores

- The best model should have the highest Excess Mass score
- The best model should have the lowest Mass Volume score
