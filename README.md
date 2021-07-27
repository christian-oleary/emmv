# EMMV

Implementation of EM/MV metrics based on N. Goix et al.

This is a means of evaluating anomaly detection models without anomaly labels

## Installation

```shell
pip install emmv
```

## Example Use

```python
from emmv import emmv_scores

test_scores = emmv_scores(model, features)
```

- Where 'model' is your **trained** scikit-learn, PyOD, or PyCaret model
- Where 'features' is a 2D DataFrame of features (the *X* matrix)

Example resulting object:

```json
{ 
    "em": 0.77586,
    "mv": 0.25367
}
```

## Running Examples

```shell
pip install .
python ./examples/sklearn_example.py
```

## Interpreting scores

- The best model should have the **highest** Excess Mass score
- The best model should have the **lowest** Mass Volume score
- Probably easiest to just use one of the metrics
- Extreme values are possible
