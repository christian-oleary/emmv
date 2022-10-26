# EMMV

[![Downloads](https://pepy.tech/badge/emmv)](https://pepy.tech/project/emmv) [![Downloads](https://pepy.tech/badge/emmv/month)](https://pepy.tech/project/emmv) [![Downloads](https://pepy.tech/badge/emmv/week)](https://pepy.tech/project/emmv)

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

If you are using models without a built-in *decision_function* (e.g. Keras or ADTK models), then you need to specify an anomaly scoring function. Please see examples in the examples folder.

## Running Examples

Examples exist for ADTK, Alibi-Detect, PyCaret, PyOD, scikit-learn, and TensorFlow (Keras).

```shell
pip install .
python ./examples/sklearn_example.py
```

## Interpreting scores

- The best model should have the **highest** Excess Mass score
- The best model should have the **lowest** Mass Volume score
- Probably easiest to just use one of the metrics
- Extreme values are possible

## Contact

Please feel free to get in touch at christian.oleary@mtu.ie

## Citation

Christian O'Leary. (2021–2022). EMMV library.

```latex
@Misc{emmv,
author = {Christian O'Leary},
title = {EMMV library},
howpublished = {\url{https://pypi.org/project/emmv/}},
year = {2021--2022}
}
```
