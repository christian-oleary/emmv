# EMMV

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/christian-oleary/emmv/graphs/commit-activity)  [![Downloads](https://pepy.tech/badge/emmv)](https://pepy.tech/project/emmv) [![Downloads](https://pepy.tech/badge/emmv/month)](https://pepy.tech/project/emmv) [![Downloads](https://pepy.tech/badge/emmv/week)](https://pepy.tech/project/emmv)

Implementation of EM/MV metrics based on [N. Goix et al.](https://github.com/ngoix/EMMV_benchmarks)

This is a means of evaluating anomaly detection models without anomaly labels.

## Installation

```bash
pip install emmv
```

## Example Use

```python
from emmv import emmv_scores

test_scores = emmv_scores(model, features)
```

- Where 'model' is your **trained** scikit-learn/PyOD/PyCaret/etc. model
- Where 'features' is a 2D DataFrame of features (the *X* matrix)

Example resulting object:

```json
{
    "em": 0.77586,
    "mv": 0.25367
}
```

If you are using models without a built-in *decision_function* (e.g. Keras or ADTK models), then you need to specify an anomaly scoring function. Please see examples in [./src/emmv/examples](https://github.com/christian-oleary/emmv/tree/master/src/emmv/examples) folder.

## Running Examples

Examples exist for ADTK, Alibi-Detect, PyCaret, PyOD, scikit-learn, and TensorFlow (Keras).

```bash
pip install -r requirements.txt
pip install -e .
python src/emmv/examples/adtk_example.py          # For an ADTK example
python src/emmv/examples/alibi_detect_example.py  # For a Alibi Detect example
python src/emmv/examples/keras_example.py         # For a Keras example
python src/emmv/examples/pycaret_example.py       # For a PyCaret example
python src/emmv/examples/pyod_example.py          # For a PyOD example
python src/emmv/examples/sklearn_example.py       # For a scikit-learn example
```

## Interpreting scores

- The best model should have the **highest** Excess Mass score
- The best model should have the **lowest** Mass Volume score
- Probably easiest to just use one of the metrics
- Extreme values are possible

## Contact

Please feel free to get in touch at <christian.oleary@mtu.ie>

## Citation

Christian O'Leary (2024) EMMV library

```latex
@Misc{emmv,
    author = {Christian O'Leary},
    title = {EMMV library},
    howpublished = {\url{https://pypi.org/project/emmv/}},
    year = {2021}
}
```

## Development

```bash
conda create -n emmv python=3.10 -y
conda activate emmv
pip install -r ./tests/requirements.txt
pip install -e .
conda install pre-commit
pre-commit install
pytest
pylint
```
