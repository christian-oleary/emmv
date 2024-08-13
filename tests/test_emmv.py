"""Unit tests for the EMMV module."""

import sys
import unittest

import pytest

from emmv.examples.adtk_example import run as run_adtk_example
from emmv.examples.alibi_detect_example import run as run_alibi_detect_example
from emmv.examples.keras_example import run as run_keras_example
from emmv.examples.pycaret_example import run as run_pycaret_example
from emmv.examples.pyod_example import run as run_pyod_example
from emmv.examples.sklearn_example import run as run_sklearn_example


class EmmvTests(unittest.TestCase):
    """Unit tests for the EMMV module."""

    def test_adtk_example(self):
        """Test the emmv_scores function with ADTK."""
        run_adtk_example()

    @pytest.mark.skipif(sys.version_info >= (3, 11), reason='Alibi/TF mismatch for Python 11+')
    def test_alibi_detect_example(self):
        """Test the emmv_scores function with Alibi Detect."""
        run_alibi_detect_example()

    def test_keras_example(self):
        """Test the emmv_scores function with Keras."""
        run_keras_example()

    def test_pycaret_example(self):
        """Test the emmv_scores function with PyCaret."""
        run_pycaret_example()

    def test_pyod_example(self):
        """Test the emmv_scores function with PyOD."""
        run_pyod_example()

    def test_sklearn_example(self):
        """Test the emmv_scores function with scikit-learn."""
        run_sklearn_example()


if __name__ == '__main__':
    unittest.main()
