import unittest
import numpy as np
from econometrics_olsandmore.regression import ols


class TestValidation(unittest.TestCase):
    def test_raises_on_length_mismatch(self):
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2])  # longitud distinta
        with self.assertRaises(ValueError):
            ols(X, y)

    def test_raises_on_nan_in_X(self):
        X = np.array([[1.0], [np.nan], [3.0]])
        y = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            ols(X, y)

    def test_raises_when_n_le_k(self):
        # n=2, k=2 regresores sin constante -> con constante k=3 => n<=k
        X = np.array([[1.0, 2.0],
                      [3.0, 4.0]])
        y = np.array([1.0, 2.0])
        with self.assertRaises(ValueError):
            ols(X, y)


if __name__ == "__main__":
    unittest.main()
