import unittest
import numpy as np
import pandas as pd
from econometrics_olsandmore.regression import ols, OLSModel


class TestOLS(unittest.TestCase):
    def test_manual_ols_perfect_line(self):
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])

        result = ols(X, y)

        # beta = [const, slope]
        self.assertAlmostEqual(float(result.beta[0]), 0.0, places=6)
        self.assertAlmostEqual(float(result.beta[1]), 2.0, places=6)
        self.assertAlmostEqual(result.r2, 1.0, places=6)

    def test_feature_names_with_dataframe(self):
        X = pd.DataFrame({"edad": [1, 2, 3, 4, 5]})
        y = pd.Series([2, 4, 6, 8, 10])

        model = OLSModel().fit(X, y)
        summary = model.result.summary()

        self.assertIn("const", summary.index)
        self.assertIn("edad", summary.index)


if __name__ == "__main__":
    unittest.main()
