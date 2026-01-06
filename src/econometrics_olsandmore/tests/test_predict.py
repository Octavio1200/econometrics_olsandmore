import unittest
import numpy as np
import pandas as pd
from econometrics_olsandmore.regression import OLSModel, ols


class TestPredict(unittest.TestCase):
    def test_predict_raises_if_not_fit(self):
        model = OLSModel()
        X_new = np.array([[1], [2], [3]])
        with self.assertRaises(ValueError):
            model.predict(X_new)

    def test_predict_output_length_numpy(self):
        X = np.array([[1], [2], [3], [4], [5]], dtype=float)
        y = np.array([2, 4, 6, 8, 10], dtype=float)

        model = OLSModel().fit(X, y)
        X_new = np.array([[10], [11], [12]], dtype=float)
        y_pred = model.predict(X_new)

        self.assertEqual(len(y_pred), 3)

    def test_predict_dataframe_consistency(self):
        X = pd.DataFrame({"edad": [1, 2, 3, 4, 5]})
        y = pd.Series([2, 4, 6, 8, 10])

        model = OLSModel().fit(X, y)

        X_new = pd.DataFrame({"edad": [10, 11, 12]})
        y_pred = model.predict(X_new)

        self.assertEqual(len(y_pred), 3)
        self.assertTrue(np.all(np.isfinite(y_pred)))


if __name__ == "__main__":
    unittest.main()
