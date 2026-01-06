import unittest
import pandas as pd
from econometrics_olsandmore.correlation import correlation_matrix


class TestCorrelation(unittest.TestCase):
    def test_pearson_identity(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
        corr = correlation_matrix(df, method="pearson")
        self.assertAlmostEqual(float(corr.loc["a", "b"]), 1.0, places=9)

    def test_spearman_monotonic(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
        corr = correlation_matrix(df, method="spearman")
        self.assertAlmostEqual(float(corr.loc["a", "b"]), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
