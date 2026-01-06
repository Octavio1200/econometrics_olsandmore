import unittest
import numpy as np
from econometrics_olsandmore.unit_root import adf_test


class TestADF(unittest.TestCase):
    def test_adf_random_walk_not_reject(self):
        np.random.seed(0)
        n = 600
        e = np.random.normal(size=n)
        y = np.cumsum(e)  # random walk (típicamente no estacionaria)

        out = adf_test(y, lags=1, regression="c")

        # En random walk, normalmente NO rechaza al 5%
        self.assertFalse(out["reject_5"])

    def test_adf_stationary_reject(self):
        np.random.seed(1)
        n = 600
        e = np.random.normal(size=n)
        y = np.zeros(n)
        phi = 0.6  # estacionario
        for t in range(1, n):
            y[t] = phi * y[t - 1] + e[t]

        out = adf_test(y, lags=1, regression="c")

        # En AR(1) estacionario, normalmente rechaza al 5%
        self.assertTrue(out["reject_5"])


if __name__ == "__main__":
    unittest.main()
