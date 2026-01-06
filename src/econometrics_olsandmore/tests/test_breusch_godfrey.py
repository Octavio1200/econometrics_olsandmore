import unittest
import numpy as np
from econometrics_olsandmore.regression import ols
from econometrics_olsandmore.diagnostics import breusch_godfrey


class TestBreuschGodfrey(unittest.TestCase):
    def test_bg_runs(self):
        np.random.seed(0)
        n = 400
        X = np.arange(1, n + 1).reshape(-1, 1).astype(float)

        # Generamos un error AR(1) para que típicamente haya autocorrelación
        e = np.zeros(n)
        for t in range(1, n):
            e[t] = 0.8 * e[t - 1] + np.random.normal()

        y = 2 * X.flatten() + e

        res = ols(X, y)
        out = breusch_godfrey(res.residuals, X, lags=1)

        self.assertIn("lm", out)
        self.assertIn("p_value", out)
        self.assertGreaterEqual(out["p_value"], 0.0)
        self.assertLessEqual(out["p_value"], 1.0)

        # Con AR(1) fuerte, normalmente p_value será pequeño (no siempre 100% garantizado)
        self.assertTrue(out["p_value"] < 0.05)


if __name__ == "__main__":
    unittest.main()
