import unittest
import numpy as np
from econometrics_olsandmore.regression import ols
from econometrics_olsandmore.diagnostics import white_test


class TestWhite(unittest.TestCase):
    def test_white_runs_and_returns_valid_output(self):
        np.random.seed(1)
        n = 200
        X = np.column_stack([
            np.arange(1, n + 1).astype(float),
            np.random.normal(size=n)
        ])

        # y con ruido (no buscamos “detectar” sí o sí, solo que corra bien)
        y = 3 + 2 * X[:, 0] - 1.5 * X[:, 1] + np.random.normal(size=n)

        res = ols(X, y)
        out = white_test(res.residuals, X)

        self.assertIn("lm", out)
        self.assertIn("df", out)
        self.assertIn("p_value", out)
        self.assertGreaterEqual(out["p_value"], 0.0)
        self.assertLessEqual(out["p_value"], 1.0)
        self.assertGreaterEqual(out["df"], 1)


if __name__ == "__main__":
    unittest.main()
