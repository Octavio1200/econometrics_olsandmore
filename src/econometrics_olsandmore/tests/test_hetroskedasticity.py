import unittest
import numpy as np
from econometrics_olsandmore.regression import ols
from econometrics_olsandmore.diagnostics import breusch_pagan


class TestHeteroskedasticity(unittest.TestCase):
    def test_breusch_pagan_runs(self):
        # Datos: y = 2x + ruido
        np.random.seed(0)
        n = 200
        X = np.arange(1, n + 1).reshape(-1, 1).astype(float)
        noise = np.random.normal(0, 1, size=n)
        y = 2 * X.flatten() + noise

        res = ols(X, y)

        out = breusch_pagan(res.residuals, X)

        # Solo validamos que regresa estructura correcta
        self.assertIn("lm", out)
        self.assertIn("df", out)
        self.assertIn("p_value", out)
        self.assertIn("aux_r2", out)

        self.assertGreaterEqual(out["p_value"], 0.0)
        self.assertLessEqual(out["p_value"], 1.0)
        self.assertGreaterEqual(out["df"], 1)


if __name__ == "__main__":
    unittest.main()
