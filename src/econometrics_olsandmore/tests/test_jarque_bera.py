import unittest
import numpy as np
from econometrics_olsandmore.diagnostics import jarque_bera


class TestJarqueBera(unittest.TestCase):
    def test_jb_runs_and_returns_valid_output(self):
        np.random.seed(123)
        e = np.random.normal(size=1000)

        out = jarque_bera(e)

        self.assertIn("jb", out)
        self.assertIn("p_value", out)
        self.assertIn("skewness", out)
        self.assertIn("kurtosis", out)

        self.assertGreaterEqual(out["p_value"], 0.0)
        self.assertLessEqual(out["p_value"], 1.0)

    def test_jb_detects_non_normal_example(self):
        np.random.seed(7)
        # distribución claramente no normal (asimétrica)
        e = np.random.exponential(scale=1.0, size=2000)

        out = jarque_bera(e)

        # normalmente el p-value será muy pequeño
        self.assertTrue(out["p_value"] < 0.05)


if __name__ == "__main__":
    unittest.main()
