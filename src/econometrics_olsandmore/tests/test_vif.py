import unittest
import numpy as np
import pandas as pd
from econometrics_olsandmore.multicollinearity import vif


class TestVIF(unittest.TestCase):
    def test_vif_low_when_independent(self):
        np.random.seed(0)
        n = 300
        x1 = np.random.normal(size=n)
        x2 = np.random.normal(size=n)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        out = vif(X)

        # Con independencia, VIF cercano a 1 (no exacto)
        self.assertTrue(out.loc["x1", "vif"] < 2.0)
        self.assertTrue(out.loc["x2", "vif"] < 2.0)

    def test_vif_high_when_collinear(self):
        np.random.seed(1)
        n = 300
        x1 = np.random.normal(size=n)
        x2 = 3 * x1 + np.random.normal(scale=0.01, size=n)  # casi colineal
        X = pd.DataFrame({"x1": x1, "x2": x2})

        out = vif(X)

        # Debe ser alto por colinealidad fuerte
        self.assertTrue(out.loc["x1", "vif"] > 10.0)
        self.assertTrue(out.loc["x2", "vif"] > 10.0)


if __name__ == "__main__":
    unittest.main()
