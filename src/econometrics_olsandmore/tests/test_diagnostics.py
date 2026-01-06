import unittest
import numpy as np
from econometrics_olsandmore.diagnostics import durbin_watson


class TestDiagnostics(unittest.TestCase):
    def test_durbin_watson_known_value(self):
        # Residuos simples:
        # e = [1, 2, 3]
        # diff = [1, 1] => sum(diff^2)=2
        # sum(e^2)=1+4+9=14
        # DW = 2/14 = 0.142857...
        e = np.array([1, 2, 3])
        dw = durbin_watson(e)
        self.assertAlmostEqual(dw, 2/14, places=9)


if __name__ == "__main__":
    unittest.main()
