[![tests](https://github.com/Octavio1200/econometrics_olsandmore/actions/workflows/tests.yml/badge.svg)](https://github.com/Octavio1200/econometrics_olsandmore/actions/workflows/tests.yml)

# econometrics_olsandmore

Librería de econometría en Python con implementación manual:
- OLS por API
- Diagnósticos: Durbin–Watson, Breusch–Pagan, White, Jarque–Bera, Breusch–Godfrey
- Multicolinealidad: VIF
- Correlación: Pearson/Spearman
- Raíz unitaria: ADF
- Tests automatizados con unittest

## Instalación (dev)
pip install -e .

## Ejemplo rápido
```python
import pandas as pd
from econometrics_olsandmore.regression import OLSModel
from econometrics_olsandmore.diagnostics import jarque_bera

X = pd.DataFrame({"x1":[1,2,3,4,5], "x2":[2,1,0,1,2]})
y = pd.Series([1,2,1.3,3.75,2.25])

model = OLSModel().fit(X, y)
print(model.result.summary())

jb = jarque_bera(model.result.residuals)
print(jb)