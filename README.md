# econometrics_olsandmore

Librería de econometría en Python (implementación manual) con:
- OLS (API tipo sklearn)
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


## 3) Agrega GitHub Actions (CI)
Crea esta carpeta y archivo:

```powershell
New-Item -ItemType Directory -Force .github/workflows
New-Item .github/workflows/tests.yml -ItemType File
