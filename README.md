[![tests](https://github.com/Octavio1200/econometrics_olsandmore/actions/workflows/tests.yml/badge.svg)](https://github.com/Octavio1200/econometrics_olsandmore/actions/workflows/tests.yml)

# econometrics_olsandmore

Librería de econometría en Python con implementación **manual** para:
- **OLS**
- Diagnósticos: **Durbin–Watson**, **Breusch–Godfrey**, **Breusch–Pagan**, **White**, **Jarque–Bera**
- **VIF** (multicolinealidad)
- Matriz de correlación: **Pearson** / **Spearman**
- Raíz unitaria: **ADF (Augmented Dickey-Fuller)**
- **Tests automatizados** + **CI con GitHub Actions**
- Publicada en **PyPI**

---

## Instalación

### Desde PyPI
```bash
pip install econometrics_olsandmore
```

### Desde GitHub
```bash
pip install git+https://github.com/Octavio1200/econometrics_olsandmore.git
```

---

## Uso rápido

### OLS (API tipo sklearn)
```python
import pandas as pd
from econometrics_olsandmore.regression import OLSModel

X = pd.DataFrame({"x1": [1,2,3,4,5], "x2": [2,1,0,1,2]})
y = pd.Series([1,2,1.3,3.75,2.25])

model = OLSModel().fit(X, y)
print(model.result.summary())

y_pred = model.predict(X)
print("Predicciones:", y_pred)
```

### Diagnósticos sobre residuos
```python
from econometrics_olsandmore.diagnostics import (
    durbin_watson_from_model,
    breusch_pagan,
    white_test,
    jarque_bera,
    breusch_godfrey,
)

dw = durbin_watson_from_model(model)
print("Durbin–Watson:", dw)

bp = breusch_pagan(model.result.residuals, X)
print("Breusch–Pagan:", bp)

wt = white_test(model.result.residuals, X)
print("White:", wt)

jb = jarque_bera(model.result.residuals)
print("Jarque–Bera:", jb)

bg = breusch_godfrey(model.result.residuals, X, lags=1)
print("Breusch–Godfrey:", bg)
```

### Correlación y VIF
```python
from econometrics_olsandmore.correlation import correlation_matrix
from econometrics_olsandmore.multicollinearity import vif

print(correlation_matrix(X, method="pearson"))
print(vif(X))
```

### ADF (raíz unitaria)
```python
import numpy as np
from econometrics_olsandmore.unit_root import adf_test

np.random.seed(0)
y_rw = np.cumsum(np.random.normal(size=500))  # random walk
print(adf_test(y_rw, lags=1))
```

---

## Desarrollo

### Ejecutar tests
```bash
python -m unittest discover -s src -v
```

### Instalación editable (contributors)
```bash
pip install -e .
```

---

## Estructura del proyecto

```text
econometrics_olsandmore/
├─ src/econometrics_olsandmore/
│  ├─ regression.py
│  ├─ diagnostics.py
│  ├─ unit_root.py
│  ├─ multicollinearity.py
│  ├─ correlation.py
│  └─ tests/
├─ .github/workflows/tests.yml
├─ pyproject.toml
├─ requirements.txt
└─ README.md
```

---

## Roadmap (opcional)
- Soporte para series temporales y selección automática de rezagos

---

## Licencia
MIT

---

## Autor
Octavio Pacheco — GitHub: https://github.com/Octavio1200
