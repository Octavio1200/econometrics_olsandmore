import numpy as np
import pandas as pd
from econometrics_olsandmore.regression import ols


def _to_df(X):
    if isinstance(X, pd.DataFrame):
        return X.copy()
    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    cols = [f"x{i}" for i in range(X_arr.shape[1])]
    return pd.DataFrame(X_arr, columns=cols)


def vif(X):
    """
    Calcula VIF para cada regresor en X (SIN constante).

    Parámetros
    ----------
    X : array-like o DataFrame (n, k)
        Matriz de variables explicativas sin constante.

    Retorna
    -------
    pd.DataFrame con columnas:
        - variable
        - vif
        - r2_aux
    """
    df = _to_df(X)

    # asegurar numérico
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="raise")

    k = df.shape[1]
    out = []

    for j, col in enumerate(df.columns):
        y = df[col].to_numpy()
        X_others = df.drop(columns=[col])

        # Si no hay otras variables, VIF no aplica
        if X_others.shape[1] == 0:
            out.append({"variable": col, "vif": 1.0, "r2_aux": 0.0})
            continue

        aux = ols(X_others, y)
        r2_j = aux.r2

        # evitar división por cero por redondeo
        denom = max(1e-12, 1.0 - r2_j)
        vif_j = 1.0 / denom

        out.append({"variable": col, "vif": float(vif_j), "r2_aux": float(r2_j)})

    return pd.DataFrame(out).set_index("variable")

import numpy as np
import pandas as pd
from econometrics_olsandmore.regression import ols


def _to_df(X):
    if isinstance(X, pd.DataFrame):
        return X.copy()
    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    cols = [f"x{i}" for i in range(X_arr.shape[1])]
    return pd.DataFrame(X_arr, columns=cols)


def vif(X):
    """
    Calcula VIF para cada regresor en X (SIN constante).

    VIF_j = 1 / (1 - R^2_j), donde R^2_j viene de la regresión auxiliar:
        x_j ~ X_{-j}

    Parámetros
    ----------
    X : array-like o DataFrame (n, k)
        Matriz de variables explicativas sin constante.

    Retorna
    -------
    pd.DataFrame indexado por variable, con:
        - vif
        - r2_aux
    """
    df = _to_df(X)

    # asegurar numérico
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="raise")

    out = []

    for col in df.columns:
        y = df[col].to_numpy()
        X_others = df.drop(columns=[col])

        # Si no hay otras variables, VIF no aplica (lo dejamos en 1)
        if X_others.shape[1] == 0:
            out.append({"variable": col, "vif": 1.0, "r2_aux": 0.0})
            continue

        aux = ols(X_others, y)
        r2_j = aux.r2

        denom = max(1e-12, 1.0 - r2_j)  # evita división por cero numérica
        vif_j = 1.0 / denom

        out.append({"variable": col, "vif": float(vif_j), "r2_aux": float(r2_j)})

    return pd.DataFrame(out).set_index("variable")

