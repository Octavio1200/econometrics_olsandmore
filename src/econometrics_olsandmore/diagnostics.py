import numpy as np


def durbin_watson(residuals):
    """
    Calcula el estadístico Durbin-Watson.

    DW = sum_{t=2..n} (e_t - e_{t-1})^2 / sum_{t=1..n} e_t^2

    Parámetros
    ----------
    residuals : array-like
        Residuos del modelo (n,) o (n,1)

    Retorna
    -------
    float
        Estadístico Durbin-Watson.
    """
    e = np.asarray(residuals).reshape(-1)
    if e.size < 2:
        raise ValueError("Se requieren al menos 2 residuos para calcular Durbin-Watson.")

    diff = np.diff(e)  # e_t - e_{t-1}
    num = np.sum(diff ** 2)
    den = np.sum(e ** 2)

    if den == 0:
        raise ValueError("La suma de cuadrados de residuos es 0; Durbin-Watson indefinido.")

    return float(num / den)


def durbin_watson_from_model(model_or_result):
    """
    Convenience helper: calcula DW directamente desde un OLSModel o OLSResult.

    Acepta:
    - OLSModel (con .result)
    - OLSResult (con .residuals)
    """
    if hasattr(model_or_result, "result") and model_or_result.result is not None:
        res = model_or_result.result.residuals
    elif hasattr(model_or_result, "residuals"):
        res = model_or_result.residuals
    else:
        raise TypeError("Se esperaba un OLSModel con .result o un OLSResult con .residuals")

    return durbin_watson(res)

import numpy as np
from scipy import stats
from econometrics_olsandmore.regression import ols


def breusch_pagan(residuals, X):
    """
    Prueba Breusch-Pagan para heterocedasticidad.

    Parámetros
    ----------
    residuals : array-like
        Residuos del modelo principal (n,) o (n,1)
    X : array-like o DataFrame
        Matriz de regresores del modelo principal (SIN constante).
        (n, k)

    Retorna
    -------
    dict
        {
          "lm": estadístico LM,
          "df": grados de libertad (k),
          "p_value": p-valor Chi-cuadrado,
          "aux_r2": R^2 de la regresión auxiliar
        }
    """
    e = np.asarray(residuals).reshape(-1, 1)
    n = e.shape[0]

    # y auxiliar = e^2
    y_aux = (e ** 2).reshape(-1)

    # Regresión auxiliar: e^2 ~ X
    aux = ols(X, y_aux)
    r2_aux = aux.r2

    # k = número de regresores sin contar constante
    # En aux.k sí incluye const, por eso df = aux.k - 1
    df = aux.k - 1

    lm = n * r2_aux
    p_value = 1 - stats.chi2.cdf(lm, df=df)

    return {
        "lm": float(lm),
        "df": int(df),
        "p_value": float(p_value),
        "aux_r2": float(r2_aux),
    }

import numpy as np
from scipy import stats
from econometrics_olsandmore.regression import ols


def _as_2d_numeric_array(X):
    """Convierte X a np.array 2D (n,k). Acepta DataFrame o array."""
    try:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
    except Exception:
        X_arr = np.asarray(X, dtype=float)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    return X_arr


def white_test(residuals, X):
    """
    White test para heterocedasticidad.

    Idea:
    1) y_aux = e^2
    2) Z = [X, X^2, (x_i * x_j) para i<j]  (SIN constante)
    3) Ajuste auxiliar: e^2 ~ Z
    4) LM = n * R^2_aux ~ Chi2(df) con df = #regresores auxiliares (sin const)

    Parámetros
    ----------
    residuals : array-like (n,) o (n,1)
    X : array-like o DataFrame (n,k) SIN constante

    Retorna
    -------
    dict: {"lm","df","p_value","aux_r2","n_features_aux"}
    """
    e = np.asarray(residuals).reshape(-1, 1)
    n = e.shape[0]

    X0 = _as_2d_numeric_array(X)
    if X0.shape[0] != n:
        raise ValueError(f"X tiene {X0.shape[0]} filas pero residuals tiene {n}")

    k = X0.shape[1]

    # Construir Z = [X, X^2, cross products]
    parts = [X0, X0 ** 2]

    # interacciones cruzadas
    cross = []
    for i in range(k):
        for j in range(i + 1, k):
            cross.append((X0[:, i] * X0[:, j]).reshape(-1, 1))
    if cross:
        parts.append(np.hstack(cross))

    Z = np.hstack(parts)  # (n, m)

    # y auxiliar
    y_aux = (e ** 2).reshape(-1)

    aux = ols(Z, y_aux)
    r2_aux = aux.r2

    # df = m (porque en ols se agrega constante internamente)
    m = Z.shape[1]
    lm = n * r2_aux
    p_value = 1 - stats.chi2.cdf(lm, df=m)

    return {
        "lm": float(lm),
        "df": int(m),
        "p_value": float(p_value),
        "aux_r2": float(r2_aux),
        "n_features_aux": int(m),
    }

import numpy as np
from scipy import stats

def jarque_bera(residuals):
    """
    Jarque-Bera test de normalidad.

    JB = (n/6) * [ S^2 + ( (K - 3)^2 / 4 ) ]
    donde:
      S = skewness (asimetría)
      K = kurtosis (curtosis, NO excesiva)

    Bajo H0 (normalidad), JB ~ Chi2(df=2)
    """
    e = np.asarray(residuals).reshape(-1)
    n = e.size
    if n < 3:
        raise ValueError("Se requieren al menos 3 residuos para Jarque-Bera.")

    # centrar
    c = e - e.mean()

    m2 = np.mean(c ** 2)
    if m2 == 0:
        raise ValueError("Varianza 0 en residuos; Jarque-Bera indefinido.")

    m3 = np.mean(c ** 3)
    m4 = np.mean(c ** 4)

    S = m3 / (m2 ** 1.5)     # skewness
    K = m4 / (m2 ** 2.0)     # kurtosis (normal => 3)

    jb = (n / 6.0) * (S**2 + ((K - 3.0) ** 2) / 4.0)
    p_value = 1 - stats.chi2.cdf(jb, df=2)

    return {
        "jb": float(jb),
        "p_value": float(p_value),
        "skewness": float(S),
        "kurtosis": float(K),
        "n": int(n),
    }

import numpy as np
from scipy import stats
from econometrics_olsandmore.regression import ols


def breusch_godfrey(residuals, X, lags=1):
    """
    Breusch-Godfrey test para autocorrelación de orden p.

    Auxiliar:
        e_t ~ X_t + e_{t-1} + ... + e_{t-p}

    LM = n_eff * R^2_aux ~ Chi2(df=p)

    Parámetros
    ----------
    residuals : array-like (n,) o (n,1)
        Residuos del modelo principal.
    X : array-like o DataFrame (n,k) SIN constante
        Regresores originales.
    lags : int
        Orden p de autocorrelación.

    Retorna
    -------
    dict: {"lm","df","p_value","aux_r2","n_eff"}
    """
    e = np.asarray(residuals).reshape(-1)
    if lags < 1:
        raise ValueError("lags debe ser >= 1")
    n = e.size
    if n <= lags + 1:
        raise ValueError("No hay suficientes observaciones para esos lags.")

    # X a numpy 2D
    try:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X0 = X.to_numpy(dtype=float)
        else:
            X0 = np.asarray(X, dtype=float)
    except Exception:
        X0 = np.asarray(X, dtype=float)

    if X0.ndim == 1:
        X0 = X0.reshape(-1, 1)

    if X0.shape[0] != n:
        raise ValueError(f"X tiene {X0.shape[0]} filas pero residuals tiene {n}")

    # Construcción de la muestra efectiva: t = lags..n-1
    # y_aux = e_t
    y_aux = e[lags:]

    # Regresores auxiliares: X_t recortado + rezagos de e
    X_t = X0[lags:, :]  # (n-lags, k)

    lag_cols = []
    for p in range(1, lags + 1):
        lag_cols.append(e[lags - p: n - p].reshape(-1, 1))  # (n-lags,1)

    E_lags = np.hstack(lag_cols)  # (n-lags, lags)

    Z = np.hstack([X_t, E_lags])  # (n-lags, k+lags)

    aux = ols(Z, y_aux)
    r2_aux = aux.r2

    n_eff = y_aux.shape[0]
    lm = n_eff * r2_aux
    df = lags
    p_value = 1 - stats.chi2.cdf(lm, df=df)

    return {
        "lm": float(lm),
        "df": int(df),
        "p_value": float(p_value),
        "aux_r2": float(r2_aux),
        "n_eff": int(n_eff),
    }
