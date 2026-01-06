import numpy as np
from econometrics_olsandmore.regression import ols


ADF_CRITICAL_VALUES_C = {
    # Valores críticos aproximados (asintóticos) para ADF con constante (regression="c")
    "1%": -3.43,
    "5%": -2.86,
    "10%": -2.57,
}


def adf_test(y, lags=0, regression="c"):
    """
    ADF (Augmented Dickey-Fuller) test manual.

    Modelo (regression="c"):
        Δy_t = a + γ y_{t-1} + Σ_{i=1..lags} δ_i Δy_{t-i} + e_t

    H0: γ = 0 (raíz unitaria)
    H1: γ < 0 (estacionaria)

    Parámetros
    ----------
    y : array-like (n,)
    lags : int >= 0
        Número de rezagos en diferencias.
    regression : str
        Por ahora soportamos solo "c" (con constante).

    Retorna
    -------
    dict con:
      - tau: t-stat de γ
      - gamma: coeficiente γ
      - used_lags
      - n_eff
      - critical_values
      - reject_1, reject_5, reject_10 (según tau < crit)
    """
    if regression != "c":
        raise NotImplementedError("Por ahora solo soportamos regression='c' (con constante).")

    y = np.asarray(y, dtype=float).reshape(-1)
    n = y.size
    if n < (lags + 3):
        raise ValueError("Serie demasiado corta para el número de lags.")

    # Diferencias: Δy_t = y_t - y_{t-1}
    dy = np.diff(y)  # tamaño n-1

    # Construimos la muestra efectiva para t = lags..(n-2) en dy
    # dy_t corresponde a y_{t+1} - y_t; índice dy[t] corresponde al tiempo (t+1)
    start = lags
    end = dy.size  # n-1

    y_dep = dy[start:end]  # Δy_t efectivo

    # Regresor principal: y_{t-1} alineado con Δy_t
    # Para dy[start] = y[start+1]-y[start], el y_{t-1} es y[start]
    y_lag1 = y[start: end]  # misma longitud que y_dep

    X_parts = [y_lag1.reshape(-1, 1)]

    # Rezagos de diferencias: Δy_{t-1}, ..., Δy_{t-lags}
    for i in range(1, lags + 1):
        # Δy_{t-i} alineado: dy[start-i : end-i]
        X_parts.append(dy[start - i: end - i].reshape(-1, 1))

    X = np.hstack(X_parts)  # (n_eff, 1+lags)

    # OLS auxiliar (tu ols() agrega constante automáticamente)
    res = ols(X, y_dep)

    # beta: [const, gamma, delta_1, ..., delta_lags]
    gamma = float(res.beta[1])
    se_gamma = float(res.stderr[1])
    tau = gamma / se_gamma

    crit = ADF_CRITICAL_VALUES_C
    reject_1 = tau < crit["1%"]
    reject_5 = tau < crit["5%"]
    reject_10 = tau < crit["10%"]

    return {
        "tau": float(tau),
        "gamma": float(gamma),
        "used_lags": int(lags),
        "n_eff": int(len(y_dep)),
        "critical_values": crit,
        "reject_1": bool(reject_1),
        "reject_5": bool(reject_5),
        "reject_10": bool(reject_10),
    }
