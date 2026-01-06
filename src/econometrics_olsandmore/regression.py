import numpy as np
import pandas as pd
from scipy import stats

def _check_finite(arr, name="array"):
    arr = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contiene NaN o infinitos.")
    return arr

class OLSResult:
    def __init__(self, X, y, beta, stderr, y_hat, residuals, r2, r2_adj, n, k, feature_names):
        self.X = X                  # matriz con constante incluida
        self.y = y                  # (n, 1)
        self.beta = beta            # (k, 1)
        self.stderr = stderr        # (k, 1)
        self.y_hat = y_hat          # (n, 1)
        self.residuals = residuals  # (n, 1)
        self.r2 = float(r2)
        self.r2_adj = float(r2_adj)
        self.n = int(n)
        self.k = int(k)
        self.feature_names = feature_names  # lista de longitud k

    def summary(self):
        t_stats = self.beta / self.stderr
        df = self.n - self.k
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df))

        out = pd.DataFrame(
            {
                "coef": self.beta,
                "std_err": self.stderr,
                "t": t_stats,
                "P>|t|": p_values,
            },
            index=self.feature_names
        )
        return out

    def predict(self, X_new):
        Xn, _ = _prepare_X(X_new, add_constant=True)
        return (Xn @ self.beta).flatten()


class OLSModel:
    """
    API estilo clásico para regresión OLS:
        model = OLSModel().fit(X, y)
        model.predict(X_new)
        model.result.summary()
    """
    def __init__(self):
        self.result = None

    def fit(self, X, y):
        self.result = ols(X, y)
        return self

    def predict(self, X_new):
        if self.result is None:
            raise ValueError("Primero entrena el modelo con .fit(X, y)")
        return self.result.predict(X_new)


def _prepare_X(X, add_constant=True):
    """
    Convierte X a ndarray y obtiene nombres de variables si vienen en DataFrame.
    """
    feature_names = None

    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_arr = X.to_numpy()
        X_arr = _check_finite(X_arr, name="X")

    else:
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        feature_names = [f"x{i}" for i in range(X_arr.shape[1])]

    if add_constant:
        X_arr = np.column_stack((np.ones(X_arr.shape[0]), X_arr))
        feature_names = ["const"] + feature_names

    return X_arr, feature_names


def _prepare_y(y, n_expected):
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_arr = np.asarray(y).reshape(-1)
        y_arr = _check_finite(y_arr, name="y").reshape(-1)

    else:
        y_arr = np.asarray(y).reshape(-1)

    if y_arr.shape[0] != n_expected:
        raise ValueError(f"y tiene {y_arr.shape[0]} filas pero X tiene {n_expected}")

    return y_arr # (n,)


def ols(X, y):
    """
    OLS manual (MCO) con constante.
    X: array-like (n, k) o DataFrame
    y: array-like (n,) o Series
    """
    X_mat, feature_names = _prepare_X(X, add_constant=True)
    X_mat = _check_finite(X_mat, name="X")  

    n, k = X_mat.shape

    y_vec = _prepare_y(y, n_expected=n)
    y_vec = _check_finite(y_vec, name="y").reshape(-1)

    # Debe haber más observaciones que parámetros (para n-k > 0)
    if n <= k:
        raise ValueError(f"No hay suficientes observaciones: n={n} y parámetros k={k} (incluye constante).")

    # beta = (X'X)^(-1) X'y  -> beta 1D (k,)
    XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
    beta = XtX_inv @ (X_mat.T @ y_vec)

    # predicción y residuos -> 1D (n,)
    y_hat = X_mat @ beta
    residuals = y_vec - y_hat

    # varianza del error (escalar)
    sigma2 = (residuals @ residuals) / (n - k)

    # var(beta) = sigma2*(X'X)^(-1) -> stderr 1D (k,)
    var_beta = sigma2 * XtX_inv
    stderr = np.sqrt(np.diag(var_beta))

    # R2 y R2 ajustado (escalares)
    ss_total = ((y_vec - y_vec.mean()) ** 2).sum()
    ss_res = (residuals ** 2).sum()
    r2 = 1 - ss_res / ss_total
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k)

    return OLSResult(X_mat, y_vec, beta, stderr, y_hat, residuals, r2, r2_adj, n, k, feature_names)

