import numpy as np
import pandas as pd


def _to_df(X):
    if isinstance(X, pd.DataFrame):
        return X.copy()
    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    cols = [f"x{i}" for i in range(X_arr.shape[1])]
    return pd.DataFrame(X_arr, columns=cols)


def _pearson_corr_df(df: pd.DataFrame) -> pd.DataFrame:
    X = df.to_numpy(dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    n = X.shape[0]
    # cov = (Xc'Xc)/(n-1)
    cov = (Xc.T @ Xc) / (n - 1)
    std = np.sqrt(np.diag(cov))
    denom = np.outer(std, std)
    corr = cov / denom
    return pd.DataFrame(corr, index=df.columns, columns=df.columns)


def correlation_matrix(X, method="pearson"):
    """
    Matriz de correlación.

    method:
      - "pearson"
      - "spearman" (rank transform + Pearson)

    Retorna DataFrame.
    """
    df = _to_df(X)

    # Validación básica: numérico
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="raise")

    method = method.lower().strip()
    if method == "pearson":
        return _pearson_corr_df(df)

    if method == "spearman":
        ranked = df.rank(method="average")
        return _pearson_corr_df(ranked)

    raise ValueError("method debe ser 'pearson' o 'spearman'")
