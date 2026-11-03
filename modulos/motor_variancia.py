# Arquivo: modulos/motor_variancia.py
"""
Curva de variância explicada (só a matemática; o gráfico fica em modulos/graficos.py).
"""
import numpy as np
from sklearn.decomposition import PCA

from modulos.motor_pca import preparar_matriz


def calcular_variancia(X, max_pcs=20, padronizar=True, meta=95.0):
    """
    Devolve um dicionário com a variância de cada PC, a acumulada e
    quantos PCs são precisos para chegar na meta (ex.: 95%).
    pcs_para_meta é None se a meta não for atingida dentro de max_pcs.
    """
    if len(X) < 3:
        raise ValueError("São precisos pelo menos 3 espectros.")
    n_comp = min(max_pcs, len(X), X.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(preparar_matriz(X, padronizar))

    var = pca.explained_variance_ratio_ * 100
    acumulada = np.cumsum(var)
    atingiu = np.where(acumulada >= meta)[0]
    return {
        "var": var,
        "acumulada": acumulada,
        "meta": meta,
        "pcs_para_meta": int(atingiu[0] + 1) if len(atingiu) else None,
    }
