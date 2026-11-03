# Arquivo: modulos/motor_outliers.py
"""
Caça de outliers com Isolation Forest sobre os escores do PCA
(só a matemática; o gráfico fica em modulos/graficos.py).
"""
from sklearn.ensemble import IsolationForest

from modulos.motor_pca import calcular_pca


def detectar_outliers(X, contaminacao=0.03, n_pcs=2, padronizar=True):
    """
    1. PCA com n_pcs componentes.
    2. Isolation Forest nos escores: devolve -1 para anomalia e 1 para normal.

    contaminacao = fração do conjunto que o Isolation Forest vai marcar como
    anômala (0.03 = os 3% mais extremos). Atenção: ele SEMPRE marca essa fração,
    mesmo num conjunto sem problemas. Por isso, olhe o gráfico antes de mover.
    """
    if len(X) < 5:
        raise ValueError("São precisos pelo menos 5 espectros para procurar outliers.")
    pca = calcular_pca(X, n_componentes=max(2, n_pcs), padronizar=padronizar)
    escores = pca["escores"]

    iso = IsolationForest(contamination=contaminacao, random_state=42)
    previsoes = iso.fit_predict(escores[:, :n_pcs])
    # score_samples: quanto MENOR, mais anômalo. Invertemos para "nota de anomalia".
    nota = -iso.score_samples(escores[:, :n_pcs])
    return {
        "escores": escores,
        "var_exp": pca["var_exp"],
        "eh_outlier": previsoes == -1,
        "nota": nota,
    }
