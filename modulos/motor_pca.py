# Arquivo: modulos/motor_pca.py
"""
PCA dos espectros (só a matemática; os gráficos ficam em modulos/graficos.py).
"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def preparar_matriz(X, padronizar=True):
    """
    padronizar=True  -> StandardScaler: cada comprimento de onda com média 0 e variância 1.
    padronizar=False -> só centra na média. Os pixels com muito sinal (os picos)
                        pesam mais que os pixels de ruído, e os loadings ficam mais
                        fáceis de ler.
    """
    return StandardScaler(with_std=padronizar).fit_transform(X)


def calcular_pca(X, n_componentes=3, padronizar=True):
    """
    Devolve um dicionário com:
      escores  -> (n_espectros x n_componentes)
      var_exp  -> % de variância explicada de cada PC
      loadings -> (n_componentes x n_comprimentos_de_onda)
    """
    if len(X) < 3:
        raise ValueError("São precisos pelo menos 3 espectros para o PCA.")
    n = min(n_componentes, len(X), X.shape[1])
    pca = PCA(n_components=n, random_state=42)
    escores = pca.fit_transform(preparar_matriz(X, padronizar))
    return {
        "escores": escores,
        "var_exp": pca.explained_variance_ratio_ * 100,
        "loadings": pca.components_,
        "padronizado": padronizar,
    }
