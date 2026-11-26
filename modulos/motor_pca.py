# Arquivo: modulos/motor_pca.py
"""
PCA dos espectros (só a matemática; os gráficos ficam em modulos/graficos.py).
"""
import numpy as np
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


def dispersao_por_grupo(escores, grupos, n_pcs=2):
    """
    Homogeneidade de cada grupo (marca, formulação...) no espaço PC1 x PC2.
    Quanto MENOR, mais parecidos entre si são os tiros do grupo.

    Vem da função calcular_dispersao_mahalanobis que o Tomas escreveu em
    14/08/2026, com uma correção importante. Lá, cada grupo usava a PRÓPRIA
    matriz de covariância. Com isso, a média de d² dá sempre 2(n-1)/n, e o raio
    médio fica perto de 1,25 para QUALQUER grupo, seja ele espalhado ou
    concentrado: a distância se "autonormaliza" e não compara grupos.
    Aqui a distância de Mahalanobis usa a covariância COMUM (média ponderada das
    covariâncias dos grupos), que é a mesma régua para todos.

    Devolve uma lista de dicionários (um por grupo):
      grupo, n
      raio_euclidiano  -> distância média dos tiros ao centro do grupo (unidades de escore)
      area_elipse95    -> área da elipse de 95% desenhada no gráfico
      mahal_medio      -> distância de Mahalanobis média ao centro (covariância comum)
      mahal_maximo     -> o tiro mais afastado do centro (covariância comum)
    """
    E = np.asarray(escores, dtype=float)[:, :n_pcs]
    grupos = np.asarray(grupos)
    unicos = list(dict.fromkeys(grupos))

    # covariância comum (pooled) entre os grupos com 2+ tiros
    soma, gl = np.zeros((n_pcs, n_pcs)), 0
    for g in unicos:
        G = E[grupos == g]
        if len(G) >= 2:
            soma += (len(G) - 1) * np.cov(G, rowvar=False)
            gl += len(G) - 1
    inv_comum = np.linalg.pinv(soma / gl) if gl > 0 else None

    resultado = []
    for g in unicos:
        G = E[grupos == g]
        delta = G - G.mean(axis=0)
        linha = {"grupo": str(g), "n": len(G),
                 "raio_euclidiano": float(np.mean(np.linalg.norm(delta, axis=1))),
                 "area_elipse95": np.nan, "mahal_medio": np.nan, "mahal_maximo": np.nan}
        if len(G) >= 3:
            det = np.linalg.det(np.cov(G, rowvar=False))
            if det > 0:
                linha["area_elipse95"] = float(np.pi * 5.991 * np.sqrt(det))
        if inv_comum is not None:
            d = np.sqrt(np.maximum(0, np.sum(delta @ inv_comum * delta, axis=1)))
            linha["mahal_medio"] = float(d.mean())
            linha["mahal_maximo"] = float(d.max())
        resultado.append(linha)
    return resultado
