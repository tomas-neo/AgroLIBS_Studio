# Arquivo: modulos/motor_classificacao.py
"""
Motor de Classificação Supervisionada (KNN e Random Forest).

Fluxo:
1. Lê os .csv tratados (saída da Aba 1), filtrando UV ou VIS.
2. O rótulo (classe) de cada espectro vem do nome de uma pasta do caminho
   (ex.: "Adulterado_25porcento" ou "MaxGreen").
3. Validação cruzada AGRUPADA por amostra física: todos os disparos da
   "amostra_a" ficam juntos no treino OU no teste. Sem isso, disparos quase
   idênticos da mesma pastilha caem dos dois lados e a acurácia sai inflada.
4. KNN:  Padronização -> PCA -> KNN (tudo ajustado só no treino de cada fold).
   RF:   Random Forest direto no espectro (permite ver a importância de cada
         comprimento de onda).

Este módulo não usa tkinter: a interface (main.py) chama as funções e
desenha os gráficos.
"""
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modulos.pre_processador import ELEMENTS_WAVELENGTHS

PASTAS_DE_REGIAO = {"uv", "vis"}


# ==============================================================================
# 1. LEITURA DOS DADOS
# ==============================================================================
def listar_csvs(pasta_raiz, regiao):
    """Lista os .csv de uma região ('UV' ou 'Vis'), ignorando a QUARENTENA."""
    arquivos = []
    for root, dirs, files in os.walk(pasta_raiz):
        if "QUARENTENA" in root:
            continue
        for f in files:
            if not f.lower().endswith(".csv"):
                continue
            caminho = os.path.join(root, f)
            partes = os.path.relpath(caminho, pasta_raiz).replace("\\", "/").lower().split("/")
            if regiao.lower() in partes[:-1]:
                arquivos.append(caminho)
    return sorted(arquivos)


def pastas_do_caminho(caminho, pasta_raiz):
    """Pastas entre a raiz e o arquivo, sem as pastas UV/VIS."""
    partes = os.path.relpath(caminho, pasta_raiz).replace("\\", "/").split("/")[:-1]
    return [p for p in partes if p.lower() not in PASTAS_DE_REGIAO]


def opcoes_de_rotulo(arquivos, pasta_raiz):
    """
    Descobre quais níveis de pasta podem virar classe.
    Devolve uma lista de (nível, texto para o menu). Só entram níveis com 2+ valores.
    """
    niveis = {}
    for arq in arquivos:
        for i, nome in enumerate(pastas_do_caminho(arq, pasta_raiz)):
            niveis.setdefault(i, set()).add(nome.lower())
    opcoes = []
    for i in sorted(niveis):
        valores = sorted(niveis[i])
        if len(valores) >= 2:
            exemplo = ", ".join(valores[:3]) + ("..." if len(valores) > 3 else "")
            opcoes.append((i, f"Pasta nível {i + 1} ({len(valores)} classes: {exemplo})"))
    return opcoes


def rotulo_do_arquivo(caminho, pasta_raiz, nivel):
    pastas = pastas_do_caminho(caminho, pasta_raiz)
    return pastas[nivel] if nivel < len(pastas) else None


def amostra_do_arquivo(caminho, pasta_raiz):
    """
    Identifica a amostra física de onde veio o disparo.
    'NPK_25pct_Jul_15_amostra_c_VIS__07.csv' -> '<pastas>/amostra_c'
    Sem 'amostra_x' no nome, usa o nome sem o número final do disparo.
    """
    nome = os.path.splitext(os.path.basename(caminho))[0]
    achou = re.search(r"amostra[_\s-]*([A-Za-z0-9]+?)(?=_|$)", nome, re.IGNORECASE)
    if achou:
        id_amostra = "amostra_" + achou.group(1).lower()
    else:
        id_amostra = re.sub(r"[_\s-]*\d+$", "", nome).lower()
    pastas = [p.lower() for p in pastas_do_caminho(caminho, pasta_raiz)]
    return "/".join(pastas + [id_amostra])


def _ler_csv(caminho):
    df = pd.read_csv(caminho, sep=";")
    if len(df.columns) == 1:
        df = pd.read_csv(caminho, sep=",")
    col_x = next((c for c in df.columns if "onda" in c.lower() or "wave" in c.lower()), None)
    col_y = next((c for c in df.columns if "intens" in c.lower()), None)
    if col_y is None:
        return None, None
    y = df[col_y].to_numpy(dtype=float)
    x = df[col_x].to_numpy(dtype=float) if col_x else np.arange(len(y), dtype=float)
    return x, y


def carregar_dataset(arquivos, pasta_raiz, nivel_rotulo, callback=None):
    """
    Monta a matriz X (espectros), y (classes) e grupos (amostra física).
    Todos os espectros são interpolados no eixo de comprimento de onda do
    primeiro arquivo, cortado na faixa comum a todos.
    """
    espectros, rotulos, grupos, usados = [], [], [], []
    nomes_classe = {}  # 'maxgreen' -> 'MaxGreen' (junta grafias diferentes)

    for i, arq in enumerate(arquivos):
        rotulo = rotulo_do_arquivo(arq, pasta_raiz, nivel_rotulo)
        if rotulo is None:
            continue
        x, y = _ler_csv(arq)
        if y is None or len(y) < 10:
            continue
        ordem = np.argsort(x)
        espectros.append((x[ordem], y[ordem]))
        rotulos.append(nomes_classe.setdefault(rotulo.lower(), rotulo))
        grupos.append(amostra_do_arquivo(arq, pasta_raiz))
        usados.append(arq)
        if callback and (i + 1) % 100 == 0:
            callback(f"  ... {i + 1}/{len(arquivos)} arquivos lidos")

    if not espectros:
        raise ValueError("Nenhum espectro válido encontrado (falta coluna de intensidade?).")

    inicio = max(x[0] for x, _ in espectros)
    fim = min(x[-1] for x, _ in espectros)
    x_ref = espectros[0][0]
    eixo = x_ref[(x_ref >= inicio) & (x_ref <= fim)]
    if len(eixo) < 10:
        raise ValueError("Os espectros não têm uma faixa de comprimento de onda em comum.")

    X = np.vstack([np.interp(eixo, x, y) for x, y in espectros])
    return X, np.array(rotulos), np.array(grupos), eixo, usados


def resumo_dataset(y, grupos):
    linhas = []
    for classe in sorted(set(y)):
        mascara = y == classe
        n_amostras = len(set(grupos[mascara]))
        linhas.append(f"  {classe}: {mascara.sum()} espectros de {n_amostras} amostras")
    return "\n".join(linhas)


# ==============================================================================
# 2. MODELOS
# ==============================================================================
def criar_modelo(tipo, k=5, n_pcs=10, n_arvores=300):
    if tipo == "KNN":
        return Pipeline([
            ("padronizacao", StandardScaler()),
            ("pca", PCA(n_components=n_pcs, random_state=42)),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ])
    return RandomForestClassifier(n_estimators=n_arvores, n_jobs=-1, random_state=42)


def _numero_de_folds(y, grupos, maximo=5):
    """Cada classe precisa de pelo menos 2 amostras físicas para ter treino e teste."""
    menor = min(len(set(grupos[y == c])) for c in set(y))
    if menor < 2:
        raise ValueError(
            "Alguma classe tem só 1 amostra física (ex.: só 'amostra_a'). "
            "Para validar sem vazamento, cada classe precisa de pelo menos 2 amostras."
        )
    return min(maximo, menor)


def _voto_por_amostra(y_true, y_pred, grupos):
    """Classe final de cada amostra física = voto da maioria dos disparos."""
    acertos, total = 0, 0
    for g in set(grupos):
        m = grupos == g
        voto = Counter(y_pred[m]).most_common(1)[0][0]
        acertos += int(voto == y_true[m][0])
        total += 1
    return acertos, total


def avaliar_modelo(X, y, grupos, tipo, k=5, n_pcs=10, n_arvores=300,
                   curva_k=True, callback=None):
    """
    Validação cruzada agrupada por amostra. Devolve um dicionário com
    acurácia, matriz de confusão, relatório e (KNN) a curva acurácia x k.
    """
    n_folds = _numero_de_folds(y, grupos)
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    classes = sorted(set(y))

    y_pred = np.empty(len(y), dtype=object)
    valores_k = list(range(1, 26, 2)) if (tipo == "KNN" and curva_k) else []
    acertos_por_k = {kk: 0 for kk in valores_k}
    menor_treino = len(y)

    for f, (treino, teste) in enumerate(cv.split(X, y, grupos), start=1):
        if callback:
            callback(f"  Fold {f}/{n_folds}: {len(treino)} espectros de treino, {len(teste)} de teste")

        menor_treino = min(menor_treino, len(treino))
        if tipo == "KNN":
            # Padronização e PCA ajustadas SÓ no treino; depois testamos vários k
            pcs = min(n_pcs, len(treino) - 1, X.shape[1])
            prep = Pipeline([("padronizacao", StandardScaler()),
                             ("pca", PCA(n_components=pcs, random_state=42))])
            Z_treino = prep.fit_transform(X[treino])
            Z_teste = prep.transform(X[teste])
            knn = KNeighborsClassifier(n_neighbors=min(k, len(treino)))
            y_pred[teste] = knn.fit(Z_treino, y[treino]).predict(Z_teste)
            for kk in valores_k:
                if kk <= len(treino):
                    pred = KNeighborsClassifier(n_neighbors=kk).fit(Z_treino, y[treino]).predict(Z_teste)
                    acertos_por_k[kk] += int((pred == y[teste]).sum())
        else:
            modelo = criar_modelo("RF", n_arvores=n_arvores)
            y_pred[teste] = modelo.fit(X[treino], y[treino]).predict(X[teste])

    y_pred = y_pred.astype(str)
    acertos_amostra, total_amostras = _voto_por_amostra(y, y_pred, grupos)
    resultado = {
        "tipo": tipo,
        "classes": classes,
        "n_folds": n_folds,
        "acuracia": accuracy_score(y, y_pred),
        "acuracia_por_amostra": acertos_amostra / total_amostras,
        "acertos_amostra": acertos_amostra,
        "total_amostras": total_amostras,
        "matriz": confusion_matrix(y, y_pred, labels=classes),
        "relatorio": classification_report(y, y_pred, labels=classes, zero_division=0),
        "curva_k": {kk: acertos_por_k[kk] / len(y) for kk in valores_k if kk <= menor_treino},
    }
    return resultado


def treinar_modelo_final(X, y, tipo, eixo, regiao, nivel_rotulo, k=5, n_pcs=10, n_arvores=300):
    """Treina com TODOS os dados (para salvar e classificar amostras novas)."""
    pcs = min(n_pcs, len(y) - 1, X.shape[1])
    modelo = criar_modelo(tipo, k=min(k, len(y)), n_pcs=pcs, n_arvores=n_arvores)
    modelo.fit(X, y)
    return {
        "modelo": modelo,
        "tipo": tipo,
        "eixo": eixo,
        "regiao": regiao,
        "nivel_rotulo": nivel_rotulo,
        "parametros": {"k": k, "n_pcs": pcs, "n_arvores": n_arvores},
    }


def importancia_rf(pacote):
    """Importância de cada comprimento de onda no Random Forest final."""
    if pacote["tipo"] != "RF":
        return None
    return pacote["eixo"], pacote["modelo"].feature_importances_


def linhas_npk_proximas(eixo, importancias, n=10, tolerancia=1.0):
    """Os n comprimentos de onda mais importantes e se batem com alguma linha N/P/K."""
    texto = []
    for idx in np.argsort(importancias)[::-1][:n]:
        wl = eixo[idx]
        perto = [f"{el} {ref:.2f}" for el, refs in ELEMENTS_WAVELENGTHS.items()
                 for ref in refs if abs(ref - wl) <= tolerancia]
        marca = f"  <- linha {', '.join(perto)}" if perto else ""
        texto.append(f"  {wl:8.2f} nm  (importância {importancias[idx]:.4f}){marca}")
    return "\n".join(texto)


# ==============================================================================
# 3. SALVAR / CARREGAR / CLASSIFICAR AMOSTRAS NOVAS
# ==============================================================================
def salvar_modelo(pacote, caminho):
    joblib.dump(pacote, caminho)


def carregar_modelo(caminho):
    return joblib.load(caminho)


def classificar_pasta(pacote, pasta):
    """
    Classifica todos os .csv de uma pasta (mesma região do modelo).
    Devolve um DataFrame por espectro e um resumo por amostra (voto da maioria).
    """
    arquivos = listar_csvs(pasta, pacote["regiao"])
    if not arquivos:
        # Pasta sem subpastas UV/VIS: usa todos os .csv
        arquivos = sorted(os.path.join(r, f) for r, _, fs in os.walk(pasta)
                          for f in fs if f.lower().endswith(".csv") and "QUARENTENA" not in r)
    if not arquivos:
        raise ValueError("Nenhum .csv encontrado na pasta.")

    eixo = pacote["eixo"]
    linhas, espectros = [], []
    for arq in arquivos:
        x, y = _ler_csv(arq)
        if y is None:
            continue
        ordem = np.argsort(x)
        x, y = x[ordem], y[ordem]
        if x[0] > eixo[0] + 1 or x[-1] < eixo[-1] - 1:
            continue  # faixa espectral diferente da usada no treino
        espectros.append(np.interp(eixo, x, y))
        linhas.append({"arquivo": os.path.relpath(arq, pasta),
                       "amostra": amostra_do_arquivo(arq, pasta)})
    if not espectros:
        raise ValueError("Nenhum espectro compatível com a faixa do modelo treinado.")

    X = np.vstack(espectros)
    modelo = pacote["modelo"]
    pred = modelo.predict(X)
    df = pd.DataFrame(linhas)
    df["classe_prevista"] = pred
    if hasattr(modelo, "predict_proba"):
        df["confianca"] = modelo.predict_proba(X).max(axis=1)

    resumo = (df.groupby("amostra")["classe_prevista"]
                .agg(lambda s: s.value_counts().index[0])
                .rename("classe_final").to_frame())
    resumo["disparos"] = df.groupby("amostra").size()
    resumo["votos_na_classe"] = df.groupby("amostra")["classe_prevista"].agg(
        lambda s: s.value_counts().iloc[0])
    return df, resumo.reset_index()
