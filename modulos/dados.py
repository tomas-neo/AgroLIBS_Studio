# Arquivo: modulos/dados.py
"""
Leitura dos .csv tratados, compartilhada por todas as abas de análise.

Antes, cada motor (PCA, outliers, variância, classificação) tinha a sua
própria cópia da leitura. Agora todos usam estas funções, o que garante:
- o mesmo filtro UV/VIS em todas as abas;
- o mesmo eixo de comprimento de onda (interpolado na faixa comum),
  em vez de cortar os espectros pelo tamanho do menor ("vacina LIBS");
- um cache: se você roda o PCA e depois a curva de variância na mesma
  pasta, os arquivos não são lidos de novo.
"""
import os
import threading
from collections import OrderedDict

import numpy as np
import pandas as pd

PASTAS_DE_REGIAO = {"uv", "vis"}
PASTA_QUARENTENA = "QUARENTENA"

# Grafias diferentes da mesma marca viram um nome só na legenda
NOMES_BONITOS = {"maxgreen": "MaxGreen", "plantfertil": "PlantFertil"}


# ==============================================================================
# 1. ENCONTRAR ARQUIVOS
# ==============================================================================
def _partes(caminho, pasta_raiz):
    return os.path.relpath(caminho, pasta_raiz).replace("\\", "/").split("/")


def listar_csvs(pasta_raiz, regiao):
    """Lista os .csv de uma região ('UV' ou 'Vis'), ignorando a QUARENTENA."""
    arquivos = []
    for root, dirs, files in os.walk(pasta_raiz):
        # Não desce na quarentena nem em pastas ocultas
        dirs[:] = [d for d in dirs if d != PASTA_QUARENTENA and not d.startswith(".")]
        for f in files:
            if not f.lower().endswith(".csv") or f.lower().startswith("classificacao_"):
                continue
            caminho = os.path.join(root, f)
            if regiao.lower() in [p.lower() for p in _partes(caminho, pasta_raiz)[:-1]]:
                arquivos.append(caminho)
    return sorted(arquivos)


def contar_por_regiao(pasta_raiz):
    """Quantos .csv existem em cada região. Ex.: {'UV': 120, 'VIS': 300}."""
    return {"UV": len(listar_csvs(pasta_raiz, "UV")), "VIS": len(listar_csvs(pasta_raiz, "Vis"))}


def pastas_do_caminho(caminho, pasta_raiz):
    """Pastas entre a raiz e o arquivo, sem as pastas UV/VIS."""
    return [p for p in _partes(caminho, pasta_raiz)[:-1] if p.lower() not in PASTAS_DE_REGIAO]


# ==============================================================================
# 2. GRUPOS (CORES DA LEGENDA / CLASSES)
# ==============================================================================
def _nome_bonito(nome):
    return NOMES_BONITOS.get(nome.lower(), nome)


def opcoes_de_agrupamento(arquivos, pasta_raiz):
    """
    Quais níveis de pasta podem virar grupo de cor. Devolve [(texto, niveis)].
    Ex.: NPK 4-14-8/Vis/Maxgreen/500ns/01_Jul -> nível 1 = 'NPK 4-14-8',
    nível 2 = 'Maxgreen'. A primeira opção junta os níveis 1 e 2
    ('NPK 4-14-8 (MaxGreen)'), como o PCA fazia antes.
    """
    niveis = {}
    for arq in arquivos:
        for i, nome in enumerate(pastas_do_caminho(arq, pasta_raiz)):
            niveis.setdefault(i, set()).add(_nome_bonito(nome))
    uteis = [i for i in sorted(niveis) if len(niveis[i]) >= 2]
    opcoes = []
    if len(uteis) >= 2:
        a, b = uteis[0], uteis[1]
        opcoes.append((f"Pastas nível {a + 1} + {b + 1}", (a, b)))
    for i in uteis:
        valores = sorted(niveis[i])
        exemplo = ", ".join(valores[:3]) + ("..." if len(valores) > 3 else "")
        opcoes.append((f"Pasta nível {i + 1} ({len(valores)}: {exemplo})", (i,)))
    if not opcoes:
        opcoes.append(("Todos juntos", ()))
    return opcoes


def grupo_do_arquivo(caminho, pasta_raiz, niveis):
    """Nome do grupo de um arquivo. niveis=(0, 1) -> 'NPK 4-14-8 (MaxGreen)'."""
    pastas = pastas_do_caminho(caminho, pasta_raiz)
    nomes = [_nome_bonito(pastas[n]) for n in niveis if n < len(pastas)]
    if not nomes:
        return "Amostras"
    return nomes[0] if len(nomes) == 1 else f"{nomes[0]} ({', '.join(nomes[1:])})"


# ==============================================================================
# 3. LER ESPECTROS
# ==============================================================================
def ler_csv(caminho):
    """Lê um .csv tratado (aceita ';' ou ','). Devolve (x, y) ou (None, None)."""
    df = pd.read_csv(caminho, sep=";")
    if len(df.columns) == 1:
        df = pd.read_csv(caminho, sep=",")
    col_x = next((c for c in df.columns if "onda" in c.lower() or "wave" in c.lower()
                  or "lambda" in c.lower()), None)
    col_y = next((c for c in df.columns if "intens" in c.lower()), None)
    if col_y is None:
        return None, None
    y = df[col_y].to_numpy(dtype=float)
    x = df[col_x].to_numpy(dtype=float) if col_x else np.arange(len(y), dtype=float)
    return x, y


_cache = OrderedDict()   # (arquivos + datas de modificação) -> resultado
_TAMANHO_CACHE = 2       # guarda UV e VIS; mais que isso gasta muita memória
_trava = threading.Lock()  # duas abas rodando ao mesmo tempo não mexem no cache juntas


def _chave_cache(arquivos):
    return tuple((a, os.path.getmtime(a)) for a in arquivos)


def carregar_espectros(arquivos, callback=None):
    """
    Lê vários .csv e monta a matriz X (um espectro por linha).
    Todos os espectros ficam no mesmo eixo de comprimento de onda: o do
    primeiro arquivo, cortado na faixa comum a todos. Se os eixos já forem
    iguais (o normal, quando vêm da Aba 1), não há interpolação.
    Devolve (X, eixo, arquivos_usados).
    """
    with _trava:
        return _carregar_espectros(arquivos, callback)


def _carregar_espectros(arquivos, callback):
    chave = _chave_cache(arquivos)
    if chave in _cache:
        _cache.move_to_end(chave)
        if callback:
            callback(f"  (usando {len(arquivos)} espectros já lidos antes)")
        return _cache[chave]

    espectros, usados = [], []
    for i, arq in enumerate(arquivos):
        x, y = ler_csv(arq)
        if y is None or len(y) < 10:
            if callback:
                callback(f"  ⚠️ Ignorado (sem coluna de intensidade): {os.path.basename(arq)}")
            continue
        if np.any(np.diff(x) < 0):
            ordem = np.argsort(x)
            x, y = x[ordem], y[ordem]
        espectros.append((x, y))
        usados.append(arq)
        if callback and (i + 1) % 100 == 0:
            callback(f"  ... {i + 1}/{len(arquivos)} arquivos lidos")

    if not espectros:
        raise ValueError("Nenhum espectro válido encontrado (falta a coluna de intensidade?).")

    x_ref = espectros[0][0]
    if all(len(x) == len(x_ref) and np.allclose(x, x_ref) for x, _ in espectros):
        eixo = x_ref
        X = np.vstack([y for _, y in espectros])
    else:
        inicio = max(x[0] for x, _ in espectros)
        fim = min(x[-1] for x, _ in espectros)
        eixo = x_ref[(x_ref >= inicio) & (x_ref <= fim)]
        if len(eixo) < 10:
            raise ValueError("Os espectros não têm uma faixa de comprimento de onda em comum.")
        X = np.vstack([np.interp(eixo, x, y) for x, y in espectros])
        if callback:
            callback(f"  Eixos diferentes entre arquivos: tudo interpolado em {eixo[0]:.1f}-{eixo[-1]:.1f} nm")

    resultado = (X, eixo, usados)
    _cache[chave] = resultado
    while len(_cache) > _TAMANHO_CACHE:
        _cache.popitem(last=False)
    return resultado


def limpar_cache():
    _cache.clear()


# ==============================================================================
# 4. QUARENTENA (em vez de apagar)
# ==============================================================================
def mover_para_quarentena(arquivos, pasta_raiz):
    """
    Move os arquivos para <pasta_raiz>/QUARENTENA, mantendo as subpastas.
    Todas as abas ignoram essa pasta. Para desfazer, use restaurar_quarentena.
    Devolve quantos arquivos foram movidos.
    """
    movidos = 0
    for arq in arquivos:
        destino = os.path.join(pasta_raiz, PASTA_QUARENTENA, os.path.relpath(arq, pasta_raiz))
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        os.replace(arq, destino)
        movidos += 1
    limpar_cache()
    return movidos


def restaurar_quarentena(pasta_raiz):
    """Devolve tudo o que está na QUARENTENA para o lugar original."""
    base = os.path.join(pasta_raiz, PASTA_QUARENTENA)
    restaurados = 0
    for root, _, files in os.walk(base):
        for f in files:
            origem = os.path.join(root, f)
            destino = os.path.join(pasta_raiz, os.path.relpath(origem, base))
            os.makedirs(os.path.dirname(destino), exist_ok=True)
            os.replace(origem, destino)
            restaurados += 1
    # apaga as pastas vazias que sobraram
    for root, dirs, files in os.walk(base, topdown=False):
        if not os.listdir(root):
            os.rmdir(root)
    limpar_cache()
    return restaurados


def contar_quarentena(pasta_raiz):
    base = os.path.join(pasta_raiz, PASTA_QUARENTENA)
    return sum(len(fs) for _, _, fs in os.walk(base)) if os.path.isdir(base) else 0
