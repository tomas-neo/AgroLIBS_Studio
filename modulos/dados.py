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
NOME_RELATORIO_QUALIDADE = "relatorio_qualidade.csv"   # o mesmo que modulos/conversor_arquivos.py usa

# Grafias diferentes da mesma marca viram um nome só na legenda
NOMES_BONITOS = {"maxgreen": "MaxGreen", "plantfertil": "PlantFertil", "corat": "Corat"}


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
            nome = f.lower()
            # Fora: resultados de classificação e as tabelas de picos do script antigo
            # (*_picos_identificados.csv), que não são espectros.
            if (not nome.endswith(".csv") or nome.startswith("classificacao_")
                    or nome.endswith("_picos_identificados.csv")):
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
    """Lê um .csv tratado (aceita ';' ou ','). Devolve (x, y) ou (None, None).
    Sem coluna de comprimento de onda não é um espectro (ex.: tabela de picos): (None, None)."""
    df = pd.read_csv(caminho, sep=";")
    if len(df.columns) == 1:
        df = pd.read_csv(caminho, sep=",")
    col_x = next((c for c in df.columns if "onda" in c.lower() or "wave" in c.lower()
                  or "lambda" in c.lower()), None)
    col_y = next((c for c in df.columns if "intens" in c.lower()), None)
    if col_y is None or col_x is None:
        return None, None
    return df[col_x].to_numpy(dtype=float), df[col_y].to_numpy(dtype=float)


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
                callback(f"  ⚠️ Ignorado (não é espectro: falta a coluna de comprimento de onda ou de intensidade): {os.path.basename(arq)}")
            continue
        if np.any(np.diff(x) < 0):
            ordem = np.argsort(x)
            x, y = x[ordem], y[ordem]
        espectros.append((x, y))
        usados.append(arq)
        if callback and (i + 1) % 100 == 0:
            callback(f"  ... {i + 1}/{len(arquivos)} arquivos lidos")

    if not espectros:
        raise ValueError("Nenhum espectro válido encontrado (faltam as colunas de comprimento de onda e intensidade?).")

    x_ref = espectros[0][0]
    if all(len(x) == len(x_ref) and np.allclose(x, x_ref) for x, _ in espectros):
        eixo = x_ref
        X = np.vstack([y for _, y in espectros])
    else:
        inicio = max(x[0] for x, _ in espectros)
        fim = min(x[-1] for x, _ in espectros)
        eixo = x_ref[(x_ref >= inicio) & (x_ref <= fim)]
        if len(eixo) < 10:
            curto = min(espectros, key=lambda e: e[0][-1] - e[0][0])
            arq_curto = os.path.basename(usados[[id(e) for e in espectros].index(id(curto))])
            raise ValueError("Os espectros não têm uma faixa de comprimento de onda em comum "
                             f"(ex.: {arq_curto} vai de {curto[0][0]:.1f} a {curto[0][-1]:.1f} nm). "
                             "Confira se a pasta mistura UV e VIS ou arquivos que não são espectros.")
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


def reprovados_no_filtro(pasta_raiz):
    """
    Caminhos (relativos à QUARENTENA, com '/') dos .csv que o filtro de qualidade
    da Aba 1 reprovou, lidos do relatorio_qualidade.csv. Vazio se não houver relatório.
    """
    relatorio = os.path.join(pasta_raiz, NOME_RELATORIO_QUALIDADE)
    if not os.path.isfile(relatorio):
        return set()
    try:
        df = pd.read_csv(relatorio, sep=";")
    except Exception:
        return set()
    if "arquivo" not in df.columns or "aprovado" not in df.columns:
        return set()
    aprovado = df["aprovado"].astype(str).str.strip().str.lower().isin(["true", "1", "verdadeiro"])
    # 'arquivo' é o .esf relativo à pasta de origem (ex.: "UV/a/tiro1.esf" ou "./tiro1.esf");
    # na QUARENTENA ele está como .csv, no mesmo caminho relativo
    return {os.path.normpath(os.path.splitext(str(a))[0] + ".csv").replace("\\", "/")
            for a in df.loc[~aprovado, "arquivo"]}


def _arquivos_da_quarentena(pasta_raiz, incluir_reprovados_filtro):
    """(origem, caminho relativo) de cada arquivo da QUARENTENA que pode ser devolvido."""
    base = os.path.join(pasta_raiz, PASTA_QUARENTENA)
    reprovados = set() if incluir_reprovados_filtro else reprovados_no_filtro(pasta_raiz)
    for root, _, files in os.walk(base):
        for f in files:
            origem = os.path.join(root, f)
            relativo = os.path.relpath(origem, base).replace("\\", "/")
            if relativo not in reprovados:
                yield origem, relativo


def restaurar_quarentena(pasta_raiz, incluir_reprovados_filtro=False):
    """
    Devolve para o lugar original o que está na QUARENTENA.
    Por padrão, NÃO devolve os tiros reprovados no filtro de qualidade da Aba 1
    (fracos ou saturados, listados no relatorio_qualidade.csv): restaurar os
    outliers da Aba 3 não pode trazer de volta tiros sem plasma ou saturados.
    """
    base = os.path.join(pasta_raiz, PASTA_QUARENTENA)
    restaurados = 0
    for origem, relativo in list(_arquivos_da_quarentena(pasta_raiz, incluir_reprovados_filtro)):
        destino = os.path.join(pasta_raiz, relativo)
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        os.replace(origem, destino)
        restaurados += 1
    # apaga as pastas vazias que sobraram
    for root, dirs, files in os.walk(base, topdown=False):
        if not os.listdir(root):
            os.rmdir(root)
    limpar_cache()
    return restaurados


def contar_quarentena(pasta_raiz, so_restauraveis=False):
    """
    Quantos arquivos estão na QUARENTENA. Com so_restauraveis=True, conta só os
    que o 'Restaurar' devolveria (sem os reprovados no filtro de qualidade).
    """
    base = os.path.join(pasta_raiz, PASTA_QUARENTENA)
    if not os.path.isdir(base):
        return 0
    if so_restauraveis:
        return sum(1 for _ in _arquivos_da_quarentena(pasta_raiz, incluir_reprovados_filtro=False))
    return sum(len(fs) for _, _, fs in os.walk(base))
