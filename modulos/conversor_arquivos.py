# Arquivo: modulos/conversor_arquivos.py
import os
import pandas as pd
from modulos.pre_processador import carregar_arquivo_esf, filtrar_ruido, remover_baseline, normalizar_por_area


def listar_esf(pasta_origem):
    """Todos os .esf dentro da pasta (e subpastas), em ordem."""
    encontrados = []
    for root, dirs, files in os.walk(pasta_origem):
        dirs.sort()
        encontrados.extend(os.path.join(root, f) for f in sorted(files) if f.lower().endswith('.esf'))
    return encontrados


def tratar_espectro(eixo_x, eixo_y, usar_sg=True, usar_baseline=True, usar_norm=True):
    """Aplica os filtros escolhidos. É a mesma função usada no preview e no lote."""
    intens = eixo_y.copy()
    if usar_sg:
        intens = filtrar_ruido(intens)
    if usar_baseline:
        intens, _ = remover_baseline(intens)
    if usar_norm:
        intens = normalizar_por_area(eixo_x, intens)
    return intens


def converter_pasta_esf_para_csv(pasta_origem, pasta_destino, usar_sg=True, usar_baseline=True, usar_norm=True,
                                 callback=None, progresso=None):
    """
    Lê .esf, aplica os filtros matemáticos escolhidos pelo utilizador na interface,
    e guarda os dados tratados e espelhados como .csv.
    progresso(feitos, total) é chamado a cada arquivo (para a barra de progresso).
    """
    sucesso = 0
    falhas = 0

    if callback: callback(f"Iniciando varredura em: {pasta_origem}\n")

    arquivos = listar_esf(pasta_origem)
    total = len(arquivos)
    if total == 0:
        msg = "Nenhum arquivo .esf encontrado."
        if callback: callback(msg)
        return 0, 0, msg

    pasta_anterior = None
    for n, caminho_completo_origem in enumerate(arquivos, start=1):
        root = os.path.dirname(caminho_completo_origem)
        arquivo = os.path.basename(caminho_completo_origem)
        caminho_relativo = os.path.relpath(root, pasta_origem)
        pasta_destino_atual = os.path.join(pasta_destino, caminho_relativo)
        if root != pasta_anterior:
            os.makedirs(pasta_destino_atual, exist_ok=True)
            if callback: callback(f"\n📁 A processar subpasta: {caminho_relativo}")
            pasta_anterior = root

        caminho_completo_destino = os.path.join(pasta_destino_atual, os.path.splitext(arquivo)[0] + ".csv")
        try:
            eixo_x, eixo_y = carregar_arquivo_esf(caminho_completo_origem)
            intens_tratada = tratar_espectro(eixo_x, eixo_y, usar_sg, usar_baseline, usar_norm)

            df = pd.DataFrame({'Comprimento_Onda': eixo_x, 'Intensidade': intens_tratada})
            # 10 algarismos significativos: a escrita fica ~2x mais rápida, sem perda útil
            df.to_csv(caminho_completo_destino, index=False, sep=';', float_format='%.10g')
            sucesso += 1
            if callback: callback(f"  ✔️ Processado: {arquivo}")
        except Exception as e:
            falhas += 1
            if callback: callback(f"  ❌ Falha: {arquivo} (Erro: {str(e)})")
        if progresso: progresso(n, total)

    mensagem = f"\n✅ Varredura concluída: {sucesso} sucessos, {falhas} falhas."
    if callback: callback(mensagem)
    return sucesso, falhas, mensagem
