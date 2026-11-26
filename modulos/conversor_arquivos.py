# Arquivo: modulos/conversor_arquivos.py
import os
import pandas as pd
from modulos.pre_processador import carregar_arquivo_esf, filtrar_ruido, remover_baseline, normalizar_por_area
from modulos.qualidade import avaliar_tiro

PASTA_QUARENTENA = "QUARENTENA"           # a mesma que modulos/dados.py usa
NOME_RELATORIO_QUALIDADE = "relatorio_qualidade.csv"


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
                                 callback=None, progresso=None, qualidade=None):
    """
    Lê .esf, aplica os filtros matemáticos escolhidos pelo utilizador na interface,
    e guarda os dados tratados e espelhados como .csv.
    progresso(feitos, total) é chamado a cada arquivo (para a barra de progresso).

    qualidade: None (sem filtro) ou {'snr_minimo': 20, 'teto_saturacao': 578000}.
    Com o filtro, cada tiro é avaliado no espectro BRUTO (modulos/qualidade.py).
    Os reprovados também viram .csv, mas vão para <destino>/QUARENTENA (as abas
    2 a 5 ignoram essa pasta, e o botão "Restaurar quarentena" da Aba 3 desfaz).
    Todos os tiros ficam listados em <destino>/relatorio_qualidade.csv.

    Devolve (sucesso, falhas, mensagem, relatorio), em que relatorio é a lista
    de dicionários da avaliação (vazia sem filtro).
    """
    sucesso = 0
    falhas = 0
    relatorio = []

    if callback: callback(f"Iniciando varredura em: {pasta_origem}\n")

    arquivos = listar_esf(pasta_origem)
    total = len(arquivos)
    if total == 0:
        msg = "Nenhum arquivo .esf encontrado."
        if callback: callback(msg)
        return 0, 0, msg, relatorio

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

        nome_csv = os.path.splitext(arquivo)[0] + ".csv"
        caminho_completo_destino = os.path.join(pasta_destino_atual, nome_csv)
        try:
            eixo_x, eixo_y = carregar_arquivo_esf(caminho_completo_origem)
            if qualidade is not None:
                nota = avaliar_tiro(eixo_y, qualidade.get('snr_minimo'), qualidade.get('teto_saturacao'))
                relatorio.append({'arquivo': os.path.join(caminho_relativo, arquivo).replace('\\', '/'), **nota})
                if not nota['aprovado']:
                    pasta_q = os.path.join(pasta_destino, PASTA_QUARENTENA, caminho_relativo)
                    os.makedirs(pasta_q, exist_ok=True)
                    caminho_completo_destino = os.path.join(pasta_q, nome_csv)
                    # um .csv antigo (de um lote sem filtro) não pode ficar fora da quarentena
                    antigo = os.path.join(pasta_destino_atual, nome_csv)
                    if os.path.exists(antigo):
                        os.remove(antigo)
                else:
                    # e um tiro aprovado agora não pode ter uma cópia velha na quarentena
                    velho = os.path.join(pasta_destino, PASTA_QUARENTENA, caminho_relativo, nome_csv)
                    if os.path.exists(velho):
                        os.remove(velho)
            intens_tratada = tratar_espectro(eixo_x, eixo_y, usar_sg, usar_baseline, usar_norm)

            df = pd.DataFrame({'Comprimento_Onda': eixo_x, 'Intensidade': intens_tratada})
            # 10 algarismos significativos: a escrita fica ~2x mais rápida, sem perda útil
            df.to_csv(caminho_completo_destino, index=False, sep=';', float_format='%.10g')
            sucesso += 1
            if qualidade is not None and not nota['aprovado']:
                if callback: callback(f"  ⚠️ Quarentena ({nota['motivo']}, SNR {nota['snr']:.0f}): {arquivo}")
            elif callback:
                callback(f"  ✔️ Processado: {arquivo}")
        except Exception as e:
            falhas += 1
            if callback: callback(f"  ❌ Falha: {arquivo} (Erro: {str(e)})")
        if progresso: progresso(n, total)

    mensagem = f"\n✅ Varredura concluída: {sucesso} sucessos, {falhas} falhas."
    if relatorio:
        reprovados = [r for r in relatorio if not r['aprovado']]
        os.makedirs(pasta_destino, exist_ok=True)
        pd.DataFrame(relatorio).to_csv(os.path.join(pasta_destino, NOME_RELATORIO_QUALIDADE),
                                       index=False, sep=';', float_format='%.6g')
        mensagem += (f"\n🔎 Filtro de qualidade: {len(reprovados)} de {len(relatorio)} tiros na QUARENTENA "
                     f"({sum('fraco' in r['motivo'] for r in reprovados)} fracos, "
                     f"{sum('saturado' in r['motivo'] for r in reprovados)} saturados). "
                     f"Detalhes em {NOME_RELATORIO_QUALIDADE}.")
    if callback: callback(mensagem)
    return sucesso, falhas, mensagem, relatorio
