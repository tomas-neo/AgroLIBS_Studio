# Arquivo: modulos/conversor_arquivos.py
import os
import pandas as pd
# Importante: Garantir que todas as funções do pré-processador estão importadas aqui!
from modulos.pre_processador import carregar_arquivo_esf, filtrar_ruido, remover_baseline, normalizar_por_area

def converter_pasta_esf_para_csv(pasta_origem, pasta_destino, usar_sg=True, usar_baseline=True, usar_norm=True, callback=None):
    """
    Lê .esf, aplica os filtros matemáticos escolhidos pelo utilizador na interface,
    e guarda os dados tratados e espelhados como .csv.
    """
    sucesso = 0
    falhas = 0
    
    if callback: callback(f"Iniciando varredura em: {pasta_origem}\n")

    for root, dirs, files in os.walk(pasta_origem):
        arquivos_esf = [f for f in files if f.lower().endswith('.esf')]
        if not arquivos_esf: continue
            
        caminho_relativo = os.path.relpath(root, pasta_origem)
        pasta_destino_atual = os.path.join(pasta_destino, caminho_relativo)
        os.makedirs(pasta_destino_atual, exist_ok=True)
        
        if callback: callback(f"\n📁 A processar subpasta: {caminho_relativo}")

        for arquivo in arquivos_esf:
            caminho_completo_origem = os.path.join(root, arquivo)
            nome_sem_extensao = os.path.splitext(arquivo)[0]
            nome_novo_csv = f"{nome_sem_extensao}.csv"
            caminho_completo_destino = os.path.join(pasta_destino_atual, nome_novo_csv)
            
            try:
                # 1. Carrega o dado bruto
                eixo_x, eixo_y = carregar_arquivo_esf(caminho_completo_origem)
                intens_tratada = eixo_y.copy()
                
                # 2. Aplica os filtros baseados nas escolhas da interface
                if usar_sg:
                    intens_tratada = filtrar_ruido(intens_tratada)
                if usar_baseline:
                    intens_tratada, _ = remover_baseline(intens_tratada)
                if usar_norm:
                    intens_tratada = normalizar_por_area(eixo_x, intens_tratada)
                
                # 3. Guarda o dado tratado
                df = pd.DataFrame({'Comprimento_Onda': eixo_x, 'Intensidade': intens_tratada})
                df.to_csv(caminho_completo_destino, index=False, sep=';')
                sucesso += 1
                
                if callback: callback(f"  ✔️ Processado: {arquivo}")
                    
            except Exception as e:
                falhas += 1
                if callback: callback(f"  ❌ Falha: {arquivo} (Erro: {str(e)})")
                
    if sucesso == 0 and falhas == 0:
        msg = "Nenhum arquivo .esf encontrado."
        if callback: callback(msg)
        return 0, 0, msg
        
    mensagem = f"\n✅ Varredura concluída: {sucesso} sucessos, {falhas} falhas."
    if callback: callback(mensagem)
    return sucesso, falhas, mensagem
