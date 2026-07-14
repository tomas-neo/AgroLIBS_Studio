# Arquivo: modulos/pre_processador.py
import numpy as np
from scipy.signal import savgol_filter

def carregar_arquivo_esf(caminho_arquivo):
    """Lê o arquivo .esf de forma adaptativa, pulando cabeçalhos de texto."""
    encodings = ['utf-8', 'utf-16', 'latin-1']
    for enc in encodings:
        try:
            with open(caminho_arquivo, 'r', encoding=enc, errors='ignore') as f:
                linhas = f.readlines()
            
            if not linhas: continue
                
            dados_num = []
            for linha in linhas:
                linha_txt = linha.strip()
                if not linha_txt: continue
                
                if '\t' in linha_txt: partes = linha_txt.split('\t')
                elif ';' in linha_txt: partes = linha_txt.split(';')
                else: partes = linha_txt.split()
                
                if len(partes) >= 2:
                    try:
                        p_limpos = [p.strip().replace(',', '.') for p in partes]
                        valores = [float(p) for p in p_limpos if p]
                        
                        if len(valores) >= 3:
                            if valores[0] < 25000 and 180 <= valores[1] <= 1100:
                                dados_num.append([valores[1], valores[2]]) 
                            else:
                                dados_num.append([valores[0], valores[1]])
                        elif len(valores) == 2:
                            dados_num.append([valores[0], valores[1]])
                    except ValueError:
                        continue 
            
            if len(dados_num) > 0:
                dados_num = np.array(dados_num)
                # Ordena os dados pelo comprimento de onda (eixo X)
                dados_num = dados_num[dados_num[:, 0].argsort()]
                return dados_num[:, 0], dados_num[:, 1]
                
        except Exception:
            continue
            
    raise ValueError("Não foi possível extrair duas colunas numéricas válidas do arquivo.")

def filtrar_ruido(intensidade, janela=11, ordem=2):
    """Aplica o filtro Savitzky-Golay para remover ruído de alta frequência."""
    if janela % 2 == 0: janela += 1  
    if len(intensidade) <= janela: return intensidade
    return savgol_filter(intensidade, window_length=janela, polyorder=ordem)

def remover_baseline(intensidade, grau=2, iteracoes=10):
    """Remove a linha de base contínua do espectro iterativamente."""
    base = intensidade.copy()
    n = len(intensidade)
    if n == 0: return intensidade, np.zeros_like(intensidade)
        
    x_escalonado = np.linspace(-1, 1, n)
    for _ in range(iteracoes):
        p = np.polyfit(x_escalonado, base, grau)
        fit = np.polyval(p, x_escalonado)
        base = np.minimum(base, fit)
        
    baseline_estimado = np.polyval(np.polyfit(x_escalonado, base, grau), x_escalonado)
    espectro_corrigido = intensidade - baseline_estimado
    return np.clip(espectro_corrigido, 0, None), baseline_estimado

def normalizar_por_area(comprimento_onda, intensidade):
    """Normaliza o espectro dividindo pela área total sob a curva."""
    area_total = np.trapezoid(intensidade, comprimento_onda) # Usando trapezoid para versões mais recentes do numpy
    return intensidade / area_total if area_total != 0 else intensidade
