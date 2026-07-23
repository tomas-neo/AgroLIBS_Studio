# Arquivo: modulos/pre_processador.py
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks

# Dicionários de química
ELEMENTS_WAVELENGTHS = {
    'N': [742.36, 744.23, 746.83, 821.63, 868.02],
    'P': [213.62, 214.91, 253.56, 255.33],
    'K': [766.49, 769.90, 404.72]
}

def carregar_arquivo_esf(caminho_arquivo):
    dados_num = []
    iniciou_dados = False
    
    with open(caminho_arquivo, 'r', encoding='utf-8', errors='ignore') as f:
        for linha in f:
            linha = linha.strip()
            if not iniciou_dados:
                if linha == '[data section]':
                    iniciou_dados = True
                continue
            if not linha: continue
            
            partes = linha.split(';')
            if len(partes) >= 3:
                try:
                    wave = float(partes[0].replace(',', '.'))
                    intensity = float(partes[2].replace(',', '.'))
                    if wave > 0:
                        dados_num.append([wave, intensity])
                except ValueError:
                    continue
                    
    if len(dados_num) > 0:
        df = pd.DataFrame(dados_num, columns=['lambda', 'intensidade'])
        
        # O SEGREDO DO CÓDIGO DE BARRAS: Arredondar para 2 casas decimais antes de agrupar!
        # Isso força as lentes sobrepostas do Andor a encaixarem exatamente no mesmo número.
        df['lambda'] = np.round(df['lambda'], 2)
        
        df_processado = df.groupby('lambda', as_index=False).mean()
        return df_processado['lambda'].values, df_processado['intensidade'].values
        
    raise ValueError("Não foi possível extrair dados válidos da seção [data section].")

def filtrar_ruido(intensidade, janela=11, ordem=2):
    if janela % 2 == 0: janela += 1  
    if len(intensidade) <= janela: return intensidade
    return savgol_filter(intensidade, window_length=janela, polyorder=ordem)

def remover_baseline(intensidade, grau=2, iteracoes=10):
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
    area_total = np.trapezoid(intensidade, comprimento_onda)
    return intensidade / area_total if area_total != 0 else intensidade

def identificar_picos_npk(comprimentos_onda, intensidades, tolerancia=1.0, prom_frac=0.003):
    """
    Identificador ultra-sensível para N, P e K. 
    Aumenta a tolerância para 1.0 nm e reduz drasticamente o prom_frac 
    para capturar os picos fracos de Fósforo (P) na região do UV.
    """
    # prom_frac muito baixo (0.3% da altura máxima) para abraçar os picos baixos do Fósforo no UV
    prominencia_min = np.max(intensidades) * prom_frac
    
    # Reduzimos a distância para permitir picos múltiplos próximos (comum em doublets)
    picos_idx, _ = find_peaks(intensidades, prominence=prominencia_min, distance=2)
    picos_identificados = []
    
    for idx in picos_idx:
        w_pico, i_pico = comprimentos_onda[idx], intensidades[idx]
        melhor_elemento, linha_ref, menor_diff = None, None, float('inf')
        
        for elemento, linhas in ELEMENTS_WAVELENGTHS.items():
            for linha in linhas:
                diff = abs(w_pico - linha)
                # Tolerância de 1.0 nm para absorver pequenos desvios de calibração do espectrómetro no UV
                if diff <= tolerancia and diff < menor_diff:
                    menor_diff, melhor_elemento, linha_ref = diff, elemento, linha
                    
        if melhor_elemento:
            picos_identificados.append({
                'index': idx, 'elemento': melhor_elemento, 'w_medido': w_pico,
                'w_referencia': linha_ref, 'intensidade': i_pico, 'desvio': menor_diff
            })
            
    return picos_identificados
