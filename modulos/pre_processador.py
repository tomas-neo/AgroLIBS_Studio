# Arquivo: modulos/pre_processador.py
import numpy as np
from scipy.signal import savgol_filter

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
    # Retorna o espectro corrigido e a linha que foi removida (para o gráfico de preview)
    return np.clip(espectro_corrigido, 0, None), baseline_estimado

def normalizar_por_area(comprimento_onda, intensidade):
    """Normaliza o espectro dividindo pela área total sob a curva."""
    area_total = np.trapz(intensidade, comprimento_onda)
    return intensidade / area_total if area_total != 0 else intensidade
