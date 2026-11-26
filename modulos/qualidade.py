# Arquivo: modulos/qualidade.py
"""
Controle de qualidade de cada tiro do laser, feito no espectro BRUTO (.esf),
antes de qualquer filtro ou normalização.

Por que no bruto: a normalização por área "estica" um tiro fraco até ele ficar
com a mesma escala de um tiro bom. Depois disso, o ruído dele parece sinal, e
o Isolation Forest da Aba 3 (que sempre marca uma fração fixa) não consegue
separar esses tiros quando eles são muitos.

Dois critérios:

1. Tiro fraco (relação sinal/ruído baixa)
   - ruído = desvio-padrão robusto da diferença entre pixels vizinhos
     (1,4826 x MAD / raiz de 2). Em regiões sem linhas, isso mede só o ruído
     do detector; o MAD ignora os picos.
   - sinal = altura do pico mais alto acima da mediana do espectro.
   - SNR = sinal / ruído. Um tiro que não fez plasma fica perto de 5; tiros
     bons ficam acima de 1000.

2. Tiro saturado
   - algum pixel chegou ao teto do detector (perto de 581 mil contagens no
     ARYELLE/Andor, depois da soma das ordens). O pico achatado mente sobre a
     intensidade da linha.
"""
import numpy as np

SNR_MINIMO_PADRAO = 20.0
TETO_SATURACAO_PADRAO = 578000.0


def estimar_ruido(intensidade):
    """Desvio-padrão robusto do ruído pixel a pixel."""
    y = np.asarray(intensidade, dtype=float)
    if len(y) < 3:
        return 0.0
    d = np.diff(y)
    mad = np.median(np.abs(d - np.median(d)))
    return 1.4826 * mad / np.sqrt(2)


def avaliar_tiro(intensidade, snr_minimo=SNR_MINIMO_PADRAO, teto_saturacao=TETO_SATURACAO_PADRAO):
    """
    Devolve um dicionário com:
      snr, maximo, ruido, pixels_saturados, aprovado (bool) e motivo
      ('ok', 'fraco', 'saturado' ou 'fraco+saturado').
    snr_minimo=None ou 0 desliga o critério de tiro fraco;
    teto_saturacao=None ou 0 desliga o de saturação.
    """
    y = np.asarray(intensidade, dtype=float)
    maximo = float(np.max(y)) if len(y) else 0.0
    ruido = estimar_ruido(y)
    sinal = maximo - float(np.median(y)) if len(y) else 0.0
    snr = sinal / ruido if ruido > 0 else (np.inf if sinal > 0 else 0.0)
    n_sat = int(np.sum(y >= teto_saturacao)) if teto_saturacao else 0

    problemas = []
    if snr_minimo and snr < snr_minimo:
        problemas.append("fraco")
    if n_sat > 0:
        problemas.append("saturado")
    return {
        "snr": float(snr),
        "maximo": maximo,
        "ruido": float(ruido),
        "pixels_saturados": n_sat,
        "aprovado": not problemas,
        "motivo": "+".join(problemas) if problemas else "ok",
    }
