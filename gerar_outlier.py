# Arquivo: gerar_outlier.py
"""
Ferramenta de teste: cria um outlier FALSO a partir de um .csv tratado,
para conferir se a Aba 3 (Outliers) consegue encontrá-lo.
Uso: python gerar_outlier.py  (abre uma janela para escolher o .csv)
"""
import os

import numpy as np
import pandas as pd
from tkinter import filedialog
import customtkinter as ctk

from modulos.dados import ler_csv

# Abre uma janelinha rápida só para você escolher um CSV real
ctk.CTk().withdraw()
caminho_real = filedialog.askopenfilename(
    title="Selecione UM arquivo .csv processado normal",
    filetypes=[("Arquivos CSV", "*.csv")]
)

if caminho_real:
    # 1. Lê o arquivo original (mesmo formato que a Aba 1 grava: separador ';')
    eixo, intensidade = ler_csv(caminho_real)
    if intensidade is None:
        raise SystemExit("Esse .csv não tem coluna de intensidade.")

    # 2. "Estraga" os dados para criar uma anomalia gigantesca!
    # Multiplica a intensidade por 50 e soma um ruído do tamanho do sinal
    rng = np.random.default_rng()
    falso = intensidade * 50 + rng.normal(0, np.std(intensidade) * 5, len(intensidade))

    # 3. Salva o arquivo falso na mesma pasta, no mesmo formato
    caminho_falso = os.path.join(os.path.dirname(caminho_real), "ZZ_AMOSTRA_OUTLIER_FALSO.csv")
    pd.DataFrame({"Comprimento_Onda": eixo, "Intensidade": falso}).to_csv(caminho_falso, index=False, sep=";")

    print("=" * 50)
    print("🚨 OUTLIER FALSO CRIADO COM SUCESSO! 🚨")
    print(f"Salvo em: {caminho_falso}")
    print("=" * 50)
else:
    print("Você cancelou a seleção.")
