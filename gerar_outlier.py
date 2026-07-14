import pandas as pd
import numpy as np
import os
from tkinter import filedialog
import customtkinter as ctk

# Abre uma janelinha rápida só para você escolher um CSV real
ctk.CTk().withdraw() 
caminho_real = filedialog.askopenfilename(
    title="Selecione UM arquivo .csv processado normal",
    filetypes=[("Arquivos CSV", "*.csv")]
)

if caminho_real:
    # 1. Lê o arquivo original perfeito
    df = pd.read_csv(caminho_real)
    
    # 2. "Estraga" os dados para criar uma anomalia gigantesca!
    # Vamos multiplicar toda a intensidade por 50 e somar um ruído louco
    ruido = np.random.normal(0, 50, len(df))
    df['intensidade_processada'] = (df['intensidade_processada'] * 50) + ruido
    
    # 3. Salva o arquivo falso na mesma pasta
    pasta_origem = os.path.dirname(caminho_real)
    caminho_falso = os.path.join(pasta_origem, "ZZ_AMOSTRA_OUTLIER_FALSO.csv")
    
    df.to_csv(caminho_falso, index=False)
    
    print("="*50)
    print("🚨 OUTLIER FALSO CRIADO COM SUCESSO! 🚨")
    print(f"Salvo em: {caminho_falso}")
    print("="*50)
else:
    print("Você cancelou a seleção.")
