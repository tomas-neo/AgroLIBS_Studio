# Arquivo: modulos/motor_outliers.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest # <-- A nossa nova arma secreta (Machine Learning)
import os
from tkinter import messagebox

def executar_caca_outliers(lista_arquivos_csv):
    """
    Lê os ficheiros .csv, aplica a PCA e usa Machine Learning (Isolation Forest)
    para rastrear e isolar as amostras aberrantes.
    """
    if not lista_arquivos_csv:
        raise ValueError("Nenhum arquivo para processar.")
        
    # Interceptador UV/VIS
    resp = messagebox.askyesnocancel(
        "Filtro de Região Espectral", 
        "Quais dados óticos você quer jogar na Caça aos Outliers?\n\n[SIM] = Processar SÓ UV\n[NÃO] = Processar SÓ VISÍVEL"
    )
    
    if resp is True: regiao_alvo = "UV"
    elif resp is False: regiao_alvo = "Vis"
    else: return 0, []
        
    lista_filtrada = []
    for arq in lista_arquivos_csv:
        caminho_padrao = arq.replace('\\', '/')
        if f"/{regiao_alvo.lower()}/" in caminho_padrao.lower():
            lista_filtrada.append(arq)
            
    if not lista_filtrada:
        messagebox.showerror("Erro de Filtro", f"Nenhum arquivo {regiao_alvo} encontrado!")
        return 0, []
        
    lista_arquivos_csv = lista_filtrada

    dados = []
    caminhos_absolutos = []
    
    # Leitura Blindada
    for arq in lista_arquivos_csv:
        df = pd.read_csv(arq, sep=';')
        if len(df.columns) == 1:
            df = pd.read_csv(arq, sep=',')
            
        coluna_y = None
        for col in df.columns:
            if 'intens' in col.lower():
                coluna_y = col 
                
        if coluna_y is None: continue
            
        dados.append(df[coluna_y].values)
        caminhos_absolutos.append(arq)
        
    if not dados: return 0, []

    # Vacina de Alinhamento LIBS
    tamanho_minimo = min(len(d) for d in dados)
    dados = [d[:tamanho_minimo] for d in dados]
        
    matriz_dados = np.array(dados)
    
    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(matriz_dados)
    
    pca = PCA(n_components=2)
    escores = pca.fit_transform(dados_padronizados)
    
    # =========================================================================
    # O EXTERMINADOR COM INTELIGÊNCIA ARTIFICIAL
    # =========================================================================
    # contamination=0.03 significa que ele vai caçar e isolar os 3% mais 
    # extremos do conjunto de dados, garantindo que o bandido não escape.
    iso = IsolationForest(contamination=0.03, random_state=42)
    previsoes = iso.fit_predict(escores)
    
    # O Isolation Forest devolve -1 para anomalias e 1 para dados normais
    outliers_indices = np.where(previsoes == -1)[0]
    
    lista_ruins = [caminhos_absolutos[i] for i in outliers_indices]
    
    return len(lista_ruins), lista_ruins
