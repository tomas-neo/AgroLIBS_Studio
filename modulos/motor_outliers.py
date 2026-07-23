# Arquivo: modulos/motor_outliers.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os

def executar_caca_outliers(lista_arquivos_csv):
    """
    Lê os ficheiros .csv, aplica a PCA e usa a estatística Z-Score 
    para detetar amostras com desvios anormais.
    """
    dados = []
    caminhos_absolutos = []
    
    # Lemos os novos CSVs
    for arq in lista_arquivos_csv:
        df = pd.read_csv(arq, sep=';')
        dados.append(df['Intensidade'].values)
        caminhos_absolutos.append(arq) # Guardamos o caminho completo para a quarentena!
        
    matriz_dados = np.array(dados)
    
    # Padronização
    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(matriz_dados)
    
    # Calculamos os escores da PCA
    pca = PCA(n_components=2)
    escores = pca.fit_transform(dados_padronizados)
    
    # O detetive matemático: Z-Score (calcula a distância do ponto para o centro do grupo)
    z_scores = np.abs(stats.zscore(escores))
    
    # Regra estatística: Se estiver a mais de 3 desvios padrão de distância, é anomalia!
    outliers_indices = np.where((z_scores > 3).any(axis=1))[0]
    
    # Criamos a "Lista Negra" com o endereço dos ficheiros maus
    lista_ruins = [caminhos_absolutos[i] for i in outliers_indices]
    
    return len(lista_ruins), lista_ruins
