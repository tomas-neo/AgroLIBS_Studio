# Arquivo: modulos/motor_outliers.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy import stats
import os

def executar_caca_outliers(lista_caminhos):
    """Lê os arquivos, roda uma PCA preliminar e detecta outliers usando Z-Score."""
    if len(lista_caminhos) < 3:
        raise ValueError("Selecione pelo menos 3 arquivos para a análise estatística.")

    matriz_X = []
    labels_amostras = []
    tamanho_padrao = None

    for caminho in lista_caminhos:
        df = pd.read_csv(caminho)
        espectro = df['intensidade_processada'].values
        
        if tamanho_padrao is None:
            tamanho_padrao = len(espectro)
        elif len(espectro) != tamanho_padrao:
            continue
            
        matriz_X.append(espectro)
        labels_amostras.append(os.path.basename(caminho).replace(".csv", ""))

    matriz_X = np.array(matriz_X)
    labels_amostras = np.array(labels_amostras)

    # Cálculo Estatístico
    scaler = StandardScaler()
    X_escalonado = scaler.fit_transform(matriz_X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_escalonado)
    
    # Detecção de anomalias (Distância > 3 Desvios Padrões)
    z_scores = np.abs(stats.zscore(X_pca))
    outlier_mask = (z_scores >= 3).any(axis=1)
    
    total_outliers = np.sum(outlier_mask)
    status_pontos = ['🚨 Anomalia (Outlier)' if out else '✅ Normal' for out in outlier_mask]

    # Construção do Gráfico
    plt.figure(figsize=(10, 7))
    df_pca = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Amostra': labels_amostras,
        'Status': status_pontos
    })

    # Plota os normais
    sns.scatterplot(
        x='PC1', y='PC2', hue='Amostra', palette='Set1', 
        data=df_pca[df_pca['Status'] == '✅ Normal'], 
        s=100, alpha=0.6, edgecolor='black'
    )
    
    # Plota os outliers bem destacados (vermelhos e com 'X')
    if total_outliers > 0:
        sns.scatterplot(
            x='PC1', y='PC2', data=df_pca[df_pca['Status'] == '🚨 Anomalia (Outlier)'], 
            s=200, marker='X', color='red', edgecolor='black', label='Outliers Detectados'
        )

    plt.title(f'🎯 Detecção de Outliers (Total Encontrado: {total_outliers})', fontsize=14, fontweight='bold')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    return total_outliers, df_pca[df_pca['Status'] == '🚨 Anomalia (Outlier)']['Amostra'].tolist()
