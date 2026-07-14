# Arquivo: modulos/motor_pca.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os

def executar_pca_arquivos(lista_caminhos):
    """Lê os arquivos selecionados, padroniza e plota o gráfico da PCA."""
    if len(lista_caminhos) < 3:
        raise ValueError("Selecione pelo menos 3 arquivos para gerar uma PCA.")

    matriz_X = []
    labels_amostras = []
    tamanho_padrao = None

    # 1. Montagem da Matriz de Dados
    for caminho in lista_caminhos:
        df = pd.read_csv(caminho)
        # Puxa a coluna que salvamos no pré-processamento
        espectro = df['intensidade_processada'].values
        
        if tamanho_padrao is None:
            tamanho_padrao = len(espectro)
        elif len(espectro) != tamanho_padrao:
            continue # Pula se os comprimentos de onda forem diferentes
            
        matriz_X.append(espectro)
        
        # Usa o nome do arquivo para legendar o gráfico
        nome_base = os.path.basename(caminho).replace(".csv", "")
        labels_amostras.append(nome_base)

    matriz_X = np.array(matriz_X)
    labels_amostras = np.array(labels_amostras)

    if len(matriz_X) < 2:
        raise ValueError("Arquivos insuficientes ou dados incompatíveis.")

    # 2. Cálculo da PCA
    scaler = StandardScaler()
    X_escalonado = scaler.fit_transform(matriz_X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_escalonado)
    
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100

    # 3. Construção do Gráfico
    plt.figure(figsize=(10, 7))
    df_pca = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Amostra': labels_amostras
    })

    sns.scatterplot(
        x='PC1', y='PC2', 
        hue='Amostra', 
        palette='Set1', 
        data=df_pca, 
        s=100, 
        alpha=0.8, 
        edgecolor='black'
    )

    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.title('PCA - Gráfico de Escores (Arquivos Selecionados)', fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1 ({var_pc1:.1f}% da variância)', fontsize=12)
    plt.ylabel(f'PC2 ({var_pc2:.1f}% da variância)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
