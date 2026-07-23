# Arquivo: modulos/motor_pca.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

def executar_pca_arquivos(lista_arquivos_csv):
    """
    Lê os ficheiros CSV, extrai o nome da subpasta de cada um para usar como 
    categoria, aplica PCA e gera um gráfico 3D colorido com legendas.
    """
    dados = []
    categorias = [] # Lista para guardar o nome das pastas (ex: NPK101010-PF)
    
    # 1. Carregamento e Extração de Etiquetas
    for arq in lista_arquivos_csv:
        df = pd.read_csv(arq, sep=';')
        dados.append(df['Intensidade'].values)
        
        # MAGIA AQUI: Pega o caminho todo (ex: C:/dados/NPK/amostra1.csv)
        # Extrai só a pasta mãe do arquivo (ex: NPK)
        nome_pasta = os.path.basename(os.path.dirname(arq))
        if not nome_pasta:
            nome_pasta = "Amostras"
            
        categorias.append(nome_pasta)
        
    matriz_dados = np.array(dados)
    
    # 2. Padronização (Z-Score)
    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(matriz_dados)
    
    # 3. O Motor PCA 3D
    pca = PCA(n_components=3)
    escores = pca.fit_transform(dados_padronizados)
    var_exp = pca.explained_variance_ratio_ * 100
    
    # 4. A Magia Visual: Gráfico 3D Colorido por Pastas
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Descobre quantas pastas diferentes existem (Categorias únicas)
    categorias_unicas = list(set(categorias))
    
    # Pega uma paleta de cores bonita do matplotlib
    paleta_cores = plt.get_cmap('tab10') 
    
    # Vamos desenhar os pontos grupo por grupo (pasta por pasta)
    for i, cat in enumerate(categorias_unicas):
        # Acha todos os índices (posições) dos ficheiros que pertencem a esta pasta
        indices = [j for j, x in enumerate(categorias) if x == cat]
        
        # Separa o X, Y, Z só desta pasta
        x_cat = escores[indices, 0]
        y_cat = escores[indices, 1]
        z_cat = escores[indices, 2]
        
        # Plota apenas os pontos desta pasta com uma cor específica e põe a etiqueta
        ax.scatter(x_cat, y_cat, z_cat, 
                   color=paleta_cores(i), marker='o', s=60, alpha=0.8, label=cat)
    
    ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}%)', fontweight='bold')
    ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}%)', fontweight='bold')
    ax.set_zlabel(f'PC3 ({var_exp[2]:.1f}%)', fontweight='bold')
    
    plt.title('Análise de Componentes Principais (PCA 3D)\nAgrupado por Amostras', fontsize=14)
    
    # Adiciona a legenda bonita no canto de fora do gráfico
    plt.legend(title="Categorias (Pastas)", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show() 
    
    return escores, categorias, pca
