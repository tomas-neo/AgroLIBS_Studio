# Arquivo: modulos/motor_variancia.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def executar_curva_variancia(lista_arquivos_csv):
    """
    Lê os ficheiros .csv limpos, aplica a PCA e desenha a curva
    de variância acumulada para descobrir quantos PCs são necessários.
    """
    dados = []
    
    # 1. Carregamento dos dados limpos
    for arq in lista_arquivos_csv:
        df = pd.read_csv(arq, sep=';')
        dados.append(df['Intensidade'].values)
        
    matriz_dados = np.array(dados)
    
    # 2. Padronização
    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(matriz_dados)
    
    # 3. O Motor PCA (Sem limite de componentes para ver o todo)
    pca = PCA()
    pca.fit(dados_padronizados)
    
    # 4. A Matemática da Variância
    # np.cumsum soma as percentagens em cascata (ex: PC1=60%, PC2=20% -> Acumulado=80%)
    variancia_acumulada = np.cumsum(pca.explained_variance_ratio_) * 100
    
    # Descobre matematicamente onde cruzamos a linha dos 95%
    limite = 95.0
    num_pcs_95 = np.argmax(variancia_acumulada >= limite) + 1
    
    # 5. O Gráfico Analítico
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(variancia_acumulada) + 1), variancia_acumulada, 
             marker='o', linestyle='-', color='#2980b9', linewidth=2)
    
    # Desenha a linha de meta (95%) e a linha vertical onde a cruzamos
    plt.axhline(y=limite, color='r', linestyle='--', label=f'Meta de {limite}% de Informação')
    plt.axvline(x=num_pcs_95, color='g', linestyle='--', label=f'{num_pcs_95} PCs necessários')
    
    plt.title('Curva de Variância Acumulada - AgroLIBS', fontsize=14, fontweight='bold')
    plt.xlabel('Número de Componentes Principais (PCs)', fontsize=12)
    plt.ylabel('Variância Explicada Acumulada (%)', fontsize=12)
    
    # Coloca os números exatos em cima das bolinhas (até ao limite de 10 PCs para não poluir)
    for i, var in enumerate(variancia_acumulada[:10]):
        plt.text(i + 1, var + 1.5, f'{var:.1f}%', ha='center', fontsize=9)

    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
