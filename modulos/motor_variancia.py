# Ficheiro: modulos/motor_variancia.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

def executar_curva_variancia(lista_caminhos):
    """Lê os ficheiros, calcula a PCA e plota a variância explicada acumulada."""
    if len(lista_caminhos) < 3:
        raise ValueError("Selecione pelo menos 3 ficheiros para calcular a variância.")

    matriz_X = []
    tamanho_padrao = None

    for caminho in lista_caminhos:
        df = pd.read_csv(caminho)
        espectro = df['intensidade_processada'].values
        
        if tamanho_padrao is None:
            tamanho_padrao = len(espectro)
        elif len(espectro) != tamanho_padrao:
            continue
            
        matriz_X.append(espectro)

    matriz_X = np.array(matriz_X)

    # Matemática da Variância
    scaler = StandardScaler()
    X_escalonado = scaler.fit_transform(matriz_X)

    # Não podemos pedir mais PCs do que o número de amostras
    n_comp = min(20, len(matriz_X)) 
    pca = PCA(n_components=n_comp)
    pca.fit(X_escalonado)
    
    var_explicada = pca.explained_variance_ratio_ * 100
    var_acumulada = np.cumsum(var_explicada)

    # Desenhando o Gráfico
    plt.figure(figsize=(10, 6))
    indices = np.arange(1, n_comp + 1)
    plt.plot(indices, var_acumulada, marker='o', color='#3b2c85', linewidth=2, markersize=8)

    # Destaca o ponto onde atingimos 95% de explicação
    limite_95 = np.where(var_acumulada >= 95)[0]
    if len(limite_95) > 0:
        pc_alvo = limite_95[0] + 1
        val_alvo = var_acumulada[limite_95[0]]
        
        plt.axvline(x=pc_alvo, color='#cc3333', linestyle='--', linewidth=1.5)
        plt.axhline(y=val_alvo, color='#cc3333', linestyle='--', linewidth=1.5)
        
        plt.annotate(f'PCA{pc_alvo}\n({val_alvo:.1f}%)',
                     xy=(pc_alvo, val_alvo),
                     xytext=(pc_alvo - 3, val_alvo + 2.5),
                     arrowprops=dict(facecolor='#cc3333', edgecolor='#cc3333', shrink=0.05, width=2, headwidth=8),
                     color='#cc3333', fontsize=12, fontweight='bold', ha='center')

    plt.title('📈 Variância Explicada Acumulada - PCA', fontsize=14, fontweight='bold')
    plt.xlabel('Índices dos Componentes Principais (PCs)', fontsize=12)
    plt.ylabel('% Variância Explicada Acumulada', fontsize=12)
    
    plt.xticks(indices)
    plt.grid(True, linestyle='-', alpha=0.6)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.show()
