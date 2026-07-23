# Arquivo: modulos/motor_variancia.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from tkinter import messagebox

def executar_curva_variancia(lista_arquivos_csv):
    """
    Motor da Curva de Variância:
    Aplica o estilo visual personalizado, limitando a 20 PCs para
    uma visualização limpa, com escudo Anti-Zumbi e filtro espectral.
    """
    if not lista_arquivos_csv:
        raise ValueError("Nenhum arquivo para processar.")
        
    # =========================================================================
    # 1. O INTERCEPTADOR DE REGIÃO ESPECTRAL
    # =========================================================================
    resp = messagebox.askyesnocancel(
        "Filtro de Região Espectral", 
        "Quais dados óticos você quer jogar na Curva de Variância?\n\n[SIM] = Processar SÓ UV\n[NÃO] = Processar SÓ VISÍVEL"
    )
    
    if resp is True:
        regiao_alvo = "UV"
    elif resp is False:
        regiao_alvo = "Vis"
    else:
        return 
        
    lista_filtrada = []
    for arq in lista_arquivos_csv:
        # ESCUDO ANTI-ZUMBI: Ignora a quarentena se ela foi passada por acidente
        if "QUARENTENA" in arq:
            continue
            
        caminho_padrao = arq.replace('\\', '/')
        if f"/{regiao_alvo.lower()}/" in caminho_padrao.lower():
            lista_filtrada.append(arq)
            
    if not lista_filtrada:
        messagebox.showerror("Erro de Filtro", f"Nenhum arquivo da região '{regiao_alvo}' foi encontrado nesta pasta!")
        return
        
    lista_arquivos_csv = lista_filtrada

    # =========================================================================
    # 2. LEITURA BLINDADA
    # =========================================================================
    dados = []
    for arq in lista_arquivos_csv:
        df = pd.read_csv(arq, sep=';')
        if len(df.columns) == 1:
            df = pd.read_csv(arq, sep=',')
            
        coluna_y = None
        for col in df.columns:
            if 'intens' in col.lower():
                coluna_y = col 
                
        if coluna_y is None:
            continue
            
        dados.append(df[coluna_y].values)
        
    if not dados:
        return

    # VACINA LIBS: Garante que todos os disparos tenham exatamente a mesma quantidade de linhas
    tamanho_minimo = min(len(d) for d in dados)
    dados = [d[:tamanho_minimo] for d in dados]
        
    matriz_X = np.array(dados)
    
    # =========================================================================
    # 3. MATEMÁTICA DA VARIÂNCIA (ESTILO LIMITADO)
    # =========================================================================
    scaler = StandardScaler()
    X_escalonado = scaler.fit_transform(matriz_X)

    # Limita a 20 PCs (ou ao número máximo de amostras se for menor que 20)
    n_comp = min(20, len(matriz_X))
    pca = PCA(n_components=n_comp)
    pca.fit(X_escalonado)
    
    var_explicada = pca.explained_variance_ratio_ * 100
    var_acumulada = np.cumsum(var_explicada)

    # =========================================================================
    # 4. O GRÁFICO ANALÍTICO (DESIGN PERSONALIZADO)
    # =========================================================================
    plt.figure(figsize=(10, 6))
    indices = np.arange(1, n_comp + 1)
    
    # A sua cor escolhida: #3b2c85
    plt.plot(indices, var_acumulada, marker='o', color='#3b2c85', linewidth=2, markersize=8)

    # Procura o limite de 95%. Como cortamos em 20 PCs, pode não encontrar, mas a lógica previne erros!
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

    # Anotações dos números nas bolinhas (Fica perfeito agora que são só 20 pontos)
    for i, var in enumerate(var_acumulada):
        plt.text(i + 1, var + 1.5, f'{var:.1f}%', ha='center', fontsize=9)

    plt.title(f'Variância Explicada Acumulada - PCA ({regiao_alvo.upper()})', fontsize=14, fontweight='bold')
    plt.xlabel('Número de Componentes Principais (PCs)', fontsize=12)
    plt.ylabel('% Variância Explicada Acumulada', fontsize=12)
    
    plt.xticks(indices)
    plt.grid(True, linestyle='-', alpha=0.6)
    
    # Deixa um respiro dinâmico no topo do gráfico para o texto não ficar cortado
    limite_superior_y = max(105, var_acumulada[-1] + 10)
    plt.ylim(0, limite_superior_y)
    
    plt.tight_layout()
    plt.show()
