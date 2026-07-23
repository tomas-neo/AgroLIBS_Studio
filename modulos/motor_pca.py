# Arquivo: modulos/motor_pca.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from tkinter import messagebox

def executar_pca_arquivos(lista_arquivos_csv):
    """
    Motor PCA Híbrido:
    - Filtra por UV/Visível
    - Escolhe entre 2D e 3D
    - Agrupa por Concentração + Marca
    """
    if not lista_arquivos_csv:
        raise ValueError("Nenhum arquivo para processar.")
        
    # =========================================================================
    # 1. O INTERCEPTADOR DE REGIÃO ESPECTRAL
    # =========================================================================
    resp = messagebox.askyesnocancel(
        "Filtro de Região Espectral", 
        "Quais dados óticos você quer jogar no gráfico?\n\n[SIM] = Processar SÓ UV\n[NÃO] = Processar SÓ VISÍVEL"
    )
    
    if resp is True:
        regiao_alvo = "UV"
    elif resp is False:
        regiao_alvo = "Vis"
    else:
        return None, None, None

    # =========================================================================
    # 2. O INTERCEPTADOR DE DIMENSÃO (2D ou 3D)
    # =========================================================================
    resp_dim = messagebox.askyesnocancel(
        "Dimensão do Gráfico",
        "Como você quer o gráfico do PCA?\n\n[SIM] = Gráfico 2D (Ideal para relatórios)\n[NÃO] = Gráfico 3D (Ideal para visualização)"
    )
    
    if resp_dim is None:
        return None, None, None # Se o utilizador fechar a janela, o programa aborta sem erros
        
    is_2d = resp_dim # True para 2D, False para 3D

    # Filtra os ficheiros pela região (UV ou VIS)
    lista_filtrada = []
    for arq in lista_arquivos_csv:
        caminho_padrao = arq.replace('\\', '/')
        if f"/{regiao_alvo.lower()}/" in caminho_padrao.lower():
            lista_filtrada.append(arq)
            
    if not lista_filtrada:
        messagebox.showerror("Erro de Filtro", f"Nenhum arquivo da região '{regiao_alvo}' foi encontrado nesta pasta!")
        return None, None, None
        
    lista_arquivos_csv = lista_filtrada
    # =========================================================================
        
    dados = []
    categorias = [] 
    
    pasta_raiz_comum = os.path.commonpath(lista_arquivos_csv)
    if os.path.isfile(pasta_raiz_comum):
        pasta_raiz_comum = os.path.dirname(pasta_raiz_comum)
        
    for arq in lista_arquivos_csv:
        # Leitura blindada
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
        
        # Mágica da Legenda (Concentração + Marca)
        caminho_relativo = os.path.relpath(arq, pasta_raiz_comum)
        partes_caminho = caminho_relativo.replace('\\', '/').split('/')
        
        if len(partes_caminho) >= 3:
            concentracao = partes_caminho[0]
            marca = partes_caminho[2]
            
            if 'maxgreen' in marca.lower(): marca = 'MaxGreen'
            elif 'plantfertil' in marca.lower(): marca = 'PlantFertil'
                
            nome_categoria = f"{concentracao} ({marca})"
        elif len(partes_caminho) > 0:
            nome_categoria = partes_caminho[0]
        else:
            nome_categoria = "Amostras Base"
            
        categorias.append(nome_categoria)
        
    if not dados:
        return None, None, None

    # Vacina LIBS para alinhar disparos
    tamanho_minimo = min(len(d) for d in dados)
    dados = [d[:tamanho_minimo] for d in dados]
        
    matriz_dados = np.array(dados)
    
    # Padronização e PCA
    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(matriz_dados)
    
    # Calcula as 3 primeiras componentes independentemente do gráfico
    pca = PCA(n_components=3)
    escores = pca.fit_transform(dados_padronizados)
    var_exp = pca.explained_variance_ratio_ * 100
    
    # Começa a desenhar o gráfico
    fig = plt.figure(figsize=(11, 8))
    categorias_unicas = list(set(categorias))
    paleta_cores = plt.get_cmap('tab10') 
    
    # Ramo A: Desenhar Gráfico 2D
    if is_2d:
        ax = fig.add_subplot(111) # Sem a projeção '3d'
        
        for i, cat in enumerate(categorias_unicas):
            indices = [j for j, x in enumerate(categorias) if x == cat]
            x_cat = escores[indices, 0]
            y_cat = escores[indices, 1]
            
            ax.scatter(x_cat, y_cat, color=paleta_cores(i), marker='o', s=60, alpha=0.8, edgecolors='white', label=cat)
            
        ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}%)', fontweight='bold')
        ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}%)', fontweight='bold')
        plt.title(f'Análise de Componentes Principais (PCA 2D) - {regiao_alvo.upper()}\nAgrupado por Concentração e Marca', fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.6) # Coloquei uma grelha pontilhada suave para o 2D ficar super profissional
        
    # Ramo B: Desenhar Gráfico 3D Tradicional
    else:
        ax = fig.add_subplot(111, projection='3d')
        
        for i, cat in enumerate(categorias_unicas):
            indices = [j for j, x in enumerate(categorias) if x == cat]
            x_cat = escores[indices, 0]
            y_cat = escores[indices, 1]
            z_cat = escores[indices, 2]
            
            ax.scatter(x_cat, y_cat, z_cat, color=paleta_cores(i), marker='o', s=60, alpha=0.8, edgecolors='white', label=cat)
            
        ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}%)', fontweight='bold')
        ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}%)', fontweight='bold')
        ax.set_zlabel(f'PC3 ({var_exp[2]:.1f}%)', fontweight='bold')
        plt.title(f'Análise de Componentes Principais (PCA 3D) - {regiao_alvo.upper()}\nAgrupado por Concentração e Marca', fontsize=14)

    # Legenda final igual para os dois!
    plt.legend(title="Amostras", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show() 
    
    return escores, categorias, pca
