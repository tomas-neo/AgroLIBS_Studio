# Arquivo: main.py
import customtkinter as ctk
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import pandas as pd

# AQUI ESTÁ A MÁGICA DA MODULARIZAÇÃO!
# Nota: Adicione a sua função 'carregar_arquivo_esf' no arquivo pre_processador.py!
from modulos.pre_processador import filtrar_ruido, remover_baseline, normalizar_por_area, carregar_arquivo_esf

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

janela = ctk.CTk()
janela.title("AgroLIBS Studio - Pré-processamento")
janela.geometry("650x750") 

pasta_selecionada = ""
arquivos_esf_encontrados = []

# ==============================================================================
# LÓGICA DO BACK-END
# ==============================================================================
def escolher_pasta():
    global pasta_selecionada, arquivos_esf_encontrados
    pasta = filedialog.askdirectory(title="Selecione a pasta com os arquivos .esf")
    
    if pasta:
        pasta_selecionada = pasta
        label_caminho.configure(text=f"Pasta: {pasta_selecionada}", text_color="green")
        
        arquivos_esf_encontrados = [f for f in os.listdir(pasta) if f.endswith('.esf')]
        
        if arquivos_esf_encontrados:
            menu_amostras.configure(values=arquivos_esf_encontrados, state="normal")
            menu_amostras.set(arquivos_esf_encontrados[0]) 
            botao_preview.configure(state="normal")
            botao_processar.configure(state="normal")
            texto_log.insert("end", f"[{len(arquivos_esf_encontrados)}] arquivos .esf encontrados.\n")
        else:
            texto_log.insert("end", "⚠️ Nenhum arquivo .esf encontrado nesta pasta.\n")

def processar_espectro(intensidade, wl):
    """Função auxiliar para aplicar os filtros selecionados pelo usuário."""
    intens_tratada = intensidade.copy()
    sufixo = ""
    
    if check_savgol.get() == 1:
        intens_tratada = filtrar_ruido(intens_tratada)
        sufixo += "_SG"
        
    if check_baseline.get() == 1:
        intens_tratada, _ = remover_baseline(intens_tratada)
        sufixo += "_BL"
        
    if check_normalizacao.get() == 1:
        intens_tratada = normalizar_por_area(wl, intens_tratada)
        sufixo += "_Norm"
        
    if sufixo == "":
        sufixo = "_Bruto"
        
    return intens_tratada, sufixo

def testar_filtros_preview():
    amostra_escolhida = menu_amostras.get()
    caminho_completo = os.path.join(pasta_selecionada, amostra_escolhida)
    texto_log.insert("end", f"Gerando preview para: {amostra_escolhida}\n")
    
    try:
        # 1. Lê o arquivo original
        wl, intens_bruta = carregar_arquivo_esf(caminho_completo)
        
        # 2. Aplica os filtros escolhidos nas checkboxes
        intens_tratada, sufixo = processar_espectro(intens_bruta, wl)
        
        # 3. Desenha o gráfico "Antes e Depois"
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        ax1.plot(wl, intens_bruta, color='#7f8c8d', linewidth=1)
        ax1.set_title(f"Espectro Bruto: {amostra_escolhida}", fontweight='bold')
        ax1.set_ylabel("Intensidade Absoluta")
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        ax2.plot(wl, intens_tratada, color='#2c3e50', linewidth=1)
        ax2.set_title(f"Espectro Tratado (Tags: {sufixo})", fontweight='bold')
        ax2.set_xlabel("Comprimento de Onda (nm)")
        ax2.set_ylabel("Intensidade")
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.show() 
        
    except Exception as e:
        texto_log.insert("end", f"❌ Erro ao gerar preview: {str(e)}\n")

def executar_processamento_lote():
    texto_log.insert("end", "="*40 + "\n")
    texto_log.insert("end", "Iniciando processamento em lote...\n")
    
    pasta_saida = os.path.join(pasta_selecionada, "LIBS_PROCESSADOS_CSV")
    os.makedirs(pasta_saida, exist_ok=True)
    
    sucessos = 0
    for arquivo in arquivos_esf_encontrados:
        try:
            caminho_completo = os.path.join(pasta_selecionada, arquivo)
            nome_base = os.path.splitext(arquivo)[0]
            
            # Lê e processa
            wl, intens_bruta = carregar_arquivo_esf(caminho_completo)
            intens_final, sufixo = processar_espectro(intens_bruta, wl)
            
            # Monta o nome com as tags e salva em CSV
            nome_arquivo_final = f"{nome_base}{sufixo}.csv"
            caminho_salvamento = os.path.join(pasta_saida, nome_arquivo_final)
            
            df_saida = pd.DataFrame({
                'comprimento_onda_nm': wl,
                'intensidade_processada': intens_final
            })
            df_saida.to_csv(caminho_salvamento, index=False)
            sucessos += 1
            
        except Exception as e:
            texto_log.insert("end", f"Erro em {arquivo}: {str(e)}\n")
            
    texto_log.insert("end", f"✅ Concluído! {sucessos} arquivos salvos na pasta:\n{pasta_saida}\n")
    texto_log.insert("end", "="*40 + "\n")

# ==============================================================================
# INTERFACE GRÁFICA (FRONT-END)
# ==============================================================================
botao_pasta = ctk.CTkButton(janela, text="📁 Selecionar Pasta dos .esf", command=escolher_pasta)
botao_pasta.pack(pady=10)

label_caminho = ctk.CTkLabel(janela, text="Nenhuma pasta selecionada.", text_color="gray")
label_caminho.pack(pady=5)

# --- SEÇÃO: PREVIEW ---
frame_preview = ctk.CTkFrame(janela)
frame_preview.pack(pady=10, padx=20, fill="x")

titulo_preview = ctk.CTkLabel(frame_preview, text="1. Selecione uma amostra para testar os filtros:")
titulo_preview.pack(pady=5)

menu_amostras = ctk.CTkOptionMenu(frame_preview, values=["Selecione a pasta primeiro..."], state="disabled")
menu_amostras.pack(pady=5)

botao_preview = ctk.CTkButton(frame_preview, text="👁️ Mostrar Gráfico de Preview", command=testar_filtros_preview, state="disabled")
botao_preview.pack(pady=10)

# --- SEÇÃO: FILTROS E LOTE ---
frame_filtros = ctk.CTkFrame(janela)
frame_filtros.pack(pady=10, padx=20, fill="x")

titulo_filtros = ctk.CTkLabel(frame_filtros, text="2. Filtros a serem aplicados (ordem sequencial):")
titulo_filtros.pack(pady=5)

check_savgol = ctk.CTkCheckBox(frame_filtros, text="Filtro Savitzky-Golay (Remover Ruído)")
check_savgol.pack(pady=5, anchor="w", padx=20)
check_savgol.select() 

check_baseline = ctk.CTkCheckBox(frame_filtros, text="Correção de Linha de Base (Baseline)")
check_baseline.pack(pady=5, anchor="w", padx=20)
check_baseline.select()

check_normalizacao = ctk.CTkCheckBox(frame_filtros, text="Normalização por Área")
check_normalizacao.pack(pady=5, anchor="w", padx=20)
check_normalizacao.select()

botao_processar = ctk.CTkButton(janela, text="▶️ Processar e Salvar Lote", command=executar_processamento_lote, state="disabled", fg_color="green", hover_color="darkgreen")
botao_processar.pack(pady=15)

texto_log = ctk.CTkTextbox(janela, height=120)
texto_log.pack(padx=20, fill="x", pady=10)

janela.mainloop()
