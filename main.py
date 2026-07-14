import customtkinter as ctk
from tkinter import filedialog
import os
import matplotlib.pyplot as plt

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

janela = ctk.CTk()
janela.title("Pré-processamento - Embrapa")
janela.geometry("600x650") # Aumentei um pouco a altura para caber os novos botões

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
        
        # Procura os arquivos .esf na pasta
        arquivos_esf_encontrados = [f for f in os.listdir(pasta) if f.endswith('.esf')]
        
        if arquivos_esf_encontrados:
            # Atualiza o menu suspenso com os arquivos encontrados
            menu_amostras.configure(values=arquivos_esf_encontrados, state="normal")
            menu_amostras.set(arquivos_esf_encontrados[0]) # Seleciona o primeiro por padrão
            botao_preview.configure(state="normal")
            botao_processar.configure(state="normal")
            texto_log.insert("end", f"[{len(arquivos_esf_encontrados)}] arquivos .esf encontrados.\n")
        else:
            texto_log.insert("end", "⚠️ Nenhum arquivo .esf encontrado nesta pasta.\n")

def testar_filtros_preview():
    amostra_escolhida = menu_amostras.get()
    texto_log.insert("end", f"Gerando preview para: {amostra_escolhida}\n")
    
    # Aqui entraria o seu código real para ler o arquivo e aplicar os filtros marcados
    # ...
    
    # Simulação da Janela Flutuante do Matplotlib
    plt.figure(figsize=(10, 5))
    plt.title(f"Preview: {amostra_escolhida} (Bruto vs Tratado)")
    plt.xlabel("Comprimento de Onda (nm)")
    plt.ylabel("Intensidade")
    plt.text(0.5, 0.5, "Gráfico do Espectro Aparece Aqui!", ha='center', va='center', fontsize=14)
    
    # Isso abre a janela nova com as ferramentas de zoom!
    plt.show() 

# ==============================================================================
# INTERFACE GRÁFICA (FRONT-END)
# ==============================================================================
# ... (Botão de selecionar pasta continua igual ao anterior) ...
botao_pasta = ctk.CTkButton(janela, text="📁 Selecionar Pasta dos .esf", command=escolher_pasta)
botao_pasta.pack(pady=10)

label_caminho = ctk.CTkLabel(janela, text="Nenhuma pasta selecionada.", text_color="gray")
label_caminho.pack(pady=5)

# --- NOVA SEÇÃO: PREVIEW ---
frame_preview = ctk.CTkFrame(janela)
frame_preview.pack(pady=10, padx=20, fill="x")

titulo_preview = ctk.CTkLabel(frame_preview, text="1. Selecione uma amostra para testar os filtros:")
titulo_preview.pack(pady=5)

menu_amostras = ctk.CTkOptionMenu(frame_preview, values=["Selecione a pasta primeiro..."], state="disabled")
menu_amostras.pack(pady=5)

botao_preview = ctk.CTkButton(frame_preview, text="👁️ Mostrar Gráfico de Preview", command=testar_filtros_preview, state="disabled")
botao_preview.pack(pady=10)

# --- SEÇÃO DE FILTROS (Igual ao anterior) ---
# ... (Checkboxes de Savitzky-Golay, Baseline e Normalização entram aqui) ...

texto_log = ctk.CTkTextbox(janela, height=100)
texto_log.pack(padx=20, fill="x", pady=10)

janela.mainloop()
