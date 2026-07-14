# Arquivo: main.py
import customtkinter as ctk
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import pandas as pd

# Importando nossos motores matemáticos
from modulos.pre_processador import filtrar_ruido, remover_baseline, normalizar_por_area, carregar_arquivo_esf
from modulos.motor_pca import executar_pca_arquivos

from modulos.motor_pca import executar_pca_arquivos
from modulos.motor_outliers import executar_caca_outliers # <--- ADICIONE ESTA LINHA

# Configuração Visual
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

janela = ctk.CTk()
janela.title("AgroLIBS Studio - Embrapa")
janela.geometry("700x750") 

# ==============================================================================
# SISTEMA DE ABAS (TABS)
# ==============================================================================
tabview = ctk.CTkTabview(janela, width=650)
tabview.pack(pady=10, padx=20, fill="both", expand=True)

aba_pre = tabview.add("1. Pré-processamento")
aba_pca = tabview.add("2. Análise PCA")
aba_outliers = tabview.add("3. Caçador de Outliers") # <--- ADICIONE ESTA LINHA

# ==============================================================================
# LÓGICA E INTERFACE DA ABA 1: PRÉ-PROCESSAMENTO
# ==============================================================================
pasta_selecionada = ""
arquivos_esf_encontrados = []

def escolher_pasta():
    global pasta_selecionada, arquivos_esf_encontrados
    pasta = filedialog.askdirectory(title="Selecione a pasta com os .esf")
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

def processar_espectro(intensidade, wl):
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
    if sufixo == "": sufixo = "_Bruto"
    return intens_tratada, sufixo

def testar_filtros_preview():
    amostra = menu_amostras.get()
    try:
        wl, intens_bruta = carregar_arquivo_esf(os.path.join(pasta_selecionada, amostra))
        intens_tratada, sufixo = processar_espectro(intens_bruta, wl)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(wl, intens_bruta, color='#7f8c8d')
        ax1.set_title(f"Espectro Bruto: {amostra}")
        ax2.plot(wl, intens_tratada, color='#2c3e50')
        ax2.set_title(f"Espectro Tratado (Tags: {sufixo})")
        plt.show() 
    except Exception as e:
        texto_log.insert("end", f"Erro no preview: {str(e)}\n")

def executar_processamento_lote():
    pasta_saida = os.path.join(pasta_selecionada, "LIBS_PROCESSADOS_CSV")
    os.makedirs(pasta_saida, exist_ok=True)
    sucessos = 0
    for arquivo in arquivos_esf_encontrados:
        try:
            wl, intens_bruta = carregar_arquivo_esf(os.path.join(pasta_selecionada, arquivo))
            intens_final, sufixo = processar_espectro(intens_bruta, wl)
            nome_final = f"{os.path.splitext(arquivo)[0]}{sufixo}.csv"
            pd.DataFrame({'comprimento_onda_nm': wl, 'intensidade_processada': intens_final}).to_csv(os.path.join(pasta_saida, nome_final), index=False)
            sucessos += 1
        except Exception: continue
    texto_log.insert("end", f"✅ {sucessos} arquivos salvos em LIBS_PROCESSADOS_CSV\n")

# Layout da Aba 1
botao_pasta = ctk.CTkButton(aba_pre, text="📁 1. Selecionar Pasta dos .esf", command=escolher_pasta)
botao_pasta.pack(pady=10)
label_caminho = ctk.CTkLabel(aba_pre, text="Nenhuma pasta selecionada.", text_color="gray")
label_caminho.pack()

menu_amostras = ctk.CTkOptionMenu(aba_pre, values=["Aguardando..."], state="disabled")
menu_amostras.pack(pady=5)
botao_preview = ctk.CTkButton(aba_pre, text="👁️ 2. Preview de Filtros", command=testar_filtros_preview, state="disabled")
botao_preview.pack(pady=5)

check_savgol = ctk.CTkCheckBox(aba_pre, text="Filtro Savitzky-Golay")
check_savgol.pack(pady=5)
check_savgol.select() 
check_baseline = ctk.CTkCheckBox(aba_pre, text="Correção Baseline")
check_baseline.pack(pady=5)
check_baseline.select()
check_normalizacao = ctk.CTkCheckBox(aba_pre, text="Normalização por Área")
check_normalizacao.pack(pady=5)
check_normalizacao.select()

botao_processar = ctk.CTkButton(aba_pre, text="▶️ 3. Salvar Lote (.csv)", command=executar_processamento_lote, state="disabled", fg_color="green")
botao_processar.pack(pady=10)
texto_log = ctk.CTkTextbox(aba_pre, height=80)
texto_log.pack(fill="x", padx=10, pady=5)


# ==============================================================================
# LÓGICA E INTERFACE DA ABA 2: ANÁLISE PCA
# ==============================================================================
arquivos_pca_selecionados = []

def escolher_pasta_pca():
    global arquivos_pca_selecionados
    
    pasta = filedialog.askdirectory(title="Selecione a pasta com os resultados .csv")
    
    if pasta:
        arquivos_encontrados = [
            os.path.join(pasta, f) for f in os.listdir(pasta) if f.endswith('.csv')
        ]
        
        if len(arquivos_encontrados) >= 3:
            arquivos_pca_selecionados = arquivos_encontrados
            lista_pca_texto.delete("1.0", "end")
            
            for f in arquivos_pca_selecionados:
                lista_pca_texto.insert("end", f"{os.path.basename(f)}\n")
                
            botao_gerar_pca.configure(state="normal")
            label_status_pca.configure(text=f"✅ {len(arquivos_pca_selecionados)} arquivos carregados da pasta.", text_color="green")
        else:
            lista_pca_texto.delete("1.0", "end")
            lista_pca_texto.insert("end", "⚠️ Erro: A pasta selecionada não possui arquivos .csv suficientes (Mínimo: 3).\n")
            botao_gerar_pca.configure(state="disabled")
            label_status_pca.configure(text="Arquivos insuficientes.", text_color="red")

def acionar_motor_pca():
    try:
        executar_pca_arquivos(arquivos_pca_selecionados)
    except Exception as e:
        label_status_pca.configure(text=f"Erro: {str(e)}", text_color="red")

# Layout da Aba 2
titulo_pca = ctk.CTkLabel(aba_pca, text="Análise de Componentes Principais", font=ctk.CTkFont(size=18, weight="bold"))
titulo_pca.pack(pady=10)

botao_selecionar_pca = ctk.CTkButton(aba_pca, text="📁 Selecionar Pasta com .csv", command=escolher_pasta_pca)
botao_selecionar_pca.pack(pady=10)

instrucao_pca = ctk.CTkLabel(aba_pca, text="Arquivos encontrados na pasta:")
instrucao_pca.pack()

lista_pca_texto = ctk.CTkTextbox(aba_pca, height=200)
lista_pca_texto.pack(fill="x", padx=20, pady=5)

botao_gerar_pca = ctk.CTkButton(aba_pca, text="🧠 Gerar Gráfico de Escores", command=acionar_motor_pca, state="disabled", fg_color="#8e44ad", hover_color="#9b59b6")
botao_gerar_pca.pack(pady=20)

label_status_pca = ctk.CTkLabel(aba_pca, text="", text_color="gray")
label_status_pca.pack()


# ==============================================================================
# LÓGICA E INTERFACE DA ABA 3: CAÇADOR DE OUTLIERS
# ==============================================================================
arquivos_outliers_selecionados = []

def escolher_pasta_outliers():
    global arquivos_outliers_selecionados
    pasta = filedialog.askdirectory(title="Selecione a pasta com os resultados .csv")
    if pasta:
        arquivos = [os.path.join(pasta, f) for f in os.listdir(pasta) if f.endswith('.csv')]
        if len(arquivos) >= 3:
            arquivos_outliers_selecionados = arquivos
            label_status_out.configure(text=f"✅ {len(arquivos)} arquivos carregados.", text_color="green")
            botao_gerar_out.configure(state="normal")
        else:
            label_status_out.configure(text="⚠️ Arquivos insuficientes (Mínimo 3).", text_color="red")

def acionar_motor_outliers():
    try:
        texto_log_out.delete("1.0", "end")
        texto_log_out.insert("end", "Calculando matriz de distâncias Z-Score...\n")
        
        # Chama o motor matemático
        qtd, lista_ruins = executar_caca_outliers(arquivos_outliers_selecionados)
        
        texto_log_out.insert("end", f"\nFim da varredura! {qtd} outliers encontrados.\n")
        if qtd > 0:
            texto_log_out.insert("end", "Amostras defeituosas:\n")
            for ruim in lista_ruins:
                texto_log_out.insert("end", f" -> {ruim}\n")
    except Exception as e:
        label_status_out.configure(text=f"Erro: {str(e)}", text_color="red")

titulo_out = ctk.CTkLabel(aba_outliers, text="Rastreamento de Anomalias (Z-Score)", font=ctk.CTkFont(size=18, weight="bold"))
titulo_out.pack(pady=10)

botao_selecionar_out = ctk.CTkButton(aba_outliers, text="📁 Selecionar Pasta com .csv", command=escolher_pasta_outliers)
botao_selecionar_out.pack(pady=10)

label_status_out = ctk.CTkLabel(aba_outliers, text="Nenhuma pasta selecionada.", text_color="gray")
label_status_out.pack()

botao_gerar_out = ctk.CTkButton(aba_outliers, text="🎯 Iniciar Caçada a Outliers", command=acionar_motor_outliers, state="disabled", fg_color="#c0392b", hover_color="#e74c3c")
botao_gerar_out.pack(pady=20)

texto_log_out = ctk.CTkTextbox(aba_outliers, height=150)
texto_log_out.pack(fill="x", padx=20, pady=5)



janela.mainloop()
