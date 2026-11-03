# Arquivo: main.py
"""
AgroLIBS Studio - interface gráfica.

Organização:
- Os cálculos ficam em modulos/motor_*.py e a leitura dos .csv em modulos/dados.py.
- Os gráficos ficam em modulos/graficos.py e aparecem DENTRO da janela
  (com barra de zoom / salvar), em vez de abrir janelas soltas do matplotlib.
- Tudo o que demora roda em segundo plano, para a janela não congelar.
  A thread de trabalho nunca mexe na tela: ela manda pedidos para a fila
  `no_principal`, e a janela executa esses pedidos.
"""
import os
import queue
import threading
from tkinter import filedialog, messagebox

import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import numpy as np  # noqa: E402

from modulos import dados, graficos, motor_classificacao  # noqa: E402
from modulos.conversor_arquivos import converter_pasta_esf_para_csv, listar_esf, tratar_espectro  # noqa: E402
from modulos.pre_processador import carregar_arquivo_esf, identificar_picos_npk  # noqa: E402
from modulos.motor_pca import calcular_pca  # noqa: E402
from modulos.motor_outliers import detectar_outliers  # noqa: E402
from modulos.motor_variancia import calcular_variancia  # noqa: E402

VERSAO = "2.1"

# Cores do app (as mesmas dos botões antigos, para manter a identidade)
VERDE = ("#28a745", "#218838")
ROXO = ("#8e44ad", "#71368a")
AZUL = ("#2980b9", "#3498db")
VERMELHO = ("#c0392b", "#e74c3c")
LARANJA = ("#d35400", "#e67e22")
COR_OK, COR_ERRO, COR_AVISO, COR_NEUTRA = "#2ecc71", "#e74c3c", "#f39c12", "gray"

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
graficos.aplicar_estilo()

janela = ctk.CTk()
janela.title(f"AgroLIBS Studio {VERSAO} - Embrapa")
janela.geometry("1320x860")
janela.minsize(1100, 700)


# ==============================================================================
# FERRAMENTAS GERAIS
# ==============================================================================
_fila_tela = queue.Queue()


def no_principal(funcao, *args):
    """Pede para a janela executar `funcao` (seguro de chamar de qualquer thread)."""
    _fila_tela.put((funcao, args))


def _processar_fila():
    try:
        while True:
            funcao, args = _fila_tela.get_nowait()
            try:
                funcao(*args)
            except Exception as e:  # um erro de tela não pode matar o app
                print("Erro na interface:", e)
    except queue.Empty:
        pass
    janela.after(40, _processar_fila)


def em_segundo_plano(tarefa, ao_terminar, ao_falhar):
    """Roda tarefa() numa thread; depois chama ao_terminar(resultado) ou ao_falhar(erro) na janela."""
    def rodar():
        try:
            resultado = tarefa()
        except Exception as e:
            no_principal(ao_falhar, e)
        else:
            no_principal(ao_terminar, resultado)
    threading.Thread(target=rodar, daemon=True).start()


def ler_numero(entrada, padrao, minimo, maximo, tipo=int):
    """Lê um número de uma caixa de texto; se estiver errado, volta ao padrão e corrige a caixa."""
    try:
        valor = tipo(entrada.get().replace(",", "."))
        valor = min(max(valor, minimo), maximo)
    except ValueError:
        valor = padrao
    entrada.delete(0, "end")
    entrada.insert(0, str(valor))
    return valor


def status(mensagem, cor=COR_NEUTRA):
    label_status_geral.configure(text=mensagem, text_color=cor)


class Terminal(ctk.CTkTextbox):
    """Caixa de log no estilo terminal. escrever() funciona de qualquer thread."""

    def __init__(self, master, **kw):
        super().__init__(master, fg_color="#0d0d0d", text_color="#00FF00",
                         font=("Consolas", 12), wrap="word", **kw)

    def escrever(self, msg):
        def _escreve():
            self.insert("end", msg + "\n")
            self.see("end")
        no_principal(_escreve)

    def limpar(self):
        self.delete("1.0", "end")


class PainelGrafico(ctk.CTkFrame):
    """Área de gráfico dentro da aba, com a barra do matplotlib (zoom, mover, salvar)."""

    def __init__(self, master, texto_inicial):
        super().__init__(master, fg_color="white", corner_radius=8)
        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        barra = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        barra.update()
        barra.pack(side="bottom", fill="x", padx=4, pady=(0, 4))
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self.aviso(texto_inicial)

    def desenhar(self, funcao, *args, **kw):
        try:
            funcao(self.fig, *args, **kw)
        except Exception as e:
            graficos.mensagem(self.fig, f"Não consegui desenhar o gráfico:\n{e}", cor=COR_ERRO)
        self.canvas.draw_idle()

    def aviso(self, texto, cor="#7f8c8d"):
        graficos.mensagem(self.fig, texto, cor=cor)
        self.canvas.draw_idle()


def cartao(master, titulo):
    """Um bloco com título, para agrupar os controles."""
    frame = ctk.CTkFrame(master, corner_radius=10)
    frame.pack(fill="x", padx=6, pady=6)
    ctk.CTkLabel(frame, text=titulo, font=ctk.CTkFont(size=13, weight="bold"),
                 anchor="w").pack(fill="x", padx=12, pady=(8, 4))
    return frame


def linha(master):
    frame = ctk.CTkFrame(master, fg_color="transparent")
    frame.pack(fill="x", padx=12, pady=3)
    return frame


def montar_aba(aba, texto_grafico, espaco_acima_do_grafico=False):
    """
    Coluna da esquerda (controles, com rolagem) e gráfico à direita.
    espaco_acima_do_grafico=True deixa a linha 0 da direita livre (para um seletor).
    """
    linha_grafico = 1 if espaco_acima_do_grafico else 0
    aba.grid_columnconfigure(1, weight=1)
    aba.grid_rowconfigure(linha_grafico, weight=1)
    # Cor explícita (e não "transparent"): evita cantos escuros ao trocar para o tema claro
    esquerda = ctk.CTkScrollableFrame(aba, width=340, fg_color=tabview.cget("fg_color"))
    esquerda.grid(row=0, column=0, rowspan=2, sticky="nsw", padx=(0, 8))
    painel = PainelGrafico(aba, texto_grafico)
    painel.grid(row=linha_grafico, column=1, sticky="nsew")
    return esquerda, painel


# ==============================================================================
# CABEÇALHO E ABAS
# ==============================================================================
cabecalho = ctk.CTkFrame(janela, fg_color="transparent")
cabecalho.pack(fill="x", padx=20, pady=(12, 0))
ctk.CTkLabel(cabecalho, text="🌱 AgroLIBS Studio", font=ctk.CTkFont(size=24, weight="bold")).pack(side="left")
ctk.CTkLabel(cabecalho, text=f"  v{VERSAO} · LIBS + quimiometria para fertilizantes NPK · Embrapa Instrumentação",
             text_color="gray", font=ctk.CTkFont(size=13)).pack(side="left", pady=(6, 0))


def trocar_tema(valor):
    ctk.set_appearance_mode("Light" if valor == "☀️ Claro" else "Dark")


seletor_tema = ctk.CTkSegmentedButton(cabecalho, values=["🌙 Escuro", "☀️ Claro"], command=trocar_tema)
seletor_tema.set("🌙 Escuro")
seletor_tema.pack(side="right")

barra_status = ctk.CTkFrame(janela, height=28, corner_radius=0)
barra_status.pack(side="bottom", fill="x")
label_status_geral = ctk.CTkLabel(barra_status, text="Pronto. Comece pela aba 1 (arquivos .esf) ou escolha uma pasta de .csv nas abas 2 a 5.",
                                  text_color=COR_NEUTRA, anchor="w")
label_status_geral.pack(fill="x", padx=14)

tabview = ctk.CTkTabview(janela)
tabview.pack(pady=(4, 8), padx=20, fill="both", expand=True)
aba_pre_conv = tabview.add("1 · Pré-processamento")
aba_pca = tabview.add("2 · PCA")
aba_outliers = tabview.add("3 · Outliers")
aba_variancia = tabview.add("4 · Variância")
aba_classificacao = tabview.add("5 · Classificação (KNN/RF)")


# ==============================================================================
# PASTA DE .CSV COMPARTILHADA PELAS ABAS 2 A 5
# ==============================================================================
class CartaoDados(ctk.CTkFrame):
    """
    Bloco 'Dados' das abas 2 a 5: pasta de .csv + região UV/VIS.
    A pasta é a mesma em todas as abas: escolher numa muda nas outras.
    """
    todos = []

    def __init__(self, master, ao_mudar, minimo=3):
        super().__init__(master, corner_radius=10)
        self.pack(fill="x", padx=6, pady=6)
        self.ao_mudar = ao_mudar
        self.minimo = minimo
        self.pasta = ""
        self.arquivos = []

        ctk.CTkLabel(self, text="📂 Dados", font=ctk.CTkFont(size=13, weight="bold"),
                     anchor="w").pack(fill="x", padx=12, pady=(8, 4))
        ctk.CTkButton(self, text="Escolher pasta com os .csv tratados", command=self.escolher).pack(fill="x", padx=12, pady=3)
        self.label_pasta = ctk.CTkLabel(self, text="Nenhuma pasta selecionada.", text_color=COR_NEUTRA,
                                        wraplength=300, justify="left", anchor="w")
        self.label_pasta.pack(fill="x", padx=12)
        l = linha(self)
        ctk.CTkLabel(l, text="Região:").pack(side="left")
        self.seletor_regiao = ctk.CTkSegmentedButton(l, values=["UV", "VIS"], command=self.atualizar)
        self.seletor_regiao.set("VIS")
        self.seletor_regiao.pack(side="left", padx=8)
        self.label_contagem = ctk.CTkLabel(self, text="", anchor="w", wraplength=300, justify="left")
        self.label_contagem.pack(fill="x", padx=12, pady=(0, 8))
        CartaoDados.todos.append(self)

    def escolher(self):
        pasta = filedialog.askdirectory(title="Selecione a pasta RAIZ com os .csv tratados")
        if pasta:
            definir_pasta_csv(pasta)

    def regiao(self):
        return self.seletor_regiao.get()

    def definir(self, pasta, contagem):
        self.pasta = pasta
        self.label_pasta.configure(text=pasta, text_color=("gray20", "gray80"))
        if contagem[self.regiao()] == 0 and sum(contagem.values()) > 0:
            self.seletor_regiao.set("UV" if contagem["UV"] else "VIS")
        self.atualizar()

    def atualizar(self, *_):
        self.arquivos = dados.listar_csvs(self.pasta, self.regiao()) if self.pasta else []
        if not self.pasta:
            return
        n = len(self.arquivos)
        q = dados.contar_quarentena(self.pasta)
        extra = f" ({q} na quarentena)" if q else ""
        if n >= self.minimo:
            self.label_contagem.configure(text=f"✅ {n} espectros {self.regiao()}{extra}", text_color=COR_OK)
        elif n == 0:
            self.label_contagem.configure(text=f"⚠️ Nenhum .csv numa pasta {self.regiao()}{extra}", text_color=COR_ERRO)
        else:
            self.label_contagem.configure(text=f"⚠️ Só {n} espectros (mínimo {self.minimo}){extra}", text_color=COR_ERRO)
        self.ao_mudar(self)

    def pronto(self):
        return len(self.arquivos) >= self.minimo


def definir_pasta_csv(pasta):
    contagem = dados.contar_por_regiao(pasta)
    for c in CartaoDados.todos:
        c.definir(pasta, contagem)
    status(f"Pasta de .csv: {pasta}  (UV: {contagem['UV']} · VIS: {contagem['VIS']} espectros)", COR_OK)


def opcoes_de_cor(menu, cartao_dados, memoria):
    """Preenche um menu 'Cores por' com os níveis de pasta da seleção atual."""
    opcoes = dados.opcoes_de_agrupamento(cartao_dados.arquivos, cartao_dados.pasta) if cartao_dados.arquivos else []
    memoria.clear()
    memoria.update(opcoes)
    textos = [t for t, _ in opcoes] or ["Aguardando..."]
    atual = menu.get()
    menu.configure(values=textos, state="normal" if opcoes else "disabled")
    menu.set(atual if atual in textos else textos[0])


# ==============================================================================
# ABA 1: PRÉ-PROCESSAMENTO & CONVERSÃO
# ==============================================================================
estado_pre = {"pasta": "", "arquivos": [], "filtrados": [], "indice": 0, "cache": None}

col_pre, painel_pre = montar_aba(aba_pre_conv, "Escolha a pasta com os .esf brutos\npara ver o espectro antes e depois dos filtros.")


def nome_curto(caminho, raiz):
    return os.path.relpath(caminho, raiz).replace("\\", "/")


def escolher_pasta_esf():
    pasta = filedialog.askdirectory(title="Selecione a pasta RAIZ com os .esf brutos")
    if not pasta:
        return
    arquivos = listar_esf(pasta)
    estado_pre.update(pasta=pasta, arquivos=arquivos, cache=None)
    if arquivos:
        n_pastas = len({os.path.dirname(a) for a in arquivos})
        label_status_conv.configure(text=f"✅ {len(arquivos)} arquivos .esf em {n_pastas} subpastas.", text_color=COR_OK)
        botao_processar_tudo.configure(state="normal")
        aplicar_filtro_nome()
        status(f"Pasta .esf: {pasta}", COR_OK)
    else:
        label_status_conv.configure(text="⚠️ Nenhum .esf encontrado nas subpastas.", text_color=COR_ERRO)
        botao_processar_tudo.configure(state="disabled")
        for w in (botao_anterior, botao_proximo, menu_amostras):
            w.configure(state="disabled")
        painel_pre.aviso("Nenhum arquivo .esf nessa pasta.", cor=COR_ERRO)


def aplicar_filtro_nome(*_):
    """Filtra a lista de arquivos pelo texto digitado (ex.: '25pct', 'amostra_c', 'UV')."""
    termo = entrada_filtro.get().strip().lower()
    raiz = estado_pre["pasta"]
    estado_pre["filtrados"] = [a for a in estado_pre["arquivos"] if termo in nome_curto(a, raiz).lower()]
    estado_pre["indice"] = 0
    ha_arquivos = bool(estado_pre["filtrados"])
    for w in (botao_anterior, botao_proximo, menu_amostras):
        w.configure(state="normal" if ha_arquivos else "disabled")
    if ha_arquivos:
        # O menu mostra até 50 nomes (um menu gigante trava o Tk); as setas percorrem todos
        menu_amostras.configure(values=[os.path.basename(a) for a in estado_pre["filtrados"][:50]])
        mostrar_preview()
    else:
        menu_amostras.configure(values=["Nenhum arquivo com esse nome"])
        menu_amostras.set("Nenhum arquivo com esse nome")
        label_posicao.configure(text="0 / 0")


def andar(passo):
    if estado_pre["filtrados"]:
        estado_pre["indice"] = (estado_pre["indice"] + passo) % len(estado_pre["filtrados"])
        mostrar_preview()


def escolher_no_menu(nome):
    for i, a in enumerate(estado_pre["filtrados"]):
        if os.path.basename(a) == nome:
            estado_pre["indice"] = i
            mostrar_preview()
            return


def mostrar_preview(*_):
    if not estado_pre["filtrados"]:
        return
    caminho = estado_pre["filtrados"][estado_pre["indice"]]
    nome = os.path.basename(caminho)
    menu_amostras.set(nome)
    label_posicao.configure(text=f"{estado_pre['indice'] + 1} / {len(estado_pre['filtrados'])}")
    try:
        # Guarda o último arquivo lido: marcar/desmarcar um filtro não relê o disco
        if not estado_pre["cache"] or estado_pre["cache"][0] != caminho:
            estado_pre["cache"] = (caminho, *carregar_arquivo_esf(caminho))
        _, wl, bruto = estado_pre["cache"]
        usa = (check_savgol.get(), check_baseline.get(), check_normalizacao.get())
        tratado = tratar_espectro(wl, bruto, *map(bool, usa))
        tags = " + ".join(t for t, u in zip(("SG", "BL", "Norm"), usa) if u) or "nenhum filtro"
        picos = identificar_picos_npk(wl, tratado)
        painel_pre.desenhar(graficos.espectro_antes_depois, wl, bruto, tratado, nome, tags, picos)
        resumo = ", ".join(f"{el}: {sum(p['elemento'] == el for p in picos)}" for el in "NPK")
        label_picos.configure(text=f"Picos encontrados → {resumo}  ·  {len(wl)} pontos, {wl[0]:.1f}-{wl[-1]:.1f} nm")
    except Exception as e:
        painel_pre.aviso(f"Erro ao ler {nome}:\n{e}", cor=COR_ERRO)


def iniciar_processamento_em_lote():
    pasta_origem = estado_pre["pasta"]
    if not pasta_origem:
        return
    pasta_pai = os.path.dirname(pasta_origem)
    nome_pasta_origem = os.path.basename(pasta_origem)
    pasta_destino = os.path.join(pasta_pai, f"{nome_pasta_origem}_CSV_Tratados")

    if os.path.isdir(pasta_destino) and any(os.scandir(pasta_destino)):
        if not messagebox.askyesno("Pasta já existe",
                                   f"A pasta\n{pasta_destino}\njá existe. Os .csv com o mesmo nome serão substituídos.\n\nContinuar?"):
            return

    usa_sg, usa_bl, usa_norm = bool(check_savgol.get()), bool(check_baseline.get()), bool(check_normalizacao.get())
    terminal_conversao.limpar()
    barra_progresso.set(0)
    label_progresso.configure(text="Começando...")
    label_status_conv.configure(text="A processar em segundo plano...", text_color=COR_AVISO)
    botao_processar_tudo.configure(state="disabled")
    botao_pasta_esf.configure(state="disabled")

    def progresso(feitos, total):
        no_principal(barra_progresso.set, feitos / total)
        no_principal(lambda: label_progresso.configure(text=f"{feitos} de {total} arquivos ({feitos / total:.0%})"))

    def tarefa():
        return converter_pasta_esf_para_csv(pasta_origem, pasta_destino, usar_sg=usa_sg, usar_baseline=usa_bl,
                                            usar_norm=usa_norm, callback=terminal_conversao.escrever,
                                            progresso=progresso)

    def terminou(resultado):
        sucesso, falhas, _ = resultado
        botao_processar_tudo.configure(state="normal")
        botao_pasta_esf.configure(state="normal")
        cor = COR_OK if falhas == 0 else COR_ERRO
        label_status_conv.configure(text=f"✅ {sucesso} salvos em {nome_pasta_origem}_CSV_Tratados"
                                         + (f" ({falhas} falhas)" if falhas else ""), text_color=cor)
        if sucesso:
            dados.limpar_cache()
            definir_pasta_csv(pasta_destino)
            status(f"Lote pronto: {sucesso} .csv em {pasta_destino}. As abas 2 a 5 já usam essa pasta.", cor)

    def falhou(erro):
        terminal_conversao.escrever(f"\n❌ ERRO FATAL: {erro}")
        label_status_conv.configure(text="Erro no processamento.", text_color=COR_ERRO)
        botao_processar_tudo.configure(state="normal")
        botao_pasta_esf.configure(state="normal")

    em_segundo_plano(tarefa, terminou, falhou)


# --- Layout da Aba 1 ---
c = cartao(col_pre, "1. Arquivos .esf brutos")
botao_pasta_esf = ctk.CTkButton(c, text="📁 Selecionar pasta raiz (.esf)", command=escolher_pasta_esf)
botao_pasta_esf.pack(fill="x", padx=12, pady=3)
label_status_conv = ctk.CTkLabel(c, text="Nenhuma pasta selecionada.", text_color=COR_NEUTRA, anchor="w")
label_status_conv.pack(fill="x", padx=12)
entrada_filtro = ctk.CTkEntry(c, placeholder_text="🔍 Filtrar por nome (ex.: 25pct, amostra_c, UV)")
entrada_filtro.pack(fill="x", padx=12, pady=3)
entrada_filtro.bind("<KeyRelease>", aplicar_filtro_nome)
menu_amostras = ctk.CTkOptionMenu(c, values=["Aguardando..."], state="disabled", dynamic_resizing=False,
                                  command=escolher_no_menu)
menu_amostras.pack(fill="x", padx=12, pady=3)
l = linha(c)
botao_anterior = ctk.CTkButton(l, text="◀", width=44, command=lambda: andar(-1), state="disabled")
botao_anterior.pack(side="left")
label_posicao = ctk.CTkLabel(l, text="0 / 0")
label_posicao.pack(side="left", expand=True)
botao_proximo = ctk.CTkButton(l, text="▶", width=44, command=lambda: andar(1), state="disabled")
botao_proximo.pack(side="right")
ctk.CTkLabel(c, text="").pack(pady=1)

c = cartao(col_pre, "2. Filtros (o gráfico atualiza na hora)")
check_savgol = ctk.CTkCheckBox(c, text="Filtro Savitzky-Golay (janela 11, grau 2)", command=mostrar_preview)
check_baseline = ctk.CTkCheckBox(c, text="Correção de baseline (polinômio grau 2)", command=mostrar_preview)
check_normalizacao = ctk.CTkCheckBox(c, text="Normalização por área", command=mostrar_preview)
for chk in (check_savgol, check_baseline, check_normalizacao):
    chk.select()
    chk.pack(anchor="w", padx=12, pady=4)
label_picos = ctk.CTkLabel(c, text="", text_color=COR_NEUTRA, wraplength=300, justify="left", anchor="w")
label_picos.pack(fill="x", padx=12, pady=(0, 8))

c = cartao(col_pre, "3. Converter o lote inteiro para .csv")
botao_processar_tudo = ctk.CTkButton(c, text="▶ Processar lote inteiro", command=iniciar_processamento_em_lote,
                                     state="disabled", fg_color=VERDE[0], hover_color=VERDE[1], height=38)
botao_processar_tudo.pack(fill="x", padx=12, pady=3)
barra_progresso = ctk.CTkProgressBar(c)
barra_progresso.set(0)
barra_progresso.pack(fill="x", padx=12, pady=(6, 2))
label_progresso = ctk.CTkLabel(c, text="Os .csv vão para <pasta>_CSV_Tratados, com as mesmas subpastas.",
                               text_color=COR_NEUTRA, wraplength=300, justify="left", anchor="w")
label_progresso.pack(fill="x", padx=12, pady=(0, 8))

ctk.CTkLabel(col_pre, text="Terminal de processamento:", font=ctk.CTkFont(size=12, slant="italic"),
             text_color=COR_NEUTRA, anchor="w").pack(fill="x", padx=8)
terminal_conversao = Terminal(col_pre, height=180)
terminal_conversao.pack(fill="x", padx=6, pady=(0, 6))


# ==============================================================================
# ABA 2: ANÁLISE PCA
# ==============================================================================
estado_pca = {"resultado": None, "grupos": None, "eixo": None, "regiao": ""}
opcoes_cor_pca = {}

col_pca, painel_pca = montar_aba(aba_pca, "Escolha a pasta de .csv e clique em 'Gerar PCA'.")


def ao_mudar_dados_pca(cartao_dados):
    opcoes_de_cor(menu_cor_pca, cartao_dados, opcoes_cor_pca)
    botao_gerar_pca.configure(state="normal" if cartao_dados.pronto() else "disabled")


def acionar_motor_pca():
    arquivos, pasta, regiao = dados_pca.arquivos, dados_pca.pasta, dados_pca.regiao()
    niveis = opcoes_cor_pca.get(menu_cor_pca.get(), ())
    padronizar = bool(check_padronizar_pca.get())
    botao_gerar_pca.configure(state="disabled")
    label_resultado_pca.configure(text="Calculando...", text_color=COR_AVISO)
    painel_pca.aviso(f"Lendo {len(arquivos)} espectros e calculando o PCA...")

    def tarefa():
        X, eixo, usados = dados.carregar_espectros(arquivos)
        res = calcular_pca(X, n_componentes=3, padronizar=padronizar)
        grupos = [dados.grupo_do_arquivo(a, pasta, niveis) for a in usados]
        return res, grupos, eixo

    def terminou(r):
        res, grupos, eixo = r
        estado_pca.update(resultado=res, grupos=grupos, eixo=eixo, regiao=regiao)
        botao_gerar_pca.configure(state="normal")
        v = res["var_exp"]
        label_resultado_pca.configure(
            text=f"PC1 {v[0]:.1f}% · PC2 {v[1]:.1f}%" + (f" · PC3 {v[2]:.1f}%" if len(v) > 2 else "")
                 + f"\nPC1+PC2 = {v[0] + v[1]:.1f}%  ·  {len(grupos)} espectros, {len(set(grupos))} grupos",
            text_color=COR_OK)
        redesenhar_pca()
        status(f"PCA {regiao} pronto.", COR_OK)

    def falhou(erro):
        botao_gerar_pca.configure(state="normal")
        label_resultado_pca.configure(text=f"Erro: {erro}", text_color=COR_ERRO)
        painel_pca.aviso(f"Erro no PCA:\n{erro}", cor=COR_ERRO)

    em_segundo_plano(tarefa, terminou, falhou)


def redesenhar_pca(*_):
    """Troca entre 2D, 3D e loadings sem recalcular nada."""
    res = estado_pca["resultado"]
    if res is None:
        return
    modo = seletor_modo_pca.get()
    extra = "" if res["padronizado"] else " (só centrado)"
    if modo == "Loadings":
        painel_pca.desenhar(graficos.pca_loadings, estado_pca["eixo"], res["loadings"], res["var_exp"],
                            estado_pca["regiao"], titulo_extra=extra)
    else:
        painel_pca.desenhar(graficos.pca_escores, res["escores"], estado_pca["grupos"], res["var_exp"],
                            estado_pca["regiao"], em_3d=(modo == "3D"), elipses=bool(check_elipses.get()),
                            titulo_extra=extra)


dados_pca = CartaoDados(col_pca, ao_mudar_dados_pca)
c = cartao(col_pca, "⚙️ Opções")
ctk.CTkLabel(c, text="Cores por:", anchor="w").pack(fill="x", padx=12)
menu_cor_pca = ctk.CTkOptionMenu(c, values=["Aguardando..."], state="disabled", dynamic_resizing=False)
menu_cor_pca.pack(fill="x", padx=12, pady=3)
ctk.CTkLabel(c, text="Gráfico:", anchor="w").pack(fill="x", padx=12, pady=(6, 0))
seletor_modo_pca = ctk.CTkSegmentedButton(c, values=["2D", "3D", "Loadings"], command=redesenhar_pca)
seletor_modo_pca.set("2D")
seletor_modo_pca.pack(fill="x", padx=12, pady=3)
check_elipses = ctk.CTkCheckBox(c, text="Elipses de confiança 95% (2D)", command=redesenhar_pca)
check_elipses.select()
check_elipses.pack(anchor="w", padx=12, pady=4)
check_padronizar_pca = ctk.CTkCheckBox(c, text="Padronizar (StandardScaler)")
check_padronizar_pca.select()
check_padronizar_pca.pack(anchor="w", padx=12, pady=4)
ctk.CTkLabel(c, text="Desmarcado = só centrar na média: os picos pesam\nmais que o ruído e os loadings ficam mais legíveis.",
             text_color=COR_NEUTRA, font=ctk.CTkFont(size=11), justify="left", anchor="w").pack(fill="x", padx=12, pady=(0, 8))
botao_gerar_pca = ctk.CTkButton(col_pca, text="🧠 Gerar PCA", command=acionar_motor_pca, state="disabled",
                                fg_color=ROXO[0], hover_color=ROXO[1], height=38)
botao_gerar_pca.pack(fill="x", padx=6, pady=6)
label_resultado_pca = ctk.CTkLabel(col_pca, text="", justify="left", anchor="w", wraplength=320)
label_resultado_pca.pack(fill="x", padx=10)


# ==============================================================================
# ABA 3: CAÇADOR DE OUTLIERS (com quarentena reversível)
# ==============================================================================
estado_out = {"suspeitos": [], "pasta": ""}
opcoes_cor_out = {}

col_out, painel_out = montar_aba(aba_outliers, "Escolha a pasta de .csv e clique em 'Rastrear outliers'.\n"
                                               "Os suspeitos aparecem circulados em vermelho.")


def ao_mudar_dados_out(cartao_dados):
    opcoes_de_cor(menu_cor_out, cartao_dados, opcoes_cor_out)
    botao_gerar_out.configure(state="normal" if cartao_dados.pronto() else "disabled")
    botao_quarentena.configure(state="disabled")
    q = dados.contar_quarentena(cartao_dados.pasta) if cartao_dados.pasta else 0
    botao_restaurar.configure(text=f"↩ Restaurar quarentena ({q})", state="normal" if q else "disabled")


def acionar_motor_outliers():
    arquivos, pasta, regiao = dados_out.arquivos, dados_out.pasta, dados_out.regiao()
    contaminacao = ler_numero(entrada_contaminacao, 3.0, 0.5, 30.0, float) / 100
    niveis = opcoes_cor_out.get(menu_cor_out.get(), ())
    botao_gerar_out.configure(state="disabled")
    botao_quarentena.configure(state="disabled")
    texto_log_out.limpar()
    texto_log_out.escrever(f"Lendo {len(arquivos)} espectros {regiao}...")

    def tarefa():
        X, _, usados = dados.carregar_espectros(arquivos, callback=texto_log_out.escrever)
        texto_log_out.escrever("PCA (2 PCs) + Isolation Forest...")
        res = detectar_outliers(X, contaminacao=contaminacao)
        return res, usados

    def terminou(r):
        res, usados = r
        botao_gerar_out.configure(state="normal")
        suspeitos = [usados[i] for i in np.where(res["eh_outlier"])[0]]
        estado_out.update(suspeitos=suspeitos, pasta=pasta)
        nomes = [os.path.basename(a) for a in usados]
        grupos = [dados.grupo_do_arquivo(a, pasta, niveis) for a in usados]
        painel_out.desenhar(graficos.outliers, res["escores"], res["eh_outlier"], nomes, res["var_exp"], regiao,
                            grupos=grupos)
        texto_log_out.escrever(f"\nFim da varredura! {len(suspeitos)} outliers em {len(usados)} espectros "
                               f"(contaminação {contaminacao:.1%}).")
        if suspeitos:
            texto_log_out.escrever("Suspeitos (do mais estranho para o menos):")
            ordem = sorted(np.where(res["eh_outlier"])[0], key=lambda i: -res["nota"][i])
            for i in ordem:
                texto_log_out.escrever(f" -> {nome_curto(usados[i], pasta)}")
            texto_log_out.escrever("\nConfira no gráfico antes de mover. O Isolation Forest sempre marca a fração "
                                   "pedida, mesmo quando os dados estão bons.")
            botao_quarentena.configure(state="normal")
        status(f"Outliers {regiao}: {len(suspeitos)} suspeitos.", COR_AVISO if suspeitos else COR_OK)

    def falhou(erro):
        botao_gerar_out.configure(state="normal")
        texto_log_out.escrever(f"\n❌ ERRO: {erro}")

    em_segundo_plano(tarefa, terminou, falhou)


def enviar_para_quarentena():
    """Move (não apaga) os suspeitos para <pasta>/QUARENTENA. Dá para desfazer."""
    suspeitos, pasta = estado_out["suspeitos"], estado_out["pasta"]
    if not suspeitos:
        return
    lista = "\n".join(os.path.basename(a) for a in suspeitos[:15]) + ("\n..." if len(suspeitos) > 15 else "")
    if not messagebox.askyesno("Mover para a quarentena",
                               f"Mover {len(suspeitos)} arquivos para a pasta QUARENTENA?\n\n{lista}\n\n"
                               "Eles saem de todas as análises, mas não são apagados.\n"
                               "Dá para desfazer com 'Restaurar quarentena'."):
        return
    try:
        n = dados.mover_para_quarentena(suspeitos, pasta)
        texto_log_out.escrever(f"\n📦 {n} arquivos movidos para {os.path.join(pasta, dados.PASTA_QUARENTENA)}")
    except Exception as e:
        texto_log_out.escrever(f"\n❌ Erro ao mover: {e}")
    estado_out["suspeitos"] = []
    botao_quarentena.configure(state="disabled")
    definir_pasta_csv(pasta)  # atualiza as contagens em todas as abas


def restaurar_quarentena():
    pasta = dados_out.pasta
    if not pasta or not messagebox.askyesno("Restaurar", "Devolver todos os arquivos da QUARENTENA para o lugar original?"):
        return
    n = dados.restaurar_quarentena(pasta)
    texto_log_out.escrever(f"\n↩ {n} arquivos restaurados.")
    definir_pasta_csv(pasta)


dados_out = CartaoDados(col_out, ao_mudar_dados_out, minimo=5)
c = cartao(col_out, "⚙️ Opções")
l = linha(c)
ctk.CTkLabel(l, text="Contaminação (%):").pack(side="left")
entrada_contaminacao = ctk.CTkEntry(l, width=70)
entrada_contaminacao.insert(0, "3")
entrada_contaminacao.pack(side="left", padx=8)
ctk.CTkLabel(c, text="Fração dos espectros que será marcada como suspeita.", text_color=COR_NEUTRA,
             font=ctk.CTkFont(size=11), anchor="w").pack(fill="x", padx=12)
ctk.CTkLabel(c, text="Cores por:", anchor="w").pack(fill="x", padx=12, pady=(6, 0))
menu_cor_out = ctk.CTkOptionMenu(c, values=["Aguardando..."], state="disabled", dynamic_resizing=False)
menu_cor_out.pack(fill="x", padx=12, pady=(3, 10))
botao_gerar_out = ctk.CTkButton(col_out, text="🎯 1. Rastrear outliers", command=acionar_motor_outliers,
                                state="disabled", fg_color=VERMELHO[0], hover_color=VERMELHO[1], height=38)
botao_gerar_out.pack(fill="x", padx=6, pady=6)
texto_log_out = Terminal(col_out, height=200)
texto_log_out.pack(fill="x", padx=6, pady=4)
botao_quarentena = ctk.CTkButton(col_out, text="📦 2. Mover suspeitos para a QUARENTENA", command=enviar_para_quarentena,
                                 state="disabled", fg_color=LARANJA[0], hover_color=LARANJA[1])
botao_quarentena.pack(fill="x", padx=6, pady=4)
botao_restaurar = ctk.CTkButton(col_out, text="↩ Restaurar quarentena (0)", command=restaurar_quarentena,
                                state="disabled", fg_color="transparent", border_width=1)
botao_restaurar.pack(fill="x", padx=6, pady=4)


# ==============================================================================
# ABA 4: CURVA DE VARIÂNCIA
# ==============================================================================
col_var, painel_var = montar_aba(aba_variancia, "Escolha a pasta de .csv e clique em 'Gerar curva'.\n"
                                                "O gráfico mostra quantos PCs guardam a informação química.")


def ao_mudar_dados_var(cartao_dados):
    botao_gerar_var.configure(state="normal" if cartao_dados.pronto() else "disabled")


def acionar_motor_variancia():
    arquivos, regiao = dados_var.arquivos, dados_var.regiao()
    max_pcs = ler_numero(entrada_max_pcs, 20, 2, 50)
    meta = ler_numero(entrada_meta, 95.0, 50.0, 99.9, float)
    padronizar = bool(check_padronizar_var.get())
    botao_gerar_var.configure(state="disabled")
    painel_var.aviso(f"Lendo {len(arquivos)} espectros e calculando...")

    def tarefa():
        X, _, _ = dados.carregar_espectros(arquivos)
        return calcular_variancia(X, max_pcs=max_pcs, padronizar=padronizar, meta=meta)

    def terminou(res):
        botao_gerar_var.configure(state="normal")
        painel_var.desenhar(graficos.variancia, res, regiao, titulo_extra="" if padronizar else " (só centrado)")
        pc = res["pcs_para_meta"]
        if pc:
            label_resultado_var.configure(text=f"{pc} PCs", text_color=COR_OK)
            label_resultado_var_sub.configure(text=f"para explicar {meta:.0f}% da variância ({regiao})")
        else:
            label_resultado_var.configure(text=f"> {len(res['var'])} PCs", text_color=COR_AVISO)
            label_resultado_var_sub.configure(text=f"{len(res['var'])} PCs explicam só {res['acumulada'][-1]:.1f}%")
        status(f"Curva de variância {regiao} pronta.", COR_OK)

    def falhou(erro):
        botao_gerar_var.configure(state="normal")
        painel_var.aviso(f"Erro:\n{erro}", cor=COR_ERRO)

    em_segundo_plano(tarefa, terminou, falhou)


dados_var = CartaoDados(col_var, ao_mudar_dados_var)
c = cartao(col_var, "⚙️ Opções")
l = linha(c)
ctk.CTkLabel(l, text="Máximo de PCs:").pack(side="left")
entrada_max_pcs = ctk.CTkEntry(l, width=60)
entrada_max_pcs.insert(0, "20")
entrada_max_pcs.pack(side="right")
l = linha(c)
ctk.CTkLabel(l, text="Meta de variância (%):").pack(side="left")
entrada_meta = ctk.CTkEntry(l, width=60)
entrada_meta.insert(0, "95")
entrada_meta.pack(side="right")
check_padronizar_var = ctk.CTkCheckBox(c, text="Padronizar (StandardScaler)")
check_padronizar_var.select()
check_padronizar_var.pack(anchor="w", padx=12, pady=(4, 10))
botao_gerar_var = ctk.CTkButton(col_var, text="📈 Gerar curva de variância", command=acionar_motor_variancia,
                                state="disabled", fg_color=AZUL[0], hover_color=AZUL[1], height=38)
botao_gerar_var.pack(fill="x", padx=6, pady=6)
c = cartao(col_var, "Resultado")
label_resultado_var = ctk.CTkLabel(c, text="-", font=ctk.CTkFont(size=34, weight="bold"))
label_resultado_var.pack(pady=(0, 0))
label_resultado_var_sub = ctk.CTkLabel(c, text="", text_color=COR_NEUTRA)
label_resultado_var_sub.pack(pady=(0, 10))


# ==============================================================================
# ABA 5: CLASSIFICAÇÃO SUPERVISIONADA (KNN / RANDOM FOREST)
# ==============================================================================
estado_clf = {
    "opcoes_rotulo": [],   # [(nivel, texto)]
    "pacote": None,        # modelo treinado (ou carregado) pronto para usar
    "relatorio": "",       # texto do último resultado (vai junto ao salvar)
    "resultado": None,     # resultado da validação (para trocar de gráfico)
}

col_clf, painel_clf = montar_aba(aba_classificacao, "Escolha a pasta de .csv (uma subpasta por classe)\n"
                                                    "e clique em 'Treinar e avaliar'.", espaco_acima_do_grafico=True)


def ao_mudar_dados_clf(cartao_dados):
    arquivos = cartao_dados.arquivos
    opcoes = motor_classificacao.opcoes_de_rotulo(arquivos, cartao_dados.pasta) if arquivos else []
    estado_clf["opcoes_rotulo"] = opcoes
    if opcoes:
        textos = [t for _, t in opcoes]
        menu_rotulo_clf.configure(values=textos, state="normal")
        menu_rotulo_clf.set(textos[0])
        botao_treinar_clf.configure(state="normal")
    else:
        aviso = "Precisa de subpastas diferentes (uma por classe)" if arquivos else "Aguardando..."
        menu_rotulo_clf.configure(values=[aviso], state="disabled")
        menu_rotulo_clf.set(aviso)
        botao_treinar_clf.configure(state="disabled")


def trocar_modelo_clf(*_):
    """Mostra só os parâmetros do modelo escolhido."""
    if seletor_modelo_clf.get() == "KNN":
        linha_arvores.pack_forget()
        linha_k.pack(fill="x", padx=12, pady=3, before=dica_modelo)
        linha_pcs.pack(fill="x", padx=12, pady=3, before=dica_modelo)
        dica_modelo.configure(text="Padronização → PCA → KNN (k vizinhos mais próximos).")
    else:
        linha_k.pack_forget()
        linha_pcs.pack_forget()
        linha_arvores.pack(fill="x", padx=12, pady=3, before=dica_modelo)
        dica_modelo.configure(text="Random Forest direto no espectro: mostra quais\ncomprimentos de onda decidem a classe.")


def mostrar_grafico_clf(*_):
    res, pacote = estado_clf["resultado"], estado_clf["pacote"]
    if res is None or pacote is None:
        return
    regiao = pacote["regiao"]
    escolha = seletor_grafico_clf.get()
    if escolha == "Acurácia × k" and res["curva_k"]:
        painel_clf.desenhar(graficos.curva_k, res, pacote["parametros"]["k"], pacote["parametros"]["n_pcs"], regiao)
    elif escolha == "Importância" and motor_classificacao.importancia_rf(pacote) is not None:
        painel_clf.desenhar(graficos.importancia_rf, *motor_classificacao.importancia_rf(pacote), regiao)
    else:
        seletor_grafico_clf.set("Matriz de confusão")
        painel_clf.desenhar(graficos.matriz_confusao, res, regiao)


def treinar_clf():
    if not dados_clf.arquivos:
        return
    texto_rotulo = menu_rotulo_clf.get()
    nivel = next(n for n, t in estado_clf["opcoes_rotulo"] if t == texto_rotulo)
    tipo = "KNN" if seletor_modelo_clf.get() == "KNN" else "RF"
    k = ler_numero(entrada_k_clf, 5, 1, 99)
    n_pcs = ler_numero(entrada_pcs_clf, 10, 1, 200)
    n_arvores = ler_numero(entrada_arvores_clf, 300, 10, 5000)
    pasta, arquivos, regiao = dados_clf.pasta, dados_clf.arquivos, dados_clf.regiao()
    log_clf = texto_log_clf.escrever

    texto_log_clf.limpar()
    botao_treinar_clf.configure(state="disabled")
    label_acuracia.configure(text="...", text_color=COR_AVISO)
    label_acuracia_sub.configure(text="Treinando em segundo plano...")
    painel_clf.aviso("Treinando e validando...")

    def tarefa():
        log_clf(f"Lendo {len(arquivos)} espectros {regiao}...")
        X, y, grupos, eixo, _ = motor_classificacao.carregar_dataset(arquivos, pasta, nivel, callback=log_clf)
        log_clf(f"Matriz de dados: {X.shape[0]} espectros x {X.shape[1]} comprimentos de onda")
        log_clf(motor_classificacao.resumo_dataset(y, grupos))

        nome = f"KNN (k={k}, {n_pcs} PCs)" if tipo == "KNN" else f"Random Forest ({n_arvores} árvores)"
        log_clf(f"\nValidação cruzada agrupada por amostra - {nome}")
        res = motor_classificacao.avaliar_modelo(X, y, grupos, tipo, k=k, n_pcs=n_pcs,
                                                 n_arvores=n_arvores, callback=log_clf)
        log_clf("Treinando o modelo final com todos os dados...")
        pacote = motor_classificacao.treinar_modelo_final(X, y, tipo, eixo, regiao, nivel,
                                                          k=k, n_pcs=n_pcs, n_arvores=n_arvores)
        linhas = [
            "=" * 50,
            f"RESULTADO - {nome} - região {regiao}",
            "=" * 50,
            f"Acurácia por espectro: {res['acuracia'] * 100:.1f}%",
            f"Acurácia por amostra (voto da maioria): {res['acuracia_por_amostra'] * 100:.1f}% "
            f"({res['acertos_amostra']}/{res['total_amostras']} amostras)",
            "",
            res["relatorio"],
        ]
        if res["curva_k"]:
            melhor_k = max(res["curva_k"], key=res["curva_k"].get)
            linhas.append(f"Melhor k na curva: {melhor_k} ({res['curva_k'][melhor_k] * 100:.1f}%)")
        imp = motor_classificacao.importancia_rf(pacote)
        if imp is not None:
            linhas.append("Comprimentos de onda mais importantes:")
            linhas.append(motor_classificacao.linhas_npk_proximas(*imp))
        relatorio = "\n".join(linhas)
        log_clf("\n" + relatorio)
        return res, pacote, relatorio, nome

    def terminou(r):
        res, pacote, relatorio, nome = r
        estado_clf.update(resultado=res, pacote=pacote, relatorio=relatorio)
        botao_treinar_clf.configure(state="normal")
        botao_salvar_clf.configure(state="normal")
        botao_prever_clf.configure(state="normal")
        acc = res["acuracia"] * 100
        label_acuracia.configure(text=f"{acc:.1f}%", text_color=COR_OK if acc >= 80 else COR_AVISO)
        label_acuracia_sub.configure(text=f"{nome}\npor amostra: {res['acuracia_por_amostra'] * 100:.1f}% "
                                          f"({res['acertos_amostra']}/{res['total_amostras']})")
        valores = ["Matriz de confusão"] + (["Acurácia × k"] if res["curva_k"] else []) + \
                  (["Importância"] if tipo == "RF" else [])
        seletor_grafico_clf.configure(values=valores)
        seletor_grafico_clf.set("Matriz de confusão")
        mostrar_grafico_clf()
        status(f"{nome}: {acc:.1f}% de acurácia ({regiao}).", COR_OK)

    def falhou(erro):
        log_clf(f"\n❌ ERRO: {erro}")
        botao_treinar_clf.configure(state="normal")
        label_acuracia.configure(text="Erro", text_color=COR_ERRO)
        label_acuracia_sub.configure(text="Veja o terminal abaixo.")
        painel_clf.aviso(f"Erro no treino:\n{erro}", cor=COR_ERRO)

    em_segundo_plano(tarefa, terminou, falhou)


def salvar_modelo_clf():
    pacote = estado_clf["pacote"]
    if not pacote:
        return
    caminho = filedialog.asksaveasfilename(title="Salvar modelo treinado", defaultextension=".joblib",
                                           filetypes=[("Modelo AgroLIBS", "*.joblib")],
                                           initialfile=f"modelo_{pacote['tipo']}_{pacote['regiao']}.joblib")
    if caminho:
        try:
            motor_classificacao.salvar_modelo(pacote, caminho)
            with open(os.path.splitext(caminho)[0] + "_relatorio.txt", "w", encoding="utf-8") as f:
                f.write(estado_clf["relatorio"])
            texto_log_clf.escrever(f"\n💾 Modelo salvo em {caminho} (relatório .txt ao lado)")
        except Exception as e:
            texto_log_clf.escrever(f"\n❌ Não consegui salvar: {e}")


def carregar_modelo_clf():
    caminho = filedialog.askopenfilename(title="Abrir modelo treinado", filetypes=[("Modelo AgroLIBS", "*.joblib")])
    if caminho:
        try:
            pacote = motor_classificacao.carregar_modelo(caminho)
            estado_clf.update(pacote=pacote, resultado=None)
            botao_prever_clf.configure(state="normal")
            botao_salvar_clf.configure(state="disabled")
            label_acuracia.configure(text="📂", text_color=COR_NEUTRA)
            label_acuracia_sub.configure(text=f"Modelo carregado: {pacote['tipo']} ({pacote['regiao']})")
            texto_log_clf.escrever(f"\n📂 Modelo carregado: {pacote['tipo']} ({pacote['regiao']}) - {os.path.basename(caminho)}")
        except Exception as e:
            texto_log_clf.escrever(f"\n❌ Não consegui abrir o modelo: {e}")


def classificar_pasta_clf():
    pacote = estado_clf["pacote"]
    if not pacote:
        return
    pasta = filedialog.askdirectory(title=f"Pasta com os .csv das amostras novas ({pacote['regiao']})")
    if not pasta:
        return
    botao_prever_clf.configure(state="disabled")
    texto_log_clf.escrever(f"\n🔎 Classificando os .csv de {pasta}...")

    def tarefa():
        por_espectro, por_amostra = motor_classificacao.classificar_pasta(pacote, pasta)
        destino = os.path.join(pasta, f"classificacao_{pacote['tipo']}_{pacote['regiao']}.csv")
        por_espectro.to_csv(destino, sep=";", index=False)
        return por_espectro, por_amostra, destino

    def terminou(r):
        por_espectro, por_amostra, destino = r
        botao_prever_clf.configure(state="normal")
        texto_log_clf.escrever(f"Classificação de {len(por_espectro)} espectros ({pacote['tipo']}):")
        for _, ln in por_amostra.iterrows():
            texto_log_clf.escrever(f"  {ln['amostra']}: {ln['classe_final']} "
                                   f"({ln['votos_na_classe']}/{ln['disparos']} disparos)")
        texto_log_clf.escrever(f"Detalhe por espectro salvo em: {destino}")
        status(f"{len(por_amostra)} amostras classificadas. Detalhes em {destino}", COR_OK)

    def falhou(erro):
        botao_prever_clf.configure(state="normal")
        texto_log_clf.escrever(f"\n❌ ERRO ao classificar: {erro}")

    em_segundo_plano(tarefa, terminou, falhou)


# --- Layout da Aba 5 ---
dados_clf = CartaoDados(col_clf, ao_mudar_dados_clf)
ctk.CTkLabel(dados_clf, text="Classificar por:", anchor="w").pack(fill="x", padx=12)
menu_rotulo_clf = ctk.CTkOptionMenu(dados_clf, values=["Aguardando..."], state="disabled", dynamic_resizing=False)
menu_rotulo_clf.pack(fill="x", padx=12, pady=(3, 10))

c = cartao(col_clf, "🤖 Modelo")
seletor_modelo_clf = ctk.CTkSegmentedButton(c, values=["KNN", "Random Forest"], command=trocar_modelo_clf)
seletor_modelo_clf.set("KNN")
seletor_modelo_clf.pack(fill="x", padx=12, pady=3)


def linha_com_entrada(master, texto, valor):
    frame = ctk.CTkFrame(master, fg_color="transparent")
    ctk.CTkLabel(frame, text=texto).pack(side="left")
    entrada = ctk.CTkEntry(frame, width=70)
    entrada.insert(0, valor)
    entrada.pack(side="right")
    return frame, entrada


linha_k, entrada_k_clf = linha_com_entrada(c, "k (vizinhos):", "5")
linha_pcs, entrada_pcs_clf = linha_com_entrada(c, "Nº de PCs:", "10")
linha_arvores, entrada_arvores_clf = linha_com_entrada(c, "Nº de árvores:", "300")
dica_modelo = ctk.CTkLabel(c, text="", text_color=COR_NEUTRA, font=ctk.CTkFont(size=11), justify="left", anchor="w")
dica_modelo.pack(fill="x", padx=12, pady=(2, 10))
trocar_modelo_clf()

botao_treinar_clf = ctk.CTkButton(col_clf, text="🧠 Treinar e avaliar", command=treinar_clf, state="disabled",
                                  fg_color=ROXO[0], hover_color=ROXO[1], height=38)
botao_treinar_clf.pack(fill="x", padx=6, pady=6)

c = cartao(col_clf, "Acurácia (validação por amostra)")
label_acuracia = ctk.CTkLabel(c, text="-", font=ctk.CTkFont(size=34, weight="bold"))
label_acuracia.pack()
label_acuracia_sub = ctk.CTkLabel(c, text="Todos os disparos de uma amostra ficam juntos\nno treino ou no teste.",
                                  text_color=COR_NEUTRA, justify="center")
label_acuracia_sub.pack(pady=(0, 10))

l = ctk.CTkFrame(col_clf, fg_color="transparent")
l.pack(fill="x", padx=6, pady=3)
l.grid_columnconfigure((0, 1), weight=1)
botao_salvar_clf = ctk.CTkButton(l, text="💾 Salvar modelo", command=salvar_modelo_clf, state="disabled")
botao_salvar_clf.grid(row=0, column=0, padx=(0, 3), sticky="ew")
ctk.CTkButton(l, text="📂 Carregar modelo", command=carregar_modelo_clf).grid(row=0, column=1, padx=(3, 0), sticky="ew")
botao_prever_clf = ctk.CTkButton(col_clf, text="🔎 Classificar amostras novas", command=classificar_pasta_clf,
                                 state="disabled", fg_color=VERDE[0], hover_color=VERDE[1])
botao_prever_clf.pack(fill="x", padx=6, pady=3)
texto_log_clf = Terminal(col_clf, height=220)
texto_log_clf.pack(fill="x", padx=6, pady=6)

# Seletor de gráfico, em cima do painel da direita
seletor_grafico_clf = ctk.CTkSegmentedButton(aba_classificacao, values=["Matriz de confusão"], command=mostrar_grafico_clf)
seletor_grafico_clf.set("Matriz de confusão")
seletor_grafico_clf.grid(row=0, column=1, sticky="w", pady=(0, 6))


# ==============================================================================
# ENCERRAMENTO SEGURO DA APLICAÇÃO
# ==============================================================================
def ao_fechar():
    """Fecha a janela de forma limpa, mesmo com alguma thread ainda rodando."""
    try:
        janela.quit()
        janela.destroy()
    except Exception:
        pass
    finally:
        os._exit(0)


janela.protocol("WM_DELETE_WINDOW", ao_fechar)

# Executa o loop principal
if __name__ == "__main__":
    janela.after(40, _processar_fila)
    janela.mainloop()
