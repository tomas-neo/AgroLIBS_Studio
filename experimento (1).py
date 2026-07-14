import tkinter as tk
from tkinter import messagebox, simpledialog
import random
import os
import csv
from datetime import datetime

# --- MAPEAMENTO DE RELAÇÕES (MÉTODO) ---
MAPA_CORRETO = {
    'AB': {'A1': 'B1', 'A2': 'B2', 'A3': 'B3'},
    'BA': {'B1': 'A1', 'B2': 'A2', 'B3': 'A3'},
    'AG': {'A1': 'G1', 'A2': 'G2', 'A3': 'G3'},
    'GA': {'G1': 'A1', 'G2': 'A2', 'G3': 'A3'},
    'BG': {'B1': 'G1', 'B2': 'G2', 'B3': 'G3'},
    'EF': {'E1': 'F1', 'E2': 'F2', 'E3': 'F3'},
    'FE': {'F1': 'E1', 'F2': 'E2', 'F3': 'E3'}
}

# --- MATRIZ CONDICIONAL PARA A FASE DE ENSINO EF ---
MAPA_FEEDBACK_EF_CONDICIONAL = {
    'E1': {'F1': 'B1', 'F2': 'B2', 'F3': 'B3'},
    'E2': {'F1': 'B3', 'F2': 'B1', 'F3': 'B2'},
    'E3': {'F1': 'B2', 'F2': 'B3', 'F3': 'B1'}
}

ESTIMULOS_POR_CLASSE = {
    'A': ['A1', 'A2', 'A3'], 'B': ['B1', 'B2', 'B3'], 'G': ['G1', 'G2', 'G3'],
    'E': ['E1', 'E2', 'E3'], 'F': ['F1', 'F2', 'F3']
}

class ExperimentoEquivalencia:
    def __init__(self, root):
        self.root = root
        self.root.withdraw()
        
        self.id_participante = simpledialog.askstring("Identificação", "Digite o Código/Nome do Participante (ex: P01):")
        if not self.id_participante:
            self.id_participante = "Anonimo"
            
        self.root.deiconify()
        self.root.title("MestrePython - Equivalência de Estímulos")
        self.root.geometry("800x650")
        self.root.configure(bg="#F0F0F0")
        
        self.dados_relatorio = []
        
        self.img_vazia = tk.PhotoImage(width=1, height=1)
        self.imagens_cache = {}
        self.carregar_todas_imagens()
        
        self.fases = [
            {"nome": "Pré-Teste AB", "relacao": "AB", "feedback": "none", "max_blocos": 1},
            {"nome": "Ensino AB", "relacao": "AB", "feedback": "padrao", "max_blocos": 99},
            {"nome": "Teste de Simetria BA", "relacao": "BA", "feedback": "none", "max_blocos": 2},
            {"nome": "Ensino AG", "relacao": "AG", "feedback": "padrao", "max_blocos": 99},
            {"nome": "Teste de Simetria GA", "relacao": "GA", "feedback": "none", "max_blocos": 1},
            {"nome": "Teste de Transitividade BG", "relacao": "BG", "feedback": "none", "max_blocos": 2},
            {"nome": "Pré-Teste EF", "relacao": "EF", "feedback": "none", "max_blocos": 1},
            {"nome": "Ensino EF", "relacao": "EF", "feedback": "B_estatg", "max_blocos": 99},
            {"nome": "Teste de Simetria FE", "relacao": "FE", "feedback": "none", "max_blocos": 2}
        ]
        
        self.fase_atual_idx = 0
        self.bloco_atual_num = 1
        self.tentativa_atual_idx = 0
        self.acertos_bloco = 0
        self.bloco_tentativas = []
        
        self.melhor_escore_fase = -1
        self.blocos_sem_progresso = 0
        
        self.lbl_fase = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#F0F0F0", fg="#333333")
        self.lbl_fase.pack(pady=10)
        
        self.lbl_aviso = tk.Label(root, text="", font=("Arial", 10, "italic"), bg="#F0F0F0", fg="#666666")
        self.lbl_aviso.pack(pady=5)
        
        self.lbl_tentativa = tk.Label(root, text="", font=("Arial", 11), bg="#F0F0F0", fg="#333333")
        self.lbl_tentativa.pack(pady=5)
        
        self.frame_modelo = tk.Frame(root, width=150, height=150, bg="#FFFFFF", highlightbackground="#CCCCCC", highlightthickness=1)
        self.frame_modelo.pack(pady=20)
        self.frame_modelo.pack_propagate(False)
        
        self.btn_modelo = tk.Button(self.frame_modelo, bg="#FFFFFF", activebackground="#E0E0E0", relief="groove", bd=2, command=self.mostrar_comparacoes)
        self.btn_modelo.pack(expand=True, fill="both")
        
        self.frame_comparacoes = tk.Frame(root, bg="#F0F0F0")
        self.frame_comparacoes.pack(pady=30)
        
        self.botoes_comp = []
        for i in range(3):
            f_btn = tk.Frame(self.frame_comparacoes, width=160, height=160, bg="#F0F0F0")
            f_btn.grid(row=0, column=i, padx=20)
            f_btn.pack_propagate(False)
            
            btn = tk.Button(f_btn, bg="#FFFFFF", activebackground="#E0E0E0", relief="groove", bd=2,
                            command=lambda idx=i: self.processar_resposta(idx))
            btn.pack(fill="both", expand=True)
            self.botoes_comp.append(btn)
            
        self.tela_preta = tk.Frame(root, bg="black")
        
        self.tela_branca = tk.Frame(root, bg="white")
        self.lbl_feedback_positivo = tk.Label(self.tela_branca, bg="white")
        self.lbl_feedback_positivo.pack(expand=True)
        
        self.iniciar_fase()

    def carregar_todas_imagens(self):
        formatos = ['.png']
        pasta = "imagens"

        if not os.path.exists(pasta):
            return

        for classe, estimulos in ESTIMULOS_POR_CLASSE.items():
            for cod_estimulo in estimulos:
                for ext in formatos:
                    caminho_arquivo = os.path.join(pasta, f"{cod_estimulo}{ext}")
                    if os.path.exists(caminho_arquivo):
                        try:
                            self.imagens_cache[cod_estimulo] = tk.PhotoImage(file=caminho_arquivo)
                            break
                        except Exception:
                            pass
                if cod_estimulo not in self.imagens_cache:
                    self.imagens_cache[cod_estimulo] = None

    def gerar_bloco_valido(self, tipo_relacao):
        classe_modelo = tipo_relacao[0]
        classe_comp = tipo_relacao[1]
        modelos = ESTIMULOS_POR_CLASSE[classe_modelo]
        comps = ESTIMULOS_POR_CLASSE[classe_comp]
        mapa_certo = MAPA_CORRETO[tipo_relacao]
        
        while True:
            # PASSO 1: Criar 9 tentativas base perfeitamente distribuídas (Regras 2 e 3)
            tentativas_base = []
            for mod in modelos:
                correto = mapa_certo[mod]
                errados = [x for x in comps if x != correto]
                
                # Cada modelo terá a resposta correta 1 vez em cada posição
                for pos_certa in [0, 1, 2]:
                    # Baralha a ordem dos errados para garantir a Regra 4
                    random.shuffle(errados)
                    opcoes = [None, None, None]
                    opcoes[pos_certa] = correto
                    
                    idx_errado = 0
                    for i in range(3):
                        if opcoes[i] is None:
                            opcoes[i] = errados[idx_errado]
                            idx_errado += 1
                            
                    tentativas_base.append({
                        'modelo': mod,
                        'opcoes': opcoes,
                        'correto': correto
                    })
            
            # PASSO 2: Ordenar as 9 tentativas validando a Regra 1 (Backtracking)
            random.shuffle(tentativas_base)
            usados = [False] * 9
            
            def backtrack(caminho):
                if len(caminho) == 9:
                    return caminho
                
                for i in range(9):
                    if not usados[i]:
                        candidato = tentativas_base[i]
                        
                        # Validações se já existe uma tentativa anterior
                        if len(caminho) > 0:
                            anterior = caminho[-1]
                            
                            # Regra 1: Nenhum estímulo pode estar na mesma posição da tentativa anterior
                            mesma_posicao = False
                            for p in range(3):
                                if candidato['opcoes'][p] == anterior['opcoes'][p]:
                                    mesma_posicao = True
                                    break
                            if mesma_posicao:
                                continue # Tenta a próxima configuração
                                
                            # Regra Extra (MTS Clássico): Máximo de 2 tentativas seguidas com o mesmo modelo
                            if len(caminho) >= 2:
                                ant2 = caminho[-2]
                                if candidato['modelo'] == anterior['modelo'] == ant2['modelo']:
                                    continue
                                    
                        usados[i] = True
                        resultado = backtrack(caminho + [candidato])
                        if resultado:
                            return resultado
                        usados[i] = False
                return None
            
            bloco_valido = backtrack([])
            # Se encontrou uma combinação válida, retorna. Se não, o "while True" gera novas bases.
            if bloco_valido:
                return bloco_valido

    def iniciar_fase(self):
        fase = self.fases[self.fase_atual_idx]
        self.lbl_fase.config(text=f"FASE: {fase['nome']} (Bloco {self.bloco_atual_num})")
        
        if fase['feedback'] == 'none':
            self.lbl_aviso.config(text="Suas respostas NÃO terão consequências nesta fase.", fg="blue")
        else:
            self.lbl_aviso.config(text="Suas respostas TERÃO consequências nesta fase.", fg="red")
            
        self.acertos_bloco = 0
        self.tentativa_atual_idx = 0
        self.bloco_tentativas = self.gerar_bloco_valido(fase['relacao'])
        self.apresentar_tentativa()

    def apresentar_tentativa(self):
        self.root.configure(bg="#F0F0F0")
        self.lbl_tentativa.config(text=f"Tentativa {self.tentativa_atual_idx + 1} de 9")
        
        dados = self.bloco_tentativas[self.tentativa_atual_idx]
        
        imagem_modelo = self.imagens_cache.get(dados['modelo'])
        if imagem_modelo:
            self.btn_modelo.config(image=imagem_modelo, text="", compound="none", bg="#FFFFFF", state="normal")
        else:
            self.btn_modelo.config(image=self.img_vazia, text=dados['modelo'], compound="center",
                                   font=("Arial", 28, "bold"), fg="#333333", bg="#FFFFFF", state="normal")
            
        for i in range(3):
            self.botoes_comp[i].config(image=self.img_vazia, text="", state="disabled", bg="#F0F0F0", relief="flat")

    def mostrar_comparacoes(self):
        self.btn_modelo.config(state="disabled")
        dados = self.bloco_tentativas[self.tentativa_atual_idx]
        
        for i, opcao in enumerate(dados['opcoes']):
            imagem_comp = self.imagens_cache.get(opcao)
            if imagem_comp:
                self.botoes_comp[i].config(image=imagem_comp, text="", compound="none",
                                           bg="#FFFFFF", state="normal", relief="groove")
            else:
                self.botoes_comp[i].config(image=self.img_vazia, text=opcao, compound="center",
                                           font=("Arial", 22, "bold"), fg="#333333", bg="#FFFFFF", state="normal", relief="groove")

    def processar_resposta(self, botao_idx):
        for btn in self.botoes_comp: btn.config(state="disabled")
            
        fase = self.fases[self.fase_atual_idx]
        dados = self.bloco_tentativas[self.tentativa_atual_idx]
        escolha = dados['opcoes'][botao_idx]
        modelo_atual = dados['modelo']
        
        foi_correto = (escolha == dados['correto'])
        if foi_correto:
            self.acertos_bloco += 1
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        resultado_texto = "CORRETO" if foi_correto else "INCORRETO"
        consequencia_texto = ""
        
        if fase['feedback'] == 'padrao':
            consequencia_texto = "Tela Branca (Acerto)" if foi_correto else "Tela Preta (Time-out)"
            if foi_correto:
                self.lbl_feedback_positivo.config(image=self.img_vazia, text="Você acertou!", compound="center", font=("Arial", 28, "bold"), fg="green")
                self.tela_branca.place(relx=0, rely=0, relwidth=1, relheight=1)
                self.root.after(2000, self.remover_tela_branca)
            else:
                self.tela_preta.place(relx=0, rely=0, relwidth=1, relheight=1)
                self.root.after(2000, self.remover_tela_preta)
                
        elif fase['feedback'] == 'B_estatg':
            id_b_feedback = MAPA_FEEDBACK_EF_CONDICIONAL.get(modelo_atual, {}).get(escolha, "B1")
            consequencia_texto = f"Estímulo {id_b_feedback}"
            
            imagem_b = self.imagens_cache.get(id_b_feedback)
            if imagem_b:
                img_ampliada = imagem_b.zoom(3, 3) 
                self.lbl_feedback_positivo.config(image=img_ampliada, text="", compound="none")
                self.lbl_feedback_positivo.image = img_ampliada 
            else:
                self.lbl_feedback_positivo.config(image=self.img_vazia, text=id_b_feedback, compound="center", font=("Arial", 80, "bold"), fg="#333333")
                
            self.tela_branca.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.root.after(2000, self.remover_tela_branca)
            
        else:
            consequencia_texto = "Sem feedback"
            self.root.after(2000, self.proxima_tentativa)

        linha_relatorio = {
            "ID Participante": self.id_participante,
            "Data/Hora": timestamp,
            "Fase": fase['nome'],
            "Bloco": self.bloco_atual_num,
            "Tentativa": self.tentativa_atual_idx + 1,
            "Estimulo Modelo": modelo_atual,
            "Opcoes Apresentadas": ", ".join(dados['opcoes']),
            "Escolha": escolha,
            "Correto Esperado": dados['correto'],
            "Resultado": resultado_texto,
            "Consequencia Apresentada": consequencia_texto
        }
        self.dados_relatorio.append(linha_relatorio)

    def remover_tela_branca(self):
        self.tela_branca.place_forget()
        self.proxima_tentativa()

    def remover_tela_preta(self):
        self.tela_preta.place_forget()
        self.proxima_tentativa()

    def proxima_tentativa(self):
        self.lbl_aviso.config(text="")
        self.tentativa_atual_idx += 1
        
        if self.tentativa_atual_idx < 9:
            self.apresentar_tentativa()
        else:
            self.avaliar_fim_do_bloco()

    def avaliar_fim_do_bloco(self):
        fase = self.fases[self.fase_atual_idx]
        nome_fase = fase['nome']
        
        if "Pré-Teste" in nome_fase:
            if self.acertos_bloco >= 5:
                messagebox.showerror("Fim de Jogo", f"O participante teve {self.acertos_bloco} acertos no {nome_fase}.\nComo já conhecia as relações, o experimento foi encerrado.")
                self.finalizar_experimento()
            else:
                messagebox.showinfo("Sucesso", f"Aprovado no {nome_fase} ({self.acertos_bloco}/9 acertos).\nAvançando para o Ensino...")
                self.avancar_fase()
        else:
            if self.acertos_bloco == 9:
                messagebox.showinfo("Sucesso", f"Critério de 100% atingido em {nome_fase}! Avançando...")
                self.avancar_fase()
            else:
                if nome_fase == "Ensino EF":
                    if self.acertos_bloco > self.melhor_escore_fase:
                        self.melhor_escore_fase = self.acertos_bloco
                        self.blocos_sem_progresso = 0  
                    else:
                        self.blocos_sem_progresso += 1  

                    if self.blocos_sem_progresso >= 3:
                        messagebox.showerror("Desclassificado", f"O participante não apresentou progresso (melhor escore: {self.melhor_escore_fase}) por 3 blocos consecutivos no {nome_fase}.\n\nExperimento encerrado.")
                        self.finalizar_experimento()
                        return

                if self.bloco_atual_num >= fase.get('max_blocos', 99):
                    messagebox.showerror("Desclassificado", f"O participante não atingiu o critério de 100% no {nome_fase} ({self.acertos_bloco}/9 acertos) após o limite de blocos. Experimento encerrado.")
                    self.finalizar_experimento()
                else:
                    messagebox.showwarning("Repetição", f"Critério não atingido ({self.acertos_bloco}/9 acertos).\nO bloco de {nome_fase} será repetido.")
                    self.bloco_atual_num += 1
                    self.iniciar_fase()

    def avancar_fase(self):
        self.fase_atual_idx += 1
        self.bloco_atual_num = 1
        
        self.melhor_escore_fase = -1 
        self.blocos_sem_progresso = 0
        
        if self.fase_atual_idx < len(self.fases):
            self.iniciar_fase()
        else:
            messagebox.showinfo("Parabéns!", "Fim do Experimento!\nTodos os critérios de equivalência e transferência de função foram alcançados!")
            self.finalizar_experimento()

    def finalizar_experimento(self):
        if self.dados_relatorio:
            id_limpo = "".join(c for c in self.id_participante if c.isalnum() or c in (" ", "_")).replace(" ", "_")
            nome_arquivo = f"Relatorio_{id_limpo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            colunas = ["ID Participante", "Data/Hora", "Fase", "Bloco", "Tentativa", 
                       "Estimulo Modelo", "Opcoes Apresentadas", "Escolha", 
                       "Correto Esperado", "Resultado", "Consequencia Apresentada"]
            
            with open(nome_arquivo, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=colunas, delimiter=';')
                writer.writeheader()
                for linha in self.dados_relatorio:
                    writer.writerow(linha)
                    
            print(f"Relatório gerado com sucesso: {nome_arquivo}")
            
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = ExperimentoEquivalencia(root)
    root.mainloop()