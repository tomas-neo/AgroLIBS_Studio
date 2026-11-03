# AgroLIBS Studio V2.1 🔬🌱

**AgroLIBS Studio** é uma aplicação desktop desenvolvida em Python para o pré-processamento e análise quimiométrica de espectros obtidos via Espectroscopia de Plasma Induzido por Laser (LIBS). O foco primário do software é a caracterização da variabilidade elementar de fertilizantes comerciais.

Este projeto foi desenvolvido com uma arquitetura modular, separando a interface gráfica dos motores matemáticos, garantindo alta performance, blindagem contra erros de leitura e facilidade de manutenção.

---

## 🚀 Novidades da Versão 2.1
* **Gráficos dentro do app:** todos os gráficos aparecem na própria aba, com a barra do matplotlib (zoom, mover, salvar PNG em 300 dpi, pronto para o TCC).
* **UV/VIS por botão:** a região é escolhida num botão em cada aba, sem as perguntas "Sim = UV / Não = VIS".
* **Uma pasta para todas as abas:** ao terminar o lote da Aba 1, as abas 2 a 5 já apontam para a pasta de `.csv`. Escolher uma pasta numa aba muda nas outras.
* **Quarentena reversível:** os outliers são movidos para `QUARENTENA/` (com confirmação), e não apagados. O botão "Restaurar" desfaz.
* **Mais rápido:** leitura do `.esf` com o leitor em C do pandas e escrita do `.csv` com 10 algarismos significativos (cerca de 2x mais rápido no lote), e os `.csv` lidos ficam em cache entre as abas.
* **Nada congela:** PCA, outliers, variância e classificação rodam em segundo plano.
* **Tema claro/escuro** no canto superior direito.

## 🚀 Novidades da Versão 2.0
* **Exterminador por IA:** O antigo filtro Z-Score foi substituído pelo algoritmo de Machine Learning **Isolation Forest**, que previne o mascaramento estatístico de anomalias extremas.
* **PCA Híbrido:** Opção de projeção limpa em 2D (ideal para artigos) e visualização exploratória em 3D.
* **Filtros Espectrais Globais:** Interceptador interativo que permite isolar a análise química exclusivamente para as regiões do Ultravioleta (UV) ou Visível (VIS).

---

## ⚙️ Funcionalidades

O software é dividido em 5 abas principais, simulando um pipeline completo de Ciência de Dados para espectroscopia:

1. **Pré-processamento de Dados:**
   * Leitura adaptativa de arquivos `.esf` brutos.
   * Filtro Savitzky-Golay (suavização de ruídos), Correção de Baseline (ajuste polinomial) e Normalização por Área.
   * Preview visual integrado com identificação de picos NPK: o gráfico atualiza ao marcar/desmarcar um filtro, com setas ◀ ▶ e busca por nome para percorrer todos os arquivos.
   * Processamento em lote (Batch) gerando `.csv` limpos, com barra de progresso.

2. **Análise PCA (Componentes Principais):**
   * Cores por nível de pasta (ex.: Concentração + Marca, ou só a pasta de adulteração), com o número de espectros de cada grupo na legenda.
   * Escores em 2D (com elipses de confiança de 95%) e 3D, e gráfico de **loadings** com as linhas de N, P e K marcadas.
   * Opção de só centrar na média (sem padronizar), para os picos pesarem mais que o ruído.
   * Alinhamento dos espectros por interpolação na faixa de comprimento de onda comum.

3. **Rastreamento e Extermínio de Anomalias (Outliers):**
   * Motor de Machine Learning (Isolation Forest) sobre o PCA, com a fração de contaminação ajustável e um gráfico que circula os suspeitos.
   * Quarentena reversível com Escudo Anti-Zumbi: os arquivos movidos para `QUARENTENA/` saem de todas as análises.

4. **Curva de Variância Explicada:**
   * Cálculo de quantos Componentes Principais são necessários para reter a informação química original.
   * Barras (variância de cada PC) + curva acumulada, com número máximo de PCs e meta (%) ajustáveis.

5. **Classificação Supervisionada (KNN e Random Forest):**
   * A classe de cada espectro vem do nome de uma pasta (ex.: `Adulterado_25porcento` ou `MaxGreen`); o menu "Classificar por" mostra os níveis de pasta disponíveis.
   * **KNN:** Padronização → PCA → KNN, com k e número de PCs ajustáveis, e curva de acurácia em função de k.
   * **Random Forest:** direto no espectro, com gráfico da importância de cada comprimento de onda e marcação das linhas de N, P e K.
   * Validação cruzada **agrupada por amostra física**: todos os disparos da `amostra_a` ficam juntos no treino ou no teste, para a acurácia não sair inflada.
   * Matriz de confusão, relatório por classe e acurácia por amostra (voto da maioria dos disparos).
   * Salvar/carregar o modelo (`.joblib`) e classificar uma pasta de amostras novas (gera um `.csv` com a classe prevista de cada espectro).

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.13
* **Interface Gráfica:** `customtkinter`
* **Manipulação de Dados:** `pandas`, `numpy`
* **Machine Learning & Estatística:** `scikit-learn`, `scipy`
* **Visualização Científica:** `matplotlib` (embutido na interface com `FigureCanvasTkAgg`)

---

## 💻 Compatibilidade

Atualmente, o executável pré-compilado (`dist/`) foi gerado nativamente para **Linux (Debian/Ubuntu)**.

* **Usuários Linux:** Podem utilizar o executável diretamente.
* **Usuários Windows / macOS:** Devem executar o software através do código-fonte (Python 3.13) ou realizar um novo build (compilação) em suas respectivas máquinas utilizando o `PyInstaller`, seguindo as dependências listadas no arquivo `requirements.txt`.

---

## 📂 Arquitetura do Projeto

O código segue o princípio de Separação de Preocupações (*Separation of Concerns*), dividindo a UI da matemática pesada:

```text
AgroLIBS_Studio/
│
├── main.py                # Cérebro da interface (Frontend CustomTkinter)
├── requirements.txt       # Lista de dependências blindada
├── README.md              # Documentação
│
└── modulos/               # Motores matemáticos (Backend)
    ├── pre_processador.py # Lógica de SG, Baseline e Mapeamento NPK
    ├── conversor_arquivos.py # Conversão em lote .esf -> .csv
    ├── dados.py           # Leitura dos .csv (UV/VIS, grupos, cache, quarentena)
    ├── graficos.py        # Todos os gráficos do app
    ├── motor_pca.py       # Cálculo do PCA (escores, variância, loadings)
    ├── motor_outliers.py  # Algoritmo Isolation Forest (Machine Learning)
    ├── motor_variancia.py # Cálculo da variância acumulada
    └── motor_classificacao.py # KNN e Random Forest com validação por amostra
