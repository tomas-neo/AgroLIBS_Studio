# AgroLIBS Studio 

**AgroLIBS Studio** é uma aplicação desktop desenvolvida em Python para o pré-processamento e análise quimiométrica de espectros obtidos via Espectroscopia de Plasma Induzido por Laser (LIBS). O foco primário do software é a caracterização da variabilidade elementar de fertilizantes comerciais.

Este projeto foi desenvolvido com uma arquitetura modular, separando a interface gráfica dos motores matemáticos, garantindo alta performance e facilidade de manutenção.

---

##  Funcionalidades

O software é dividido em 4 abas principais, simulando um pipeline completo de Ciência de Dados para espectroscopia:

1. **Pré-processamento de Dados:** * Leitura adaptativa de arquivos `.esf` brutos.
   * Filtro Savitzky-Golay (suavização de ruídos).
   * Correção de Linha de Base (Baseline) por ajuste polinomial iterativo.
   * Normalização pela Área sob a curva.
   * Preview visual "Antes e Depois" integrado.
   * Processamento em lote (Batch) com salvamento automático em `.csv` usando sistema de tags no nome do arquivo (ex: `_SG_BL_Norm`).

2. **Análise PCA (Componentes Principais):**
   * Geração automatizada do Gráfico de Escores (PC1 vs PC2).
   * Padronização de dados (StandardScaler) embutida.

3. **Rastreamento de Anomalias (Outliers):**
   * Motor estatístico que utiliza Z-Score sobre a matriz da PCA.
   * Identificação e plotagem de tiros de laser defeituosos ou impurezas severas (distância > 3 desvios padrões).

4. **Curva de Variância Explicada:**
   * Cálculo de quantos Componentes Principais são necessários para reter a informação química original.
   * Anotação automática do limite de 95% de variância acumulada.

---

##  Tecnologias Utilizadas

* **Linguagem:** Python 3.13
* **Interface Gráfica:** `customtkinter`
* **Manipulação de Dados:** `pandas`, `numpy`
* **Machine Learning & Estatística:** `scikit-learn`, `scipy`
* **Visualização:** `matplotlib`, `seaborn`

---

##  Arquitetura do Projeto

O código segue o princípio de Separação de Preocupações (Separation of Concerns), dividindo a UI da matemática pesada:

```text
AgroLIBS_Studio/
│
├── main.py                  # Cérebro da interface (Frontend CustomTkinter)
├── requirements.txt         # Lista de dependências do projeto
├── README.md                # Documentação
│
└── modulos/                 # Motores matemáticos (Backend)
    ├── pre_processador.py   # Lógica de Savitzky-Golay, Baseline e Leitura
    ├── motor_pca.py         # Cálculos de Componentes Principais
    ├── motor_outliers.py    # Estatística Z-Score para anomalias
    └──
