# AgroLIBS Studio V2.0 🔬🌱

**AgroLIBS Studio** é uma aplicação desktop desenvolvida em Python para o pré-processamento e análise quimiométrica de espectros obtidos via Espectroscopia de Plasma Induzido por Laser (LIBS). O foco primário do software é a caracterização da variabilidade elementar de fertilizantes comerciais.

Este projeto foi desenvolvido com uma arquitetura modular, separando a interface gráfica dos motores matemáticos, garantindo alta performance, blindagem contra erros de leitura e facilidade de manutenção.

---

## 🚀 Novidades da Versão 2.0
* **Exterminador por IA:** O antigo filtro Z-Score foi substituído pelo algoritmo de Machine Learning **Isolation Forest**, que previne o mascaramento estatístico de anomalias extremas.
* **PCA Híbrido:** Opção de projeção limpa em 2D (ideal para artigos) e visualização exploratória em 3D.
* **Filtros Espectrais Globais:** Interceptador interativo que permite isolar a análise química exclusivamente para as regiões do Ultravioleta (UV) ou Visível (VIS).

---

## ⚙️ Funcionalidades

O software é dividido em 4 abas principais, simulando um pipeline completo de Ciência de Dados para espectroscopia:

1. **Pré-processamento de Dados:**
   * Leitura adaptativa de arquivos `.esf` brutos.
   * Filtro Savitzky-Golay (suavização de ruídos), Correção de Baseline (ajuste polinomial) e Normalização por Área.
   * Preview visual integrado com identificação de picos NPK.
   * Processamento em lote (Batch) gerando `.csv` limpos.

2. **Análise PCA (Componentes Principais):**
   * Agrupamento automatizado das amostras por Concentração e Marca.
   * Geração de gráficos de Escores (PC1 vs PC2) renderizados em 2D e 3D.
   * "Vacina LIBS" para alinhamento automático de tensores de disparo.

3. **Rastreamento e Extermínio de Anomalias (Outliers):**
   * Motor de Machine Learning (Isolation Forest) para detecção de impurezas severas.
   * Sistema de exclusão definitiva (Extermínio) do dataset com Escudo Anti-Zumbi, garantindo que arquivos deletados não contaminem futuras análises.

4. **Curva de Variância Explicada:**
   * Cálculo de quantos Componentes Principais são necessários para reter a informação química original.
   * Gráfico de alto padrão (limitado a 20 PCs) com anotação automática do limite de 95% de variância.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.13
* **Interface Gráfica:** `customtkinter`
* **Manipulação de Dados:** `pandas`, `numpy`
* **Machine Learning & Estatística:** `scikit-learn`, `scipy`
* **Visualização Científica:** `matplotlib`, `seaborn`

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
    ├── motor_pca.py       # Renderização Híbrida de Componentes Principais
    ├── motor_outliers.py  # Algoritmo Isolation Forest (Machine Learning)
    └── motor_variancia.py # Cálculo e plotagem da variância acumulada
