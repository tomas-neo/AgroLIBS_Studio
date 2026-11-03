# Arquivo: modulos/graficos.py
"""
Todos os gráficos do AgroLIBS Studio.

Cada função recebe uma Figure do matplotlib já criada (a que fica dentro da
janela do app) e desenha nela. Assim os gráficos aparecem dentro da aba, com
a barra de ferramentas do matplotlib (zoom, mover, salvar PNG em 300 dpi).
"""
import re

import matplotlib as mpl
import numpy as np
from matplotlib.patches import Ellipse

from modulos.pre_processador import ELEMENTS_WAVELENGTHS

CORES_NPK = {"N": "#2e86de", "P": "#e67e22", "K": "#8e44ad"}
COR_PRINCIPAL = "#3b2c85"   # a cor que você escolheu para a curva de variância
COR_ALERTA = "#cc3333"

# Paleta para os grupos (até 20 cores bem diferentes)
_PALETA = [mpl.colormaps["tab10"](i) for i in range(10)] + \
          [mpl.colormaps["tab20"](i) for i in range(1, 20, 2)]


def aplicar_estilo():
    """Visual padrão dos gráficos (chamado uma vez quando o app abre)."""
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.6,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "legend.frameon": False,
        "font.size": 10,
        "savefig.dpi": 300,          # PNG salvo pela barra já sai pronto para o TCC
        "savefig.bbox": "tight",
    })


def ordem_natural(texto):
    """'Adulterado_5' vem antes de 'Adulterado_10'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(texto))]


def cor_do_grupo(i):
    return _PALETA[i % len(_PALETA)]


def mensagem(fig, texto, cor="#7f8c8d"):
    """Figura vazia com um aviso no meio (antes de gerar o gráfico)."""
    fig.clear()
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.5, texto, ha="center", va="center", fontsize=12, color=cor,
            transform=ax.transAxes, wrap=True)


def _marcar_linhas_npk(ax, eixo, com_legenda=True):
    vistos = set()
    for el, linhas in ELEMENTS_WAVELENGTHS.items():
        for wl in linhas:
            if eixo[0] <= wl <= eixo[-1]:
                ax.axvline(wl, color=CORES_NPK[el], linestyle=":", alpha=0.7, linewidth=1,
                           label=f"Linha de {el}" if com_legenda and el not in vistos else None)
                vistos.add(el)
    return vistos


# ==============================================================================
# ABA 1: ESPECTRO ANTES / DEPOIS
# ==============================================================================
def espectro_antes_depois(fig, wl, bruto, tratado, nome, tags, picos):
    fig.clear()
    ax1, ax2 = fig.subplots(2, 1, sharex=True)

    ax1.plot(wl, bruto, color="#7f8c8d", linewidth=0.7)
    ax1.set_title(f"Espectro bruto: {nome}", fontsize=11)
    ax1.set_ylabel("Intensidade (u.a.)")

    ax2.plot(wl, tratado, color="#2c3e50", linewidth=0.8)
    ax2.set_title(f"Espectro tratado ({tags}) com linhas N, P, K", fontsize=11)
    ax2.set_xlabel("Comprimento de onda (nm)")
    ax2.set_ylabel("Intensidade (u.a.)")

    # Vários pontos de um mesmo pico (ou de um dubleto) caem perto da mesma linha:
    # anotamos só o mais alto de cada linha de referência, para o texto não embolar.
    melhores = {}
    for pico in picos:
        chave = (pico["elemento"], pico["w_referencia"])
        if chave not in melhores or pico["intensidade"] > melhores[chave]["intensidade"]:
            melhores[chave] = pico

    vistos = set()
    for i, pico in enumerate(sorted(melhores.values(), key=lambda p: p["w_medido"])):
        el = pico["elemento"]
        cor = CORES_NPK[el]
        ax2.axvline(pico["w_medido"], color=cor, linestyle=":", alpha=0.6)
        ax2.scatter(pico["w_medido"], pico["intensidade"], color=cor, s=40, zorder=5,
                    label=f"Linha de {el}" if el not in vistos else None)
        # Rótulo em pé, no topo do gráfico: não embola quando duas linhas estão próximas
        ax2.text(pico["w_medido"], 0.98, f" {el} {pico['w_medido']:.1f}", rotation=90, fontsize=7,
                 fontweight="bold", color=cor, va="top", ha="right" if i % 2 else "left",
                 transform=ax2.get_xaxis_transform())
        vistos.add(el)
    if vistos:
        ax2.legend(loc="upper left")
    fig.tight_layout()


# ==============================================================================
# ABA 2: PCA
# ==============================================================================
def _elipse_95(ax, x, y, cor):
    """Elipse de confiança de 95% de um grupo (precisa de 3+ pontos)."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)) or np.linalg.det(cov) <= 0:
        return
    valores, vetores = np.linalg.eigh(cov)
    angulo = np.degrees(np.arctan2(vetores[1, -1], vetores[0, -1]))
    escala = np.sqrt(5.991)  # qui-quadrado com 2 graus de liberdade, 95%
    largura, altura = 2 * escala * np.sqrt(valores[::-1])
    ax.add_patch(Ellipse((np.mean(x), np.mean(y)), largura, altura, angle=angulo,
                         facecolor=cor, edgecolor=cor, alpha=0.12, linewidth=1.2))


def pca_escores(fig, escores, grupos, var_exp, regiao, em_3d=False, elipses=True, titulo_extra=""):
    fig.clear()
    grupos = np.asarray(grupos)
    unicos = sorted(set(grupos), key=ordem_natural)
    em_3d = em_3d and escores.shape[1] >= 3
    ax = fig.add_subplot(111, projection="3d" if em_3d else None)

    for i, g in enumerate(unicos):
        m = grupos == g
        cor = cor_do_grupo(i)
        rotulo = f"{g} ({m.sum()})"
        if em_3d:
            ax.scatter(escores[m, 0], escores[m, 1], escores[m, 2], color=cor, s=40,
                       alpha=0.85, edgecolors="white", linewidths=0.5, label=rotulo)
        else:
            ax.scatter(escores[m, 0], escores[m, 1], color=cor, s=50, alpha=0.85,
                       edgecolors="white", linewidths=0.6, label=rotulo, zorder=3)
            if elipses:
                _elipse_95(ax, escores[m, 0], escores[m, 1], cor)

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
    if em_3d:
        ax.set_zlabel(f"PC3 ({var_exp[2]:.1f}%)")
    else:
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.axvline(0, color="gray", linewidth=0.6)
    ax.set_title(f"PCA {'3D' if em_3d else '2D'} - {regiao.upper()}{titulo_extra}")
    ax.legend(title="Grupo (nº de espectros)", bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=9, title_fontsize=9)
    fig.tight_layout()


def pca_loadings(fig, eixo, loadings, var_exp, regiao, n_pcs=2, titulo_extra=""):
    """Peso de cada comprimento de onda nos primeiros PCs, com as linhas N, P, K."""
    fig.clear()
    n_pcs = min(n_pcs, len(loadings))
    eixos = fig.subplots(n_pcs, 1, sharex=True, squeeze=False)[:, 0]
    for i, ax in enumerate(eixos):
        ax.plot(eixo, loadings[i], color=COR_PRINCIPAL, linewidth=0.7)
        ax.axhline(0, color="gray", linewidth=0.6)
        vistos = _marcar_linhas_npk(ax, eixo, com_legenda=(i == 0))
        ax.set_ylabel(f"Loading PC{i + 1}\n({var_exp[i]:.1f}%)")
        if i == 0 and vistos:
            ax.legend(loc="upper right", fontsize=9)
    eixos[0].set_title(f"Loadings do PCA - {regiao.upper()}{titulo_extra}")
    eixos[-1].set_xlabel("Comprimento de onda (nm)")
    fig.tight_layout()


# ==============================================================================
# ABA 3: OUTLIERS
# ==============================================================================
def outliers(fig, escores, eh_outlier, nomes, var_exp, regiao, grupos=None):
    fig.clear()
    ax = fig.add_subplot(111)
    normais = ~eh_outlier
    if grupos is not None:
        grupos = np.asarray(grupos)
        for i, g in enumerate(sorted(set(grupos), key=ordem_natural)):
            m = normais & (grupos == g)
            ax.scatter(escores[m, 0], escores[m, 1], color=cor_do_grupo(i), s=40, alpha=0.75,
                       edgecolors="white", linewidths=0.5, label=g, zorder=3)
    else:
        ax.scatter(escores[normais, 0], escores[normais, 1], color="#95a5a6", s=40, alpha=0.75,
                   edgecolors="white", linewidths=0.5, label="Normais", zorder=3)
    ax.scatter(escores[eh_outlier, 0], escores[eh_outlier, 1], facecolors="none",
               edgecolors=COR_ALERTA, s=180, linewidths=2, label="Outlier", zorder=4)
    for i in np.where(eh_outlier)[0]:
        ax.annotate(nomes[i], xy=(escores[i, 0], escores[i, 1]), xytext=(8, 6),
                    textcoords="offset points", fontsize=8, color=COR_ALERTA)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.axvline(0, color="gray", linewidth=0.6)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
    ax.set_title(f"Outliers (Isolation Forest) - {regiao.upper()}: "
                 f"{int(eh_outlier.sum())} de {len(eh_outlier)} espectros")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()


# ==============================================================================
# ABA 4: CURVA DE VARIÂNCIA
# ==============================================================================
def variancia(fig, res, regiao, titulo_extra=""):
    fig.clear()
    ax = fig.add_subplot(111)
    var, acum, meta = res["var"], res["acumulada"], res["meta"]
    idx = np.arange(1, len(var) + 1)

    ax.bar(idx, var, color="#b8b0e0", edgecolor=COR_PRINCIPAL, linewidth=0.6,
           label="Variância de cada PC", zorder=2)
    ax.plot(idx, acum, marker="o", color=COR_PRINCIPAL, linewidth=2, markersize=7,
            label="Variância acumulada", zorder=3)
    ax.axhline(meta, color=COR_ALERTA, linestyle="--", linewidth=1, alpha=0.6)

    for i, v in enumerate(acum):
        ax.text(i + 1, v + 2.5, f"{v:.1f}%", ha="center", fontsize=8)

    pc = res["pcs_para_meta"]
    if pc:
        ax.axvline(pc, color=COR_ALERTA, linestyle="--", linewidth=1.5)
        ax.annotate(f"PC{pc}\n({acum[pc - 1]:.1f}%)", xy=(pc, acum[pc - 1]),
                    xytext=(max(1.5, pc - 2.5), min(acum[pc - 1] - 25, meta - 20)),
                    arrowprops=dict(facecolor=COR_ALERTA, edgecolor=COR_ALERTA, shrink=0.05,
                                    width=2, headwidth=8),
                    color=COR_ALERTA, fontsize=11, fontweight="bold", ha="center")
        subtitulo = f"{pc} PCs explicam {meta:.0f}% da variância"
    else:
        subtitulo = f"{len(var)} PCs não chegam a {meta:.0f}% (ficam em {acum[-1]:.1f}%)"

    ax.set_title(f"Variância explicada - PCA ({regiao.upper()}){titulo_extra}\n{subtitulo}")
    ax.set_xlabel("Número de componentes principais (PCs)")
    ax.set_ylabel("% de variância explicada")
    ax.set_xticks(idx)
    ax.set_ylim(0, max(105, acum[-1] + 10))
    ax.legend(loc="center right")
    fig.tight_layout()


# ==============================================================================
# ABA 5: CLASSIFICAÇÃO
# ==============================================================================
def matriz_confusao(fig, res, regiao):
    fig.clear()
    ax = fig.add_subplot(111)
    classes, m = res["classes"], res["matriz"]
    # Porcentagem por linha (classe verdadeira): fica comparável entre classes
    totais = m.sum(axis=1, keepdims=True)
    pct = np.divide(m, totais, out=np.zeros_like(m, dtype=float), where=totais > 0) * 100
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{m[i, j]}\n{pct[i, j]:.0f}%", ha="center", va="center", fontsize=9,
                    color="white" if pct[i, j] > 55 else "black", fontweight="bold")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.grid(False)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe verdadeira")
    nome = "KNN" if res["tipo"] == "KNN" else "Random Forest"
    ax.set_title(f"Matriz de confusão - {nome} ({regiao.upper()})\n"
                 f"Acurácia {res['acuracia'] * 100:.1f}% por espectro, "
                 f"{res['acuracia_por_amostra'] * 100:.1f}% por amostra ({res['n_folds']} folds)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="% da classe verdadeira")
    fig.tight_layout()


def curva_k(fig, res, k_usado, n_pcs, regiao):
    fig.clear()
    ax = fig.add_subplot(111)
    ks = list(res["curva_k"].keys())
    acc = [v * 100 for v in res["curva_k"].values()]
    ax.plot(ks, acc, "o-", color="#2980b9", linewidth=2)
    melhor = ks[int(np.argmax(acc))]
    ax.scatter([melhor], [max(acc)], s=160, facecolors="none", edgecolors="#27ae60", linewidths=2,
               label=f"Melhor k = {melhor} ({max(acc):.1f}%)", zorder=4)
    if k_usado in res["curva_k"]:
        ax.axvline(k_usado, color=COR_ALERTA, linestyle=":", label=f"k usado = {k_usado}")
    ax.set_xticks(ks)
    ax.set_xlabel("Número de vizinhos (k)")
    ax.set_ylabel("Acurácia (%)")
    ax.set_title(f"KNN: acurácia x k ({regiao.upper()}, {n_pcs} PCs)")
    ax.legend()
    fig.tight_layout()


def importancia_rf(fig, eixo, importancias, regiao):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.plot(eixo, importancias, color="#2c3e50", linewidth=0.8)
    vistos = _marcar_linhas_npk(ax, eixo)
    top = []
    for idx in np.argsort(importancias)[::-1]:   # os 5 maiores, a pelo menos 2 nm um do outro
        if all(abs(eixo[idx] - eixo[j]) > 2 for j in top):
            top.append(idx)
        if len(top) == 5:
            break
    for idx in top:
        ax.annotate(f"{eixo[idx]:.2f}", xy=(eixo[idx], importancias[idx]), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=8, color=COR_ALERTA)
    if vistos:
        ax.legend(loc="upper left")
    ax.set_xlabel("Comprimento de onda (nm)")
    ax.set_ylabel("Importância (Gini)")
    ax.set_title(f"Random Forest: comprimentos de onda mais importantes ({regiao.upper()})")
    fig.tight_layout()
