# Arquivo: modulos/tema.py
"""
Cores do AgroLIBS Studio.

Cada tema é uma paleta pequena; a função aplicar() transforma essa paleta
num tema do customtkinter (o mesmo formato dos arquivos .json que vêm com
a biblioteca) e devolve a paleta para o main.py usar nos detalhes.
"""
import json
import os
import tempfile

import customtkinter as ctk

TEMAS = {
    "Embrapa Verde": {
        "modo": "Light",
        "fundo": "#eef3ee", "superficie": "#ffffff", "cartao": "#f5f9f5", "borda": "#d3dfd3",
        "campo": "#ffffff", "texto": "#1d2b1f", "suave": "#6b7c6d", "texto_botao": "#ffffff",
        "primaria": "#2e7d32", "hover": "#1b5e20", "selecao": "#c8e6c9", "selecao_hover": "#a5d6a7", "texto_selecao": "#ffffff",
        "sucesso": "#2e7d32", "perigo": "#c62828", "alerta": "#e65100",
        "terminal": "#12261a", "terminal_texto": "#a7f3b8",
    },
    "Noite Azul": {
        "modo": "Dark",
        "fundo": "#0b1220", "superficie": "#111a2e", "cartao": "#17223a", "borda": "#27354f",
        "campo": "#0e1626", "texto": "#e6edf7", "suave": "#8b9bb4", "texto_botao": "#ffffff",
        "primaria": "#3b82f6", "hover": "#2563eb", "selecao": "#2563eb", "selecao_hover": "#1d4ed8", "texto_selecao": "#ffffff",
        "sucesso": "#10b981", "perigo": "#ef4444", "alerta": "#f59e0b",
        "terminal": "#060b16", "terminal_texto": "#7dd3fc",
    },
    "Laboratório": {
        "modo": "Light",
        "fundo": "#f3f5f8", "superficie": "#ffffff", "cartao": "#f7f9fb", "borda": "#dde3ea",
        "campo": "#ffffff", "texto": "#1f2937", "suave": "#6b7280", "texto_botao": "#ffffff",
        "primaria": "#0f766e", "hover": "#115e59", "selecao": "#ccfbf1", "selecao_hover": "#99f6e4", "texto_selecao": "#ffffff",
        "sucesso": "#15803d", "perigo": "#b91c1c", "alerta": "#c2410c",
        "terminal": "#0f172a", "terminal_texto": "#5eead4",
    },
    "Grafite Âmbar": {
        "modo": "Dark",
        "fundo": "#161412", "superficie": "#1f1c19", "cartao": "#2a2622", "borda": "#3d362f",
        "campo": "#1a1815", "texto": "#f2ede6", "suave": "#a39a8e", "texto_botao": "#1a1206",
        "primaria": "#f59e0b", "hover": "#d97706", "selecao": "#b45309", "selecao_hover": "#92400e", "texto_selecao": "#1a1206",
        "sucesso": "#84cc16", "perigo": "#ef4444", "alerta": "#fb923c",
        "terminal": "#0c0b0a", "terminal_texto": "#fcd34d",
    },
}

TEMA_PADRAO = "Noite Azul"


def _par(cor):
    """O customtkinter guarda [cor no modo claro, cor no modo escuro]; usamos a mesma nos dois."""
    return [cor, cor]


def _montar_json(p):
    base = os.path.join(os.path.dirname(ctk.__file__), "assets", "themes", "blue.json")
    with open(base, encoding="utf-8") as f:
        t = json.load(f)

    t["CTk"]["fg_color"] = _par(p["fundo"])
    t["CTkToplevel"]["fg_color"] = _par(p["fundo"])
    t["CTkFrame"].update(fg_color=_par(p["superficie"]), top_fg_color=_par(p["cartao"]),
                         border_color=_par(p["borda"]))
    t["CTkButton"].update(fg_color=_par(p["primaria"]), hover_color=_par(p["hover"]),
                          border_color=_par(p["borda"]), text_color=_par(p["texto_botao"]),
                          text_color_disabled=_par(p["suave"]))
    t["CTkLabel"]["text_color"] = _par(p["texto"])
    t["CTkEntry"].update(fg_color=_par(p["campo"]), border_color=_par(p["borda"]),
                         text_color=_par(p["texto"]), placeholder_text_color=_par(p["suave"]))
    t["CTkCheckBox"].update(fg_color=_par(p["primaria"]), border_color=_par(p["suave"]),
                            hover_color=_par(p["hover"]), checkmark_color=_par(p["texto_botao"]),
                            text_color=_par(p["texto"]), text_color_disabled=_par(p["suave"]))
    t["CTkProgressBar"].update(fg_color=_par(p["borda"]), progress_color=_par(p["primaria"]),
                               border_color=_par(p["borda"]))
    t["CTkOptionMenu"].update(fg_color=_par(p["campo"]), button_color=_par(p["borda"]),
                              button_hover_color=_par(p["suave"]), text_color=_par(p["texto"]),
                              text_color_disabled=_par(p["suave"]))
    t["CTkSegmentedButton"].update(fg_color=_par(p["borda"]), selected_color=_par(p["selecao"]),
                                   selected_hover_color=_par(p["selecao_hover"]),
                                   unselected_color=_par(p["campo"]),
                                   unselected_hover_color=_par(p["cartao"]),
                                   text_color=_par(p["texto"]), text_color_disabled=_par(p["suave"]))
    t["CTkScrollbar"].update(button_color=_par(p["borda"]), button_hover_color=_par(p["suave"]))
    t["CTkTextbox"].update(fg_color=_par(p["campo"]), border_color=_par(p["borda"]),
                           text_color=_par(p["texto"]), scrollbar_button_color=_par(p["borda"]),
                           scrollbar_button_hover_color=_par(p["suave"]))
    t["CTkScrollableFrame"]["label_fg_color"] = _par(p["cartao"])
    t["DropdownMenu"].update(fg_color=_par(p["cartao"]), hover_color=_par(p["borda"]),
                             text_color=_par(p["texto"]))
    return t


def aplicar(nome=None):
    """Ativa o tema (antes de criar a janela) e devolve a paleta."""
    nome = nome if nome in TEMAS else TEMA_PADRAO
    paleta = dict(TEMAS[nome], nome=nome)
    caminho = os.path.join(tempfile.gettempdir(), f"agrolibs_tema_{abs(hash(nome))}.json")
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(_montar_json(paleta), f)
    ctk.set_appearance_mode(paleta["modo"])
    ctk.set_default_color_theme(caminho)
    return paleta
