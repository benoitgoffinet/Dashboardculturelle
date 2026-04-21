# app.py
import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from data import DF, VARIABLES_CATEGORIELLES, VARIABLES_NUMERIQUES
from model import predire_affluence, importance_df, MAE, R2

# ============================================================
# INIT
# ============================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="Dashboard",
    suppress_callback_exceptions=True,
)
server = app.server

# ============================================================
# PALETTE
# ============================================================
C = {
    "bg":      "#0d1117",
    "card":    "#161b22",
    "border":  "#30363d",
    "purple":  "#8b5cf6",
    "pink":    "#ec4899",
    "blue":    "#3b82f6",
    "green":   "#22c55e",
    "orange":  "#f59e0b",
    "red":     "#ef4444",
    "text":    "#e6edf3",
    "sub":     "#8b949e",
}

PALETTE = [C["purple"], C["pink"], C["blue"],
           C["green"], C["orange"], C["red"]]

def theme_fig(fig, height=320):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text"], size=12),
        height=height,
        margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d",
                   tickfont_color=C["sub"]),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d",
                   tickfont_color=C["sub"]),
    )
    return fig

# ─── Composants réutilisables ────────────────────────────────
def card(children, className="", **kwargs):
    return html.Div(
        children,
        className=f"dc-card {className}".strip(),
        style={
            "backgroundColor": C["card"],
            "border": f"1px solid {C['border']}",
            "borderRadius": "12px",
            "padding": "18px",
            **kwargs
        }
    )

def kpi(titre, valeur, couleur, icone, sous_titre=""):
    return card(
        html.Div([
            html.Div([
                html.Span(icone, style={"fontSize": "26px"}),
                html.Span(titre, style={"color": C["sub"],
                                        "fontSize": "13px",
                                        "marginLeft": "8px"})
            ]),
            html.Div(valeur,
                     style={"color": couleur, "fontSize": "28px",
                            "fontWeight": "700", "margin": "8px 0 2px"}),
            html.Div(sous_titre,
                     style={"color": C["sub"], "fontSize": "12px"})
        ])
    )

# ============================================================
# OPTIONS DE FILTRES
# ============================================================
def opts(col):
    vals = sorted(DF[col].unique())
    return [{"label": str(v), "value": v} for v in vals]

DROPDOWN_STYLE = {"backgroundColor": C["card"], "color": C["text"]}

# ============================================================
# LAYOUT
# ============================================================
app.layout = html.Div(
    className="app-shell",
    style={"backgroundColor": C["bg"], "minHeight": "100vh",
           "fontFamily": "'Inter', 'Segoe UI', sans-serif",
           "padding": "16px 24px"},
    children=[

        # ── HEADER ──────────────────────────────────────────
        html.Div([
            html.Div([
                html.H1("🎭 Organisme fictif",
                        style={"color": C["text"], "margin": 0,
                               "fontSize": "24px", "fontWeight": "700"}),
                html.P("Dashboard prédictif d'affluence",
                       style={"color": C["sub"], "margin": "4px 0 0",
                              "fontSize": "13px"})
            ], style={"minWidth": "230px", "flex": "1"}),
            html.Div([
                html.Span(f"MAE : ±{MAE} spectateurs",
                          style={"color": C["orange"], "fontSize": "13px",
                                 "marginRight": "20px"}),
                html.Span(f"R² : {R2}",
                          style={"color": C["green"], "fontSize": "13px"})
            ])
        ], className="app-header", style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center",
                  "borderBottom": f"1px solid {C['border']}",
                  "paddingBottom": "14px", "marginBottom": "20px"}),

        # ── ONGLETS ──────────────────────────────────────────
        dcc.Tabs(
            id="onglets",
            value="analyse",
            style={"marginBottom": "20px"},
            colors={"border": C["border"],
                    "primary": C["purple"],
                    "background": C["card"]},
            children=[
                dcc.Tab(label="📊  Analyse Globale",  value="analyse",
                        style={"color": C["sub"]},
                        selected_style={"color": C["text"],
                                        "backgroundColor": C["card"],
                                        "borderTop": f"3px solid {C['purple']}"}),
                dcc.Tab(label="🔍  Analyse par Variable", value="variable",
                        style={"color": C["sub"]},
                        selected_style={"color": C["text"],
                                        "backgroundColor": C["card"],
                                        "borderTop": f"3px solid {C['pink']}"}),
                dcc.Tab(label="🤖  Prédicteur",       value="predict",
                        style={"color": C["sub"]},
                        selected_style={"color": C["text"],
                                        "backgroundColor": C["card"],
                                        "borderTop": f"3px solid {C['blue']}"}),
                dcc.Tab(label="📋  Données",          value="data",
                        style={"color": C["sub"]},
                        selected_style={"color": C["text"],
                                        "backgroundColor": C["card"],
                                        "borderTop": f"3px solid {C['green']}"}),
            ]
        ),

        html.Div(id="contenu-onglet"),
    ]
)

# ============================================================
# CONTENU DES ONGLETS
# ============================================================

# ── ONGLET 1 : ANALYSE GLOBALE ───────────────────────────────
layout_analyse = html.Div([

    # Filtres
    card(
        html.Div([
            html.Span("🔎 Filtres globaux",
                      style={"color": C["text"], "fontWeight": "600",
                             "marginRight": "20px"}),
            *[
                html.Div([
                    html.Label(lbl, style={"color": C["sub"],
                                           "fontSize": "12px",
                                           "marginBottom": "4px",
                                           "display": "block"}),
                    dcc.Dropdown(id=fid, options=opts(col),
                                 multi=True, placeholder=f"Tous",
                                 style={"minWidth": "160px"})
                ])
                for lbl, fid, col in [
                    ("Genre",       "f-genre",   "genre"),
                    ("Jour",        "f-jour",    "jour"),
                    ("Saison",      "f-saison",  "saison"),
                    ("Horaire",     "f-horaire", "tranche_horaire"),
                    ("Météo",       "f-meteo",   "meteo"),
                ]
            ]
        ], style={"display": "flex", "alignItems": "flex-end",
                  "flexWrap": "wrap", "gap": "16px"}),
        marginBottom="16px"
    ),

    # KPIs
    html.Div(id="kpis-globaux", className="kpi-grid",
             style={"display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                    "gap": "12px", "marginBottom": "16px"}),

    # Ligne 1 : évolution + donut
    html.Div([
        html.Div([
            card([html.H4("📈 Évolution de l'affluence",
                          style={"color": C["text"], "margin": "0 0 8px",
                                 "fontSize": "14px"}),
                  dcc.Graph(id="g-evolution",
                            config={"displayModeBar": False})
                  ])
        ], style={"flex": "2", "minWidth": "320px"}),
        html.Div([
            card([html.H4("🎭 Affluence par Genre",
                          style={"color": C["text"], "margin": "0 0 8px",
                                 "fontSize": "14px"}),
                  dcc.Graph(id="g-genre-donut",
                            config={"displayModeBar": False})
                  ])
        ], style={"flex": "1", "minWidth": "280px"}),
    ], className="split-row", style={"display": "flex", "gap": "12px", "marginBottom": "16px", "flexWrap": "wrap"}),

    # Ligne 2 : heatmap + boxplot
    html.Div([
        card([html.H4("🗓️ Heatmap Jour × Saison",
                      style={"color": C["text"], "margin": "0 0 8px",
                             "fontSize": "14px"}),
              dcc.Graph(id="g-heatmap",
                        config={"displayModeBar": False})
              ], **{"flex": "1"}),
        card([html.H4("📦 Distribution Taux de Remplissage",
                      style={"color": C["text"], "margin": "0 0 8px",
                             "fontSize": "14px"}),
              dcc.Graph(id="g-box",
                        config={"displayModeBar": False})
              ], **{"flex": "1"}),
    ], className="split-row", style={"display": "flex", "gap": "12px", "marginBottom": "16px", "flexWrap": "wrap"}),
])
    

# ── ONGLET 2 : ANALYSE PAR VARIABLE ─────────────────────────
layout_variable = html.Div([

    card(
        html.Div([
            html.Div([
                html.Label("Variable à analyser",
                           style={"color": C["sub"], "fontSize": "12px",
                                  "display": "block", "marginBottom": "4px"}),
                dcc.Dropdown(
                    id="var-select",
                    options=[{"label": v.replace("_", " ").title(), "value": v}
                             for v in VARIABLES_CATEGORIELLES],
                    value="genre",
                    clearable=False,
                    style={"width": "100%"}
                )
            ], style={"minWidth": "230px", "flex": "1"}),
            html.Div([
                html.Label("Métrique",
                           style={"color": C["sub"], "fontSize": "12px",
                                  "display": "block", "marginBottom": "4px"}),
                dcc.RadioItems(
    id="var-metric",
    options=[
        {"label": " Affluence moyenne", "value": "affluence"},
        {"label": " Taux remplissage (%)", "value": "taux_remplissage"},
        {"label": " Chiffre d'affaires", "value": "chiffre_affaire"},
    ],
    value="affluence",
    inline=True,
    labelStyle={"color": "white", "fontSize": "13px"},
    inputStyle={"marginRight": "5px", "marginLeft": "15px"}
)
            ])
        ], style={"display": "flex", "alignItems": "flex-end",
                  "gap": "30px", "flexWrap": "wrap"}),
        marginBottom="16px"
    ),

    # KPIs par valeur de la variable
    html.Div(id="kpis-variable",
             style={"marginBottom": "16px"}),

    # Graphiques
    html.Div([
        card([
            html.H4(id="titre-bar-var",
                    style={"color": C["text"], "margin": "0 0 8px",
                           "fontSize": "14px"}),
            dcc.Graph(id="g-bar-variable",
                      config={"displayModeBar": False})
        ], flex="1", minWidth="320px"),
        card([
            html.H4("📦 Distribution par modalité",
                    style={"color": C["text"], "margin": "0 0 8px",
                           "fontSize": "14px"}),
            dcc.Graph(id="g-violin-variable",
                      config={"displayModeBar": False})
        ], flex="1", minWidth="320px"),
    ], className="split-row", style={"display": "flex", "gap": "12px", "marginBottom": "16px", "flexWrap": "wrap"}),

    # Croisement avec une 2ème variable
    card([
        html.Div([
            html.H4("🔀 Croisement avec une 2ème variable",
                    style={"color": C["text"], "margin": "0",
                           "fontSize": "14px"}),
            dcc.Dropdown(
                id="var-select2",
                options=[{"label": v.replace("_", " ").title(), "value": v}
                         for v in VARIABLES_CATEGORIELLES],
                value="jour",
                clearable=False,
                style={"width": "100%", "minWidth": "180px", "maxWidth": "240px"}
            )
        ], className="cross-header", style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "marginBottom": "10px"}),
        dcc.Graph(id="g-heatmap-variable",
                  config={"displayModeBar": False})
    ]),
])

# ── ONGLET 3 : PRÉDICTEUR ────────────────────────────────────
layout_predict = html.Div([
    html.Div([

        # Formulaire
        card([
            html.H4("🎛️ Paramètres de l'événement",
                    style={"color": C["text"], "margin": "0 0 16px",
                           "fontSize": "16px", "fontWeight": "700"}),

            *[
                html.Div([
                    html.Label(lbl, style={"color": C["sub"],
                                           "fontSize": "12px",
                                           "display": "block",
                                           "marginBottom": "4px"}),
                    dcc.Dropdown(
                        id=did,
                        options=[{"label": v, "value": v}
                                 for v in sorted(DF[col].unique())],
                        value=sorted(DF[col].unique())[0],
                        clearable=False,
                    )
                ], style={"marginBottom": "14px"})
                for lbl, did, col in [
                    ("Genre",       "p-genre",    "genre"),
                    ("Jour",        "p-jour",     "jour"),
                    ("Saison",      "p-saison",   "saison"),
                    ("Horaire",     "p-horaire",  "tranche_horaire"),
                    ("Météo",       "p-meteo",    "meteo"),
                ]
            ],

            html.Div([
                html.Label("Prix moyen (€)",
                           style={"color": C["sub"], "fontSize": "12px"}),
                dcc.Slider(id="p-prix", min=5, max=100, step=5, value=30,
                           marks={i: {"label": f"{i}€",
                                      "style": {"color": C["sub"],
                                                "fontSize": "10px"}}
                                  for i in [5, 25, 50, 75, 100]},
                           tooltip={"placement": "bottom"})
            ], style={"marginBottom": "18px"}),

            html.Div([
                html.Label("Capacité de la salle",
                           style={"color": C["sub"], "fontSize": "12px"}),
                dcc.Slider(id="p-capacite", min=50, max=800, step=50, value=300,
                           marks={i: {"label": str(i),
                                      "style": {"color": C["sub"],
                                                "fontSize": "10px"}}
                                  for i in [50, 200, 400, 600, 800]},
                           tooltip={"placement": "bottom"})
            ], style={"marginBottom": "18px"}),

            html.Div([
                html.Label("Note moyenne (1-5)",
                           style={"color": C["sub"], "fontSize": "12px"}),
                dcc.Slider(id="p-note", min=1, max=5, step=0.1, value=3.5,
                           marks={i: {"label": str(i),
                                      "style": {"color": C["sub"]}}
                                  for i in [1, 2, 3, 4, 5]},
                           tooltip={"placement": "bottom"})
            ], style={"marginBottom": "18px"}),

            html.Div([
                html.Label("Nombre de critiques",
                           style={"color": C["sub"], "fontSize": "12px"}),
                dcc.Slider(id="p-critiques", min=0, max=20, step=1, value=5,
                           tooltip={"placement": "bottom"})
            ], style={"marginBottom": "18px"}),

            html.Div([
                html.Label("Semaine promotionnelle",
                           style={"color": C["sub"], "fontSize": "12px"}),
                dcc.RadioItems(
                    id="p-promo",
                    options=[{"label": " Oui", "value": 1},
                             {"label": " Non", "value": 0}],
                    value=0, inline=True,
                    style={"color": C["text"], "fontSize": "13px"},
                    inputStyle={"marginRight": "5px", "marginLeft": "15px"}
                )
            ], style={"marginBottom": "20px"}),

            html.Button("🔮 Prédire l'affluence", id="btn-predict",
                        style={
                            "width": "100%", "padding": "12px",
                            "backgroundColor": C["purple"],
                            "color": "white", "border": "none",
                            "borderRadius": "8px", "fontSize": "15px",
                            "fontWeight": "700", "cursor": "pointer"
                        })
        ], flex="1", minWidth="320px"),

        # Résultats
        html.Div([
            html.Div(id="result-kpis", className="kpi-grid",
                     style={"display": "grid",
                            "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                            "gap": "12px", "marginBottom": "16px"}),
            card([
                html.H4("📊 Jauge de remplissage",
                        style={"color": C["text"], "margin": "0 0 8px",
                               "fontSize": "14px"}),
                dcc.Graph(id="g-jauge",
                          config={"displayModeBar": False})
            ], marginBottom="16px"),
            card([
                html.H4("🔀 Comparaison : votre événement vs moyenne",
                        style={"color": C["text"], "margin": "0 0 8px",
                               "fontSize": "14px"}),
                dcc.Graph(id="g-radar",
                          config={"displayModeBar": False})
            ]),
        ], style={"flex": "1.4", "minWidth": "320px"}),

    ], className="predict-layout", style={"display": "flex", "gap": "16px", "alignItems": "flex-start", "flexWrap": "wrap"}),
])

# ── ONGLET 4 : DONNÉES ───────────────────────────────────────
layout_data = html.Div([
    card([
        html.Div([
            html.H4("📋 Données brutes",
                    style={"color": C["text"], "margin": 0,
                           "fontSize": "15px"}),
            html.Span(f"{len(DF)} événements",
                      style={"color": C["sub"], "fontSize": "12px"})
        ], style={"display": "flex", "justifyContent": "space-between",
                  "marginBottom": "12px"}),

        dash_table.DataTable(
            data=DF.sort_values("date", ascending=False).head(200).to_dict("records"),
            columns=[{"name": c.replace("_", " ").title(), "id": c}
                     for c in ["date", "genre", "jour", "saison",
                               "tranche_horaire", "meteo",
                               "prix_moyen", "capacite", "affluence",
                               "taux_remplissage", "chiffre_affaire"]],
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": C["purple"],
                "color": "white",
                "fontWeight": "bold",
                "border": "none",
                "fontSize": "12px"
            },
            style_cell={
                "backgroundColor": C["card"],
                "color": C["text"],
                "border": f"1px solid {C['border']}",
                "padding": "8px 12px",
                "fontSize": "12px"
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"},
                 "backgroundColor": C["bg"]},
                {"if": {"filter_query": "{taux_remplissage} >= 80"},
                 "color": C["green"]},
                {"if": {"filter_query": "{taux_remplissage} < 50"},
                 "color": C["red"]},
            ],
            page_size=15,
            sort_action="native",
            filter_action="native",
        )
    ])
])

# ============================================================
# ROUTING ONGLETS
# ============================================================
@app.callback(Output("contenu-onglet", "children"),
              Input("onglets", "value"))
def afficher_onglet(onglet):
    if onglet == "analyse":  return layout_analyse
    if onglet == "variable": return layout_variable
    if onglet == "predict":  return layout_predict
    if onglet == "data":     return layout_data

# ============================================================
# CALLBACKS ONGLET 1 — ANALYSE GLOBALE
# ============================================================
@app.callback(
    Output("kpis-globaux",     "children"),
    Output("g-evolution",      "figure"),
    Output("g-genre-donut",    "figure"),
    Output("g-heatmap",        "figure"),
    Output("g-box",            "figure"),
    Output("g-importance",     "figure"),
    Input("f-genre",     "value"),
    Input("f-jour",      "value"),
    Input("f-saison",    "value"),
    Input("f-horaire",   "value"),
    Input("f-meteo",     "value"),
)
def update_analyse(genres, jours, saisons, horaires, meteos):
    df = DF.copy()

    # Filtres
    for col, val in [("genre", genres), ("jour", jours), ("saison", saisons),
                     ("tranche_horaire", horaires),
                     ("meteo", meteos)]:
        if val:
            df = df[df[col].isin(val)]

    if df.empty:
        empty = go.Figure()
        empty.add_annotation(text="Aucune donnée", showarrow=False,
                             font=dict(color=C["text"]))
        empty = theme_fig(empty)
        return [], empty, empty, empty, empty, empty

    # ── KPIs ──────────────────────────────────────────────────
    kpis = [
        kpi("Événements",        f"{len(df):,}",
            C["purple"], "🎭"),
        kpi("Affluence Moy.",    f"{df['affluence'].mean():.0f}",
            C["blue"],   "👥",
            sous_titre=f"Max : {df['affluence'].max():,}"),
        kpi("Taux Remplissage",  f"{df['taux_remplissage'].mean():.1f} %",
            C["green"],  "📊",
            sous_titre=f"Min : {df['taux_remplissage'].min():.1f}%"),
        kpi("CA Total",          f"{df['chiffre_affaire'].sum():,.0f} €",
            C["orange"], "💶"),
        kpi("CA Moyen/Event",    f"{df['chiffre_affaire'].mean():,.0f} €",
            C["pink"],   "💰"),
    ]

    # ── Évolution ─────────────────────────────────────────────
    ev = (df.groupby("mois")
            .agg(affluence_moy=("affluence", "mean"),
                 nb_events=("affluence", "count"))
            .reset_index())
    # Trier par date
    ev["sort_key"] = pd.to_datetime(ev["mois"], format="%b %Y")
    ev = ev.sort_values("sort_key")

    fig_ev = go.Figure()
    fig_ev.add_trace(go.Scatter(
        x=ev["mois"], y=ev["affluence_moy"],
        mode="lines+markers",
        name="Affluence moy.",
        line=dict(color=C["purple"], width=2.5),
        marker=dict(size=7),
        fill="tozeroy",
        fillcolor="rgba(139,92,246,0.12)"
    ))
    fig_ev.add_trace(go.Bar(
        x=ev["mois"], y=ev["nb_events"],
        name="Nb événements",
        marker_color=C["blue"],
        opacity=0.4,
        yaxis="y2"
    ))
    fig_ev.update_layout(
        yaxis2=dict(overlaying="y", side="right",
                    gridcolor="rgba(0,0,0,0)",
                    tickfont_color=C["sub"])
    )
    fig_ev = theme_fig(fig_ev)

    # ── Donut Genre ───────────────────────────────────────────
    g_genre = (df.groupby("genre")["affluence"]
                 .mean()
                 .reset_index()
                 .sort_values("affluence", ascending=False))
    fig_donut = go.Figure(go.Pie(
        labels=g_genre["genre"],
        values=g_genre["affluence"].round(0),
        hole=0.5,
        marker_colors=PALETTE,
        textfont_size=11,
    ))
    fig_donut.update_layout(
        annotations=[dict(text="Affluence<br>moy.", x=0.5, y=0.5,
                          font_size=11, showarrow=False,
                          font_color=C["sub"])]
    )
    fig_donut = theme_fig(fig_donut)

    # ── Heatmap Jour × Saison ─────────────────────────────────
    heat = (df.groupby(["jour", "saison"])["taux_remplissage"]
              .mean()
              .reset_index()
              .pivot(index="jour", columns="saison",
                     values="taux_remplissage"))

    # Ordonner les jours
    ordre_jours = ["Lundi", "Mardi", "Mercredi", "Jeudi",
                   "Vendredi", "Samedi", "Dimanche"]
    heat = heat.reindex([j for j in ordre_jours if j in heat.index])

    fig_heat = go.Figure(go.Heatmap(
        z=heat.values,
        x=heat.columns.tolist(),
        y=heat.index.tolist(),
        colorscale=[[0, "#0d1117"], [0.5, C["purple"]], [1, C["pink"]]],
        text=heat.values.round(1),
        texttemplate="%{text}%",
        colorbar=dict(tickfont_color=C["sub"])
    ))
    fig_heat = theme_fig(fig_heat)

    # ── Boxplot Taux Remplissage ──────────────────────────────
    fig_box = go.Figure()
    for i, genre in enumerate(df["genre"].unique()):
        sub = df[df["genre"] == genre]
        fig_box.add_trace(go.Box(
            y=sub["taux_remplissage"],
            name=genre,
            marker_color=PALETTE[i % len(PALETTE)],
            boxmean=True
        ))
    fig_box = theme_fig(fig_box)

    # ── Importance Variables ──────────────────────────────────
    fig_imp = go.Figure(go.Bar(
        x=importance_df["importance"],
        y=importance_df["variable"].str.replace("_", " ").str.title(),
        orientation="h",
        marker=dict(
            color=importance_df["importance"],
            colorscale=[[0, C["blue"]], [1, C["purple"]]]
        ),
        text=importance_df["importance"].round(3),
        textposition="outside",
        textfont_color=C["sub"]
    ))
    fig_imp.update_layout(yaxis=dict(autorange="reversed"))
    fig_imp = theme_fig(fig_imp, height=300)

    return kpis, fig_ev, fig_donut, fig_heat, fig_box, fig_imp


# ============================================================
# CALLBACKS ONGLET 2 — ANALYSE PAR VARIABLE
# ============================================================
@app.callback(
    Output("kpis-variable",     "children"),
    Output("titre-bar-var",     "children"),
    Output("g-bar-variable",    "figure"),
    Output("g-violin-variable", "figure"),
    Output("g-heatmap-variable","figure"),
    Input("var-select",  "value"),
    Input("var-metric",  "value"),
    Input("var-select2", "value"),
)
def update_variable(var, metric, var2):
    labels = {
        "affluence":       "Affluence moyenne",
        "taux_remplissage":"Taux de remplissage (%)",
        "chiffre_affaire": "Chiffre d'affaires moyen (€)"
    }
    label_metric = labels[metric]

    # ── Agrégation par modalité ───────────────────────────────
    agg = (DF.groupby(var)[metric]
             .agg(["mean", "median", "std", "count"])
             .reset_index()
             .rename(columns={"mean": "Moyenne", "median": "Médiane",
                               "std": "Écart-type", "count": "Nb événements"}))
    agg = agg.sort_values("Moyenne", ascending=False)

    # ── KPIs par modalité (cartes) ────────────────────────────
    couleurs_kpi = [C["purple"], C["pink"], C["blue"],
                    C["green"], C["orange"], C["red"]]
    kpi_cards = html.Div(
        [
            card(
                html.Div([
                    html.Div(str(row[var]),
                             style={"color": couleurs_kpi[i % len(couleurs_kpi)],
                                    "fontWeight": "700", "fontSize": "15px"}),
                    html.Div(f"Moy. : {row['Moyenne']:.1f}",
                             style={"color": C["text"], "fontSize": "22px",
                                    "fontWeight": "700", "margin": "6px 0 2px"}),
                    html.Div(f"Méd. : {row['Médiane']:.1f}",
                             style={"color": C["sub"], "fontSize": "12px"}),
                    html.Div(f"{int(row['Nb événements'])} évén.",
                             style={"color": C["sub"], "fontSize": "11px"})
                ])
            )
            for i, row in agg.iterrows()
        ],
        style={"display": "grid",
               "gridTemplateColumns": "repeat(auto-fit, minmax(170px, 1fr))",
               "gap": "10px"}
    )

    # ── Bar chart ─────────────────────────────────────────────
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=agg[var],
        y=agg["Moyenne"],
        name="Moyenne",
        marker=dict(color=PALETTE[:len(agg)]),
        error_y=dict(type="data", array=agg["Écart-type"],
                     visible=True, color=C["sub"]),
        text=agg["Moyenne"].round(1),
        textposition="outside",
        textfont_color=C["sub"]
    ))
    fig_bar.add_trace(go.Scatter(
        x=agg[var], y=agg["Médiane"],
        mode="markers", name="Médiane",
        marker=dict(symbol="diamond", size=10, color=C["orange"])
    ))
    fig_bar = theme_fig(fig_bar)

    # ── Violin ────────────────────────────────────────────────
    fig_violin = go.Figure()
    for i, val in enumerate(DF[var].unique()):
        sub = DF[DF[var] == val]
        fig_violin.add_trace(go.Violin(
            y=sub[metric], name=str(val),
            box_visible=True,
            meanline_visible=True,
            fillcolor=PALETTE[i % len(PALETTE)],
            line_color=PALETTE[i % len(PALETTE)],
            opacity=0.7
        ))
    fig_violin = theme_fig(fig_violin)

    # ── Heatmap croisée ──────────────────────────────────────
    if var != var2:
        cross = (DF.groupby([var, var2])[metric]
                   .mean()
                   .reset_index()
                   .pivot(index=var, columns=var2, values=metric))
        fig_hm = go.Figure(go.Heatmap(
            z=cross.values,
            x=[str(c) for c in cross.columns],
            y=[str(i) for i in cross.index],
            colorscale=[[0, C["bg"]], [0.5, C["purple"]], [1, C["pink"]]],
            text=cross.values.round(1),
            texttemplate="%{text}",
            colorbar=dict(tickfont_color=C["sub"])
        ))
    else:
        fig_hm = go.Figure()
        fig_hm.add_annotation(
            text="Choisir deux variables différentes",
            showarrow=False, font=dict(color=C["sub"])
        )
    fig_hm = theme_fig(fig_hm, height=350)

    titre = f"📊 {label_metric} par {var.replace('_', ' ').title()}"
    return kpi_cards, titre, fig_bar, fig_violin, fig_hm


# ============================================================
# CALLBACKS ONGLET 3 — PRÉDICTEUR
# ============================================================
@app.callback(
    Output("result-kpis", "children"),
    Output("g-jauge",     "figure"),
    Output("g-radar",     "figure"),
    Input("btn-predict",  "n_clicks"),
    State("p-genre",     "value"),
    State("p-jour",      "value"),
    State("p-saison",    "value"),
    State("p-horaire",   "value"),
    State("p-meteo",     "value"),
    State("p-prix",      "value"),
    State("p-capacite",  "value"),
    State("p-note",      "value"),
    State("p-critiques", "value"),
    State("p-promo",     "value"),
    prevent_initial_call=True,
)
def predict(n, genre, jour, saison, horaire,
             meteo, prix, capacite, note, critiques, promo):

    est_we = 1 if jour in ["Samedi", "Dimanche"] else 0

    params = {
        "genre": genre, "jour": jour, "saison": saison,
        "tranche_horaire": horaire,
        "meteo": meteo, "prix_moyen": prix, "capacite": capacite,
        "note_moyenne": note, "nb_critiques": critiques,
        "semaine_promo": promo, "est_weekend": est_we,
    }

    res = predire_affluence(params)
    aff  = res["affluence_predite"]
    taux = res["taux_remplissage"]
    ca   = res["chiffre_affaire"]
    cap  = res["capacite"]

    # Couleur selon taux
    coul = (C["green"] if taux >= 75
            else C["orange"] if taux >= 50
            else C["red"])

    # KPI résultats
    kpis_res = [
        kpi("Affluence prédite", f"{aff:,}", coul, "👥",
            sous_titre=f"/ {cap} places"),
        kpi("Taux de remplissage", f"{taux} %", coul, "📊"),
        kpi("CA estimé", f"{ca:,.0f} €", C["purple"], "💶"),
        kpi("Places libres", f"{cap - aff:,}", C["sub"], "🪑"),
    ]

    # ── Jauge ─────────────────────────────────────────────────
    fig_jauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=taux,
        title={"text": "Taux de Remplissage (%)",
               "font": {"color": C["text"], "size": 14}},
        number={"suffix": "%", "font": {"color": coul, "size": 36}},
        delta={"reference": DF["taux_remplissage"].mean(),
               "increasing": {"color": C["green"]},
               "decreasing": {"color": C["red"]}},
        gauge={
            "axis": {"range": [0, 100], "tickfont_color": C["sub"]},
            "bar": {"color": coul},
            "bgcolor": C["bg"],
            "steps": [
                {"range": [0, 50],  "color": "rgba(239,68,68,0.15)"},
                {"range": [50, 75], "color": "rgba(245,158,11,0.15)"},
                {"range": [75, 100],"color": "rgba(34,197,94,0.15)"},
            ],
            "threshold": {
                "line": {"color": C["orange"], "width": 3},
                "thickness": 0.85,
                "value": DF["taux_remplissage"].mean()
            }
        }
    ))
    fig_jauge = theme_fig(fig_jauge, height=280)

    # ── Radar comparaison ─────────────────────────────────────
    # Normalisation 0-100 pour chaque dimension
    dims = {
        "Prix attractif":   100 - (prix - 5) / (100 - 5) * 100,
        "Note":             (note - 1) / 4 * 100,
        "Critiques":        critiques / 20 * 100,
        "Promo":            promo * 100,
        "Taux prédit":      taux,
        "Week-end":         est_we * 100,
    }
    moy_df = {
        "Prix attractif":   100 - (DF["prix_moyen"].mean() - 5) / 95 * 100,
        "Note":             (DF["note_moyenne"].mean() - 1) / 4 * 100,
        "Critiques":        DF["nb_critiques"].mean() / 20 * 100,
        "Promo":            DF["semaine_promo"].mean() * 100,
        "Taux prédit":      DF["taux_remplissage"].mean(),
        "Week-end":         DF["est_weekend"].mean() * 100,
    }

    cats = list(dims.keys())

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=list(dims.values()) + [list(dims.values())[0]],
        theta=cats + [cats[0]],
        fill="toself",
        name="Votre événement",
        line_color=C["purple"],
        fillcolor="rgba(139,92,246,0.2)"
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=list(moy_df.values()) + [list(moy_df.values())[0]],
        theta=cats + [cats[0]],
        fill="toself",
        name="Moyenne dataset",
        line_color=C["orange"],
        fillcolor="rgba(245,158,11,0.15)"
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor=C["bg"],
            radialaxis=dict(visible=True, range=[0, 100],
                            gridcolor=C["border"],
                            tickfont_color=C["sub"]),
            angularaxis=dict(gridcolor=C["border"],
                             tickfont_color=C["text"])
        )
    )
    fig_radar = theme_fig(fig_radar, height=320)

    return kpis_res, fig_jauge, fig_radar


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
