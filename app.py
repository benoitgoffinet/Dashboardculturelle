print("APP START")
from dash import dcc, html, Input, Output, State, dash_table, ctx
import dash
import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import os

from data import (
    DF,
    VARIABLES_CATEGORIELLES,
    VARIABLES_NUMERIQUES
)
print("DATA IMPORT OK")



from model import (
    predire_affluence,
    importance_df,
    MAE,
    R2
)
print("MODEL IMPORT OK")

#INIT


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Dashboard",
    suppress_callback_exceptions=True,
)

print("SERVER CREATED")
server = app.server



#PALETTE


# Charte graphique du site
C = {
    "bg": "#0b1021",
    "page_start": "#e2e8f0",
    "page_mid": "#dbeafe",
    "card": "#f8fafc",
    "border": "#cbd5e1",
    "text": "#0f172a",
    "sub": "#475569",
    "primary": "#f59e0b",
    "primary_dark": "#d97706",
    "hero_dark": "#080d1c",
    "hero_mid": "#172554",
    "hero_blue": "#1e3a8a",
    "blue": "#3b82f6",
    "green": "#22c55e",
    "red": "#ef4444",
    "orange": "#f59e0b",
    "purple": "#8b5cf6",
    "pink": "#ec4899",

    # Compatibilité avec le code existant
    "surface": "#f8fafc",
    "muted": "#475569",
}

PALETTE = [C["primary"], C["primary_dark"], C["hero_blue"],C["blue"], C["hero_mid"], C["sub"]]

def theme_fig(fig, height=320):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text"], size=12),
        height=height,
        margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font_size=11
        ),
        xaxis=dict(
            gridcolor="rgba(203,213,225,0.95)",
            linecolor=C["border"],
            tickfont_color=C["sub"]
        ),
        yaxis=dict(
            gridcolor="rgba(203,213,225,0.95)",
            linecolor=C["border"],
            tickfont_color=C["sub"]
        ),
    )
    return fig


def card(children, className="", **kwargs):return html.Div(children,className=f"dc-card {className}".strip(),style={"backgroundColor": C["card"],"border": f"1px solid {C['border']}","borderRadius": "12px","padding": "18px",**kwargs})

def kpi(titre, valeur, couleur, icone, sous_titre=""):return card(html.Div([html.Div([html.Span(icone, style={"fontSize": "26px"}),html.Span(titre, style={"color": C["sub"],"fontSize": "13px","marginLeft": "8px"})]),html.Div(valeur,style={"color": couleur, "fontSize": "28px","fontWeight": "700", "margin": "8px 0 2px"}),html.Div(sous_titre,style={"color": C["sub"], "fontSize": "12px"})]))


#OPTIONS DE FILTRES


def opts(col):
    vals = sorted(DF[col].unique())
    return [{"label": str(v), "value": v} for v in vals]

CRITERES_ANALYSE = {
    "genre": "Genre",
    "jour": "Jour",
    "saison": "Saison",
    "tranche_horaire": "Horaire",
    "meteo": "Météo",
    "tranche_prix": "Prix par tranche",

}

DROPDOWN_STYLE = {
    "backgroundColor": C["card"],
    "color": C["text"],
}


#LAYOUT

print("LAYOUT START")
app.layout = html.Div(className="app-shell",style={"minHeight": "100vh","fontFamily": "'Inter', 'Segoe UI', sans-serif","padding": "16px 24px"},children=[

    # ── HEADER ──────────────────────────────────────────
    html.Div([
        html.Div([
             html.A(
                [
                    html.Span("BenIA", className="logo-main"),
                    html.Span(".Solutions", className="logo-accent")
                ],
                className="logo",
                href="https://www.benia.solutions/",
                target ='blank',
            ),
            html.Div([
                html.H1("Dashboard de présentation", className="header-title"),
                html.P("Analyse culturelle & prédiction d'affluence", className="header-subtitle")
            ], className="header-copy")
        ], className="header-brand"),
        html.Div([
             html.Div([
                html.Span("🎯", className="metric-icon"),
                html.Div([
                    html.Span("MAE", className="metric-label"),
                    html.Span(f"±{MAE} spectateurs", className="metric-value mae")
                ], className="metric-text")
            ], className="metric-chip"),
            html.Div([
                html.Span("📈", className="metric-icon"),
                html.Div([
                    html.Span("R²", className="metric-label"),
                    html.Span(f"{R2}", className="metric-value r2")
                ], className="metric-text")
            ], className="metric-chip")
        ], className="header-metrics")
    ], className="app-header"),

    # ── ONGLETS ──────────────────────────────────────────
    dcc.Tabs(
        id="onglets",
        value="analyse",
        style={"marginBottom": "20px"},
        colors={"border": C["border"],
                "primary": C["primary"],
                "background": C["card"]},
        children=[
            dcc.Tab(label="📊  Analyse Globale",  value="analyse",
                    style={"color": C["sub"]},
                    selected_style={"color": C["text"],
                                    "backgroundColor": C["card"],
                                    "borderTop": f"3px solid {C['primary']}"}),
            dcc.Tab(
    label="📈 Analyse Variable",
    value="variable",
    style={"color": C["sub"]},
    selected_style={
        "color": C["text"],
        "backgroundColor": C["card"],
        "borderTop": f"3px solid {C['blue']}"
    }
),
            dcc.Tab(label="🤖  Prédicteur",       value="predict",
                    style={"color": C["sub"]},
                    selected_style={"color": C["text"],
                                    "backgroundColor": C["card"],
                                     "borderTop": f"3px solid {C['primary_dark']}"}),
            dcc.Tab(label="📋  Données",          value="data",
                    style={"color": C["sub"]},
                    selected_style={"color": C["text"],
                                    "backgroundColor": C["card"],
                                    "borderTop": f"3px solid {C['hero_blue']}"}),
        ]
    ),

    html.Div(id="contenu-onglet"),
]

)


#CONTENU DES ONGLETS


#── ONGLET 1 : ANALYSE GLOBALE ───────────────────────────────

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
    marginBottom="12px"
),

 # KPIs
html.Div(id="kpis-globaux", className="kpi-grid",
         style={"display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                "gap": "12px", "marginBottom": "16px"}),

card(
    html.Div([
        html.Label("Métrique", style={"color": C["sub"],
                                      "fontSize": "12px",
                                      "marginBottom": "8px",
                                      "display": "block"}),
        dcc.RadioItems(
            id="f-metric",
            options=[
                {"label": " Affluence", "value": "affluence"},
                {"label": " Chiffre d'affaires", "value": "chiffre_affaire"},
                {"label": " Taux remplissage (%)", "value": "taux_remplissage"},
            ],
            value="affluence",
            inline=True,
            labelStyle={"color": C["text"], "fontSize": "13px", "marginRight": "20px"},
            inputStyle={"marginRight": "5px", "marginLeft": "0px"}
        )
    ]),
    marginBottom="16px"
),


# Ligne 1 : analyse par critère (pleine largeur)

card([
html.Div(
[
html.H4(
id="titre-critere",
style={
"color": C["text"],
"margin": "0",
"fontSize": "14px"
}
),


        html.Div(
            [
                html.Label(
                    "Critère à analyser",
                    style={
                        "color": C["sub"],
                        "fontSize": "12px",
                        "display": "block",
                        "marginBottom": "4px"
                    }
                ),

                dcc.Dropdown(
                    id="f-critere",
                    options=[
                        {"label": label, "value": value}
                        for value, label in CRITERES_ANALYSE.items()
                    ],
                    value="genre",
                    clearable=False,
                    style={"minWidth": "220px"}
                ),
            ]
        ),
    ],
    style={
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "flex-start",
        "gap": "16px",
        "flexWrap": "wrap",
        "marginBottom": "8px",
    },
),

dcc.Graph(
    id="g-critere-bar",
    config={"displayModeBar": False}
),

], marginBottom="16px"),

# Ligne 2 : évolution (pleine largeur)

card([
html.H4(
id="titre-evolution",
style={
"color": C["text"],
"margin": "0 0 8px",
"fontSize": "14px"
}
),
dcc.Graph(
id="g-evolution",
config={"displayModeBar": False}
)
], marginBottom="16px"),

# Ligne 3 : importance features (pleine largeur)

card([
html.H4(
"Facteurs ayant le plus d’impact sur la fréquentation",
style={
"color": C["text"],
"margin": "0 0 8px",
"fontSize": "14px"
}
),
dcc.Graph(
id="g-importance",
config={"displayModeBar": False}
)
]),

])

#── ONGLET 2 : ANALYSE PAR VARIABLE ─────────────────────────

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
    html.Label(
        "Variable secondaire",
        style={
            "color": C["sub"],
            "fontSize": "12px",
            "display": "block",
            "marginBottom": "4px"
        }
    ),
    dcc.Dropdown(
        id="var-select2",
        options=[
            {
                "label": v.replace("_", " ").title(),
                "value": v
            }
            for v in VARIABLES_CATEGORIELLES
        ],
        value="jour",
        clearable=False,
        style={"width": "100%"}
    )
], style={"minWidth": "230px", "flex": "1"}),
        html.Div([
    html.Label("Filtre"),
    dcc.Dropdown(
        id="var-filter",
        options=[
            {
                "label": v.replace("_", " ").title(),
                "value": v
            }
            for v in VARIABLES_CATEGORIELLES
        ],
        placeholder="Aucun filtre"
    )
]),
        html.Div([
    html.Label("Valeur"),
    dcc.Dropdown(
        id="var-filter-value",
        placeholder="Choisir une valeur"
    )
]),
        html.Div([
            html.Label("Métrique",
                       style={"color": C["sub"], "fontSize": "12px",
                              "display": "block", "marginBottom": "4px"}),
            dcc.RadioItems(
id="var-metric",
options=[
    {"label": " Affluence", "value": "affluence"},
    {"label": " Taux remplissage (%)", "value": "taux_remplissage"},
    {"label": " Chiffre d'affaires", "value": "chiffre_affaire"},
],
value="affluence",
inline=True,
labelStyle={"color": C["text"], "fontSize": "13px"},
inputStyle={"marginRight": "5px", "marginLeft": "15px"}

)])], style={"display": "flex", "alignItems": "flex-end","gap": "30px", "flexWrap": "wrap"}),marginBottom="16px"),

# KPIs par valeur de la variable
html.Div(id="kpis-variable",
         style={"marginBottom": "16px"}),

#  métrique choisie par critère

card(
    [
        html.Div(
            [
                html.H4(
                    id="titre-variable",
                    style={
                        "color": C["text"],
                        "margin": "0",
                        "fontSize": "14px"
                    }
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "flex-start",
                "gap": "16px",
                "flexWrap": "wrap",
                "marginBottom": "8px",
            },
        ),

        dcc.Graph(
            id="g-variable-bar",
            config={"displayModeBar": False}
        ),
    ],
    marginBottom="16px",
),

])

#── ONGLET 3 : PRÉDICTEUR ────────────────────────────────────

layout_predict = html.Div([html.Div([

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
                        "backgroundColor": C["primary"],
                        "color": "#111827", "border": "none",
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

#── ONGLET 4 : DONNÉES ───────────────────────────────────────

layout_data = html.Div([card([html.Div([html.H4("📋 Données brutes",style={"color": C["text"], "margin": 0,"fontSize": "15px"}),html.Span(f"{len(DF)} événements",style={"color": C["sub"], "fontSize": "12px"})], style={"display": "flex", "justifyContent": "space-between","marginBottom": "12px"}),

    dash_table.DataTable(
        data=DF.sort_values("date", ascending=False).head(200).to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c}
                 for c in ["date", "genre", "jour", "saison",
                           "tranche_horaire", "meteo",
                           "prix_moyen", "capacite", "affluence",
                           "taux_remplissage", "chiffre_affaire"]],
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": C["primary"],
            "color": "#111827",
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
             "backgroundColor": "#e2e8f0"},
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



#ROUTING ONGLETS


print("LAYOUT END")
@app.callback(
    Output("contenu-onglet", "children"),
    Input("onglets", "value")
)
def afficher_onglet(onglet):
    if onglet == "analyse":
        return layout_analyse

    if onglet == "variable":
        return layout_variable

    if onglet == "predict":
        return layout_predict

    if onglet == "data":
        return layout_data



#CALLBACKS ONGLET 1 — ANALYSE GLOBALE


@app.callback(
    Output("kpis-globaux", "children"),
    Output("titre-evolution", "children"),
    Output("g-evolution", "figure"),
    Output("titre-critere", "children"),
    Output("g-critere-bar", "figure"),
    Output("g-importance", "figure"),

    Input("f-genre", "value"),
    Input("f-jour", "value"),
    Input("f-saison", "value"),
    Input("f-horaire", "value"),
    Input("f-meteo", "value"),
    Input("f-metric", "value"),
    Input("f-critere", "value"),
)

def update_analyse(
    genres,
    jours,
    saisons,
    horaires,
    meteos,
    metric,
    critere,
):
    df = DF.copy()

    metric_labels = {
        "affluence": "Affluence",
        "taux_remplissage": "Taux de remplissage",
        "chiffre_affaire": "Chiffre d'affaires",
    }

    metric_units = {
        "affluence": "",
        "taux_remplissage": " %",
        "chiffre_affaire": " €",
    }

    metric_label = metric_labels[metric]
    metric_unit = metric_units[metric]

    # Filtres
    for col, val in [
        ("genre", genres),
        ("jour", jours),
        ("saison", saisons),
        ("tranche_horaire", horaires),
        ("meteo", meteos),
    ]:
        if val:
            df = df[df[col].isin(val)]

    if df.empty:
        empty = go.Figure()
        empty.add_annotation(
            text="Aucune donnée",
            showarrow=False,
            font=dict(color=C["text"])
        )
        empty = theme_fig(empty)

        return (
    [],
    "📈 Évolution",
    empty,
    "📊 Analyse",
    empty,
    empty,
)

    # KPIs
    kpis = [
        kpi("Événements", f"{len(df):,}", C["blue"], "🎭"),
        kpi(
            "Affluence Moy.",
            f"{df['affluence'].mean():.0f}",
            C["blue"],
            "👥",
            sous_titre=f"Max : {df['affluence'].max():,.0f}",
        ),
        kpi(
            "Taux Remplissage",
            f"{df['taux_remplissage'].mean():.1f} %",
            C["green"],
            "📊",
        ),
        kpi(
            "CA Total",
            f"{df['chiffre_affaire'].sum():,.0f} €",
            C["primary"],
            "💶",
        ),
    ]

    # Evolution
    ev = (
        df.groupby("mois")
        .agg(metric_moy=(metric, "mean"))
        .reset_index()
    )

    fig_ev = px.line(
        ev,
        x="mois",
        y="metric_moy",
        markers=True,
        labels={
        "metric_moy": f"{metric_label} moy",
        "mois": "Mois"
    }
    )
    fig_ev = theme_fig(fig_ev)

    # Donut genre
    g_genre = (
        df.groupby("genre")[metric]
        .mean()
        .reset_index()
    )

    fig_donut = go.Figure(
        go.Pie(
            labels=g_genre["genre"],
            values=g_genre[metric],
            hole=0.5,
        )
    )
    fig_donut = theme_fig(fig_donut)
    if critere == "tranche_prix":

      prix_min = np.floor(df["prix_moyen"].min() / 5) * 5
      prix_max = np.ceil(df["prix_moyen"].max() / 5) * 5

      bins = np.arange(prix_min, prix_max + 10, 5)

      labels = [
        f"{int(bins[i])}-{int(bins[i+1])} €"
        for i in range(len(bins) - 1)
    ]

      df = df.copy()

      df["tranche_prix"] = pd.cut(
        df["prix_moyen"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

      if df["tranche_prix"].isna().any():
        df.loc[
            df["tranche_prix"].isna(),
            "tranche_prix"
        ] = labels[-1]
    # Critère sélectionné
    grp = (
        df.groupby(critere)[metric]
        .mean()
        .reset_index()
    )

    fig_critere = px.bar(
        grp,
        x=critere,
        y=metric,
    )
    fig_critere = theme_fig(fig_critere)

    # Importance
    fig_imp = go.Figure(
        go.Bar(
            x=importance_df["importance"],
            y=importance_df["variable"],
            orientation="h",
        )
    )
    fig_imp.update_layout(
        yaxis=dict(autorange="reversed")
    )
    fig_imp = theme_fig(fig_imp)

    return (
    kpis,
    f"📈 Évolution de {metric_label}",
    fig_ev,
    f"📊 {metric_label} par {CRITERES_ANALYSE.get(critere, critere)}",
    fig_critere,
    fig_imp,
)

#CALLBACKS ONGLET 2 — ANALYSE PAR VARIABLE

@app.callback(
    Output("var-filter-value", "options"),
    Input("var-filter", "value"),
)
def update_filter_values(col):

    if not col:
        return []

    vals = sorted(DF[col].dropna().unique())

    return [
        {"label": str(v), "value": v}
        for v in vals
    ]


@app.callback(
    Output("var-select", "options"),
    Output("var-select2", "options"),
    Input("var-select", "value"),
    Input("var-select2", "value"),
)
def update_variable_options(var1, var2):

    opts_var1 = [
        {
            "label": v.replace("_", " ").title(),
            "value": v
        }
        for v in VARIABLES_CATEGORIELLES
        if v != var2
    ]

    opts_var2 = [
        {
            "label": v.replace("_", " ").title(),
            "value": v
        }
        for v in VARIABLES_CATEGORIELLES
        if v != var1
    ]

    return opts_var1, opts_var2



@app.callback(
    Output("kpis-variable", "children"),
    Output("titre-variable", "children"),
    Output("g-variable-bar", "figure"),
    Input("var-select", "value"),
    Input("var-select2", "value"),
    Input("var-metric", "value"),
    Input("var-filter", "value"),
    Input("var-filter-value", "value"),
)


def update_variable(
    var,
    var2,
    metric,
    filter_col,
    filter_val,
):

    df = DF.copy()

    # Application du filtre
    if filter_col and filter_val:
        df = df[df[filter_col] == filter_val]

    # Agrégation croisée
    agg = (
        df.groupby([var, var2])[metric]
        .mean()
        .reset_index()
    )

    # Graphique
    fig = px.bar(
        agg,
        x=var,
        y=metric,
        color=var2,
        barmode="group"
    )

    fig.update_layout(
        legend_title=var2.replace("_", " ").title()
    )

    fig = theme_fig(fig)

    # Titre
    titre = (
        f"{metric.replace('_', ' ').title()} "
        f"par {var.replace('_', ' ').title()}"
    )

    if filter_col and filter_val:
        titre += (
            f" | {filter_col.replace('_', ' ').title()} : "
            f"{filter_val}"
        )

    return (
        html.Div(),
        titre,
        fig
    )



#CALLBACKS ONGLET 3 — PRÉDICTEUR


@app.callback(
    Output("result-kpis", "children"),
    Output("g-jauge", "figure"),
    Output("g-radar", "figure"),

    Input("btn-predict", "n_clicks"),

    State("p-genre", "value"),
    State("p-jour", "value"),
    State("p-saison", "value"),
    State("p-horaire", "value"),
    State("p-meteo", "value"),
    State("p-prix", "value"),
    State("p-capacite", "value"),
    State("p-note", "value"),
    State("p-critiques", "value"),
    State("p-promo", "value"),

    prevent_initial_call=True,
)
def predict(
    n,
    genre,
    jour,
    saison,
    horaire,
    meteo,
    prix,
    capacite,
    note,
    critiques,
    promo,
):
    # suite de la fonction...

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
        "bgcolor": "#e2e8f0",
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
     fillcolor="rgba(245,158,11,0.2)"
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
        bgcolor="#e2e8f0",
        radialaxis=dict(visible=True, range=[0, 100],
                        gridcolor=C["border"],
                        tickfont_color=C["sub"]),
        angularaxis=dict(gridcolor=C["border"],
                         tickfont_color=C["text"])
    )
)
    fig_radar = theme_fig(fig_radar, height=320)

    return kpis_res, fig_jauge, fig_radar



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port)
