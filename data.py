# data.py
import pandas as pd



# ── Chargement local du dataset ──────────────────────────────
DF = pd.read_parquet("theatre_data.parquet")    



VARIABLES_CATEGORIELLES = [
    "genre",
    "jour",
    "saison",
    "tranche_horaire",
    "meteo",
    "semaine_promo"
]

VARIABLES_NUMERIQUES = [
    "prix_moyen",
    "note_moyenne",
    "nb_critiques",
    "capacite"
]
