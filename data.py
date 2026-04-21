# data.py
import pandas as pd
import numpy as np

np.random.seed(42)

def generer_donnees(n=500):
    import pandas as pd
    import numpy as np

    np.random.seed(42)

    # ── Variables catégorielles ──────────────────────────────
    genres   = ["Comédie", "Tragédie", "Drame", "Musical", "One-Man-Show", "Ballet"]
    jours    = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    saisons  = ["Printemps", "Été", "Automne", "Hiver"]
    tranches = ["Matin", "Après-midi", "Soir"]
    meteos   = ["Ensoleillé", "Nuageux", "Pluvieux", "Neigeux"]

    data = {
        "genre":           np.random.choice(genres,   n),
        "jour":            np.random.choice(jours,    n),
        "saison":          np.random.choice(saisons,  n),
        "tranche_horaire": np.random.choice(tranches, n),
        "meteo":           np.random.choice(meteos,   n),
        "prix_moyen":      np.random.randint(10, 80,  n),
        "capacite":        np.random.choice([100, 200, 300, 500], n),
        "nb_critiques":    np.random.randint(0, 20,   n),
        "note_moyenne":    np.round(np.random.uniform(1, 5, n), 1),
        "semaine_promo":   np.random.choice([0, 1],   n, p=[0.7, 0.3]),
    }

    df = pd.DataFrame(data)

    # ── Features dérivées ────────────────────────────────────
    df["est_weekend"] = df["jour"].isin(["Samedi", "Dimanche"]).astype(int)

    base = df["capacite"] * 0.5

    # ── Effets de base ───────────────────────────────────────
    impact_genre = {
        "Musical": 0.25,
        "Comédie": 0.15,
        "One-Man-Show": 0.10,
        "Ballet": 0.10,
        "Drame": 0.00,
        "Tragédie": -0.05,
    }

    impact_jour = {
        "Samedi": 0.30, "Vendredi": 0.20, "Dimanche": 0.15,
        "Jeudi": 0.05, "Mercredi": 0.00, "Mardi": -0.05, "Lundi": -0.10,
    }

    impact_saison = {
        "Été": 0.15, "Printemps": 0.10, "Automne": 0.05, "Hiver": -0.05
    }

    impact_tranche = {
        "Soir": 0.25, "Après-midi": 0.10, "Matin": -0.10
    }

    # 🔁 météo corrigée (réaliste)
    impact_meteo = {
        "Ensoleillé": -0.10,
        "Nuageux": 0.00,
        "Pluvieux": 0.10,
        "Neigeux": 0.05
    }

    # ── Effet prix dépend du genre (clé !) ───────────────────
    elasticite_prix = {
        "Musical": 0.4,
        "Comédie": 0.7,
        "One-Man-Show": 0.8,
        "Ballet": 0.9,
        "Drame": 1.2,
        "Tragédie": 1.5
    }

    # ── Effet critiques dépend du genre ──────────────────────
    poids_critiques = {
        "Musical": 1,
        "Comédie": 1,
        "One-Man-Show": 1,
        "Ballet": 2,
        "Drame": 3,
        "Tragédie": 4
    }

    # ── Calcul affluence ─────────────────────────────────────
    affluence = (
        base
        + base * df["genre"].map(impact_genre)
        + base * df["jour"].map(impact_jour)
        + base * df["saison"].map(impact_saison)
        + base * df["tranche_horaire"].map(impact_tranche)
        + base * df["meteo"].map(impact_meteo)
        + df["note_moyenne"] * 12
        + df["nb_critiques"] * df["genre"].map(poids_critiques)
        + df["semaine_promo"] * base * 0.20
    )

    # 🔥 PRIX NON LINÉAIRE + dépend du genre
    affluence -= (df["prix_moyen"] ** 1.3) * df["genre"].map(elasticite_prix)

    # ── INTERACTIONS (ce qui rend ton dashboard intéressant) ─

    # 🎭 Ballet → boost fort le soir + weekend
    affluence += np.where(
        (df["genre"] == "Ballet") & (df["tranche_horaire"] == "Soir"),
        30, 0
    )
    affluence += np.where(
        (df["genre"] == "Ballet") & (df["est_weekend"] == 1),
        40, 0
    )

    # 😂 Comédie → boost vendredi + promo
    affluence += np.where(
        (df["genre"] == "Comédie") & (df["jour"] == "Vendredi"),
        25, 0
    )
    affluence += np.where(
        (df["genre"] == "Comédie") & (df["semaine_promo"] == 1),
        35, 0
    )

    # 🎭 Tragédie → dépend fortement de la note
    affluence += np.where(
        (df["genre"] == "Tragédie"),
        df["note_moyenne"] * 8,
        0
    )

    # 🎵 Musical → boost été + grande capacité
    affluence += np.where(
        (df["genre"] == "Musical") & (df["saison"] == "Été"),
        40, 0
    )
    affluence += np.where(
        (df["genre"] == "Musical") & (df["capacite"] >= 300),
        30, 0
    )

    # ── Bruit ────────────────────────────────────────────────
    affluence += np.random.normal(0, 15, n)

    # ── Finalisation ─────────────────────────────────────────
    df["affluence"] = np.clip(affluence, 0, df["capacite"]).astype(int)
    df["taux_remplissage"] = (df["affluence"] / df["capacite"] * 100).round(1)
    df["chiffre_affaire"] = (df["affluence"] * df["prix_moyen"]).round(2)

    # ── Dates ────────────────────────────────────────────────
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    dates_array = dates.to_numpy().copy()
    np.random.shuffle(dates_array)

    df["date"] = sorted(dates_array)
    df["mois"] = pd.to_datetime(df["date"]).dt.strftime("%b %Y")
    df["annee"] = pd.to_datetime(df["date"]).dt.year

    return df


# ── Export ───────────────────────────────────────────────────
DF = generer_donnees(500)

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
