# model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from data import DF

# ============================================================
# PRÉPARATION DES DONNÉES
# ============================================================
FEATURES = [
    "genre", "jour", "saison", "tranche_horaire",
    "meteo", "prix_moyen", "capacite",
    "nb_critiques", "note_moyenne", "semaine_promo", "est_weekend"
]

def preparer_features(df):
    """Encode les variables catégorielles."""
    df_enc = df[FEATURES].copy()
    encoders = {}
    for col in ["genre", "jour", "saison", "tranche_horaire", "meteo"]:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le
    return df_enc, encoders

# ── Entraînement ─────────────────────────────────────────────
X, encoders = preparer_features(DF)
y = DF["affluence"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ── Métriques ────────────────────────────────────────────────
y_pred  = model.predict(X_test)
MAE     = round(mean_absolute_error(y_test, y_pred), 1)
R2      = round(r2_score(y_test, y_pred), 3)

# ── Importance des variables ─────────────────────────────────
importance_df = pd.DataFrame({
    "variable":   FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)


# ============================================================
# FONCTION DE PRÉDICTION
# ============================================================
def predire_affluence(params: dict) -> dict:
    """
    params = {
        'genre': 'Musical',
        'jour': 'Samedi',
        'saison': 'Été',
        'tranche_horaire': 'Soir',
        'meteo': 'Ensoleillé',
        'prix_moyen': 30,
        'capacite': 300,
        'nb_critiques': 10,
        'note_moyenne': 4.2,
        'semaine_promo': 1,
        'est_weekend': 1,
    }
    """
    row = pd.DataFrame([params])

    # Encodage
    for col, le in encoders.items():
        if params[col] in le.classes_:
            row[col] = le.transform([params[col]])
        else:
            row[col] = 0

    row = row[FEATURES]
    pred = int(model.predict(row)[0])
    pred = max(0, min(pred, params["capacite"]))

    taux  = round(pred / params["capacite"] * 100, 1)
    ca    = round(pred * params["prix_moyen"], 2)

    return {
        "affluence_predite": pred,
        "taux_remplissage":  taux,
        "chiffre_affaire":   ca,
        "capacite":          params["capacite"],
    }