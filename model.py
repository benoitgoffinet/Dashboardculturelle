# model.py
import pandas as pd

import joblib

# ============================================================
# CHARGEMENT LOCAL DU MODÈLE
# ============================================================
ARTIFACT = joblib.load("theatre_model.joblib")
model = ARTIFACT["model"]
encoders = ARTIFACT["encoders"]
FEATURES = ARTIFACT["features"]
MAE = ARTIFACT["mae"]
R2 = ARTIFACT["r2"]


importance_df = pd.DataFrame({
    "variable": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)


# ============================================================
# FONCTION DE PRÉDICTION
# ============================================================
def predire_affluence(params: dict) -> dict:
    
    row = pd.DataFrame([params])

  
    for col, le in encoders.items():
        if params[col] in le.classes_:
            row[col] = le.transform([params[col]])
        else:
            row[col] = 0

    row = row[FEATURES]
    pred = int(model.predict(row)[0])
    pred = max(0, min(pred, params["capacite"]))

    taux = round(pred / params["capacite"] * 100, 1)
    ca = round(pred * params["prix_moyen"], 2)

    return {
        "affluence_predite": pred,
        "taux_remplissage": taux,
        "chiffre_affaire": ca,
        "capacite": params["capacite"],
    }
