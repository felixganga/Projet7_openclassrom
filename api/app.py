#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Flask optimisée pour le dashboard crédit.

Objectif : centraliser les chargements lourds côté API et exposer uniquement
les données nécessaires au dashboard Streamlit.

Correction principale : les colonnes envoyées au modèle sont maintenant
alignées sur celles utilisées pendant l'entraînement. Les colonnes métier
ajoutées pour l'affichage, comme YEARS_BIRTH, ne sont plus envoyées au modèle.
"""

import base64
import io
import pickle
import warnings
from functools import lru_cache
from pathlib import Path
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
from flask import Flask, jsonify, request
from sklearn.neighbors import NearestNeighbors

def download_if_missing(url, destination):
    """
    Télécharge un fichier uniquement s'il n'existe pas.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not destination.exists():
        print(f"Téléchargement de {destination.name}...")
        urllib.request.urlretrieve(url, destination)

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

download_if_missing(
    "https://github.com/felixganga/Projet7_openclassrom/releases/download/v1.0/X_rfecv_test.csv",
    DATA_DIR / "X_rfecv_test.csv",
)

download_if_missing(
    "https://github.com/felixganga/Projet7_openclassrom/releases/download/v1.0/full_df_sample.csv",
    DATA_DIR / "full_df_sample.csv",
)

ID_COL = "Unnamed: 0.1"

# Colonnes à ne jamais envoyer au modèle.
# YEARS_BIRTH est une colonne métier créée pour l'affichage ; elle n'était pas
# présente au moment du fit du pipeline, donc elle provoquait :
# "Feature names unseen at fit time: YEARS_BIRTH".
IGNORE_FEATURES = ["Unnamed: 0", ID_COL, "INDEX", "TARGET", "YEARS_BIRTH"]

DISPLAY_COLS = [
    ID_COL,
    "CODE_GENDER",
    "YEARS_BIRTH",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "AMT_CREDIT",
    "AMT_ANNUITY",
]
FILTER_COLS = ["CODE_GENDER", "YEARS_BIRTH", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]
COMPARISON_COLS = ["YEARS_BIRTH", "AMT_CREDIT"]


@lru_cache(maxsize=1)
def load_scoring_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "X_rfecv_test.csv")
    return add_business_columns(df)


@lru_cache(maxsize=1)
def load_full_data() -> pd.DataFrame:
    """Données utilisées uniquement pour la comparaison clientèle."""
    full_path = DATA_DIR / "full_df_sample.csv"
    if full_path.exists():
        df = pd.read_csv(full_path)
    else:
        df = pd.read_csv(DATA_DIR / "X_rfecv_test.csv")
    return add_business_columns(df)


@lru_cache(maxsize=1)
def load_model():
    with open(MODEL_DIR / "lgbm_optimise.pkl", "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def load_shap_explainer():
    model = load_model()
    clf = get_final_estimator(model)
    return shap.TreeExplainer(clf)


def add_business_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "DAYS_BIRTH" in df.columns and "YEARS_BIRTH" not in df.columns:
        df["YEARS_BIRTH"] = (df["DAYS_BIRTH"] / -365).astype(int)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Fallback si le modèle ne contient pas feature_names_in_."""
    return [col for col in df.columns if col not in IGNORE_FEATURES]


def get_model_feature_names(model, df: pd.DataFrame) -> list[str]:
    """Retourne les colonnes exactes attendues par le pipeline entraîné."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "steps"):
        for _, step in model.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return feature_columns(df)


def build_model_input(row: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    model = load_model()
    expected_features = get_model_feature_names(model, df)

    missing = [col for col in expected_features if col not in row.columns]
    if missing:
        raise ValueError(
            "Colonnes manquantes pour le modèle : " + ", ".join(missing)
        )

    return row[expected_features]


def get_final_estimator(model):
    if hasattr(model, "steps"):
        return model.steps[-1][1]
    return model


def transform_for_final_estimator(model, x_client: pd.DataFrame):
    """Applique les étapes de preprocessing du pipeline avant SHAP."""
    if not hasattr(model, "steps"):
        return x_client, list(x_client.columns)

    x_transformed = x_client
    for _, step in model.steps[:-1]:
        if hasattr(step, "transform"):
            x_transformed = step.transform(x_transformed)

    n_features = x_transformed.shape[1]
    original_features = list(x_client.columns)
    if n_features == len(original_features):
        feature_names = original_features
    else:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    x_transformed = pd.DataFrame(x_transformed, columns=feature_names)
    return x_transformed, feature_names


def get_client_row(df: pd.DataFrame, client_id: int) -> pd.DataFrame:
    row = df[df[ID_COL] == client_id]
    if row.empty:
        raise ValueError(f"Aucun client trouvé pour id_client={client_id}")
    return row


def parse_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df
    for col in FILTER_COLS:
        value = request.args.get(col)
        if value not in (None, "", "All") and col in filtered.columns:
            filtered = filtered[filtered[col].astype(str) == str(value)]
    return filtered


@app.route("/hello", methods=["GET"])
def hello():
    return "Hello, World!"


@app.route("/clients/filters", methods=["GET"])
def clients_filters():
    df = load_scoring_data()
    filters = {}
    for col in FILTER_COLS:
        if col in df.columns:
            filters[col] = sorted(df[col].dropna().astype(str).unique().tolist())
    return jsonify(filters)


@app.route("/clients/ids", methods=["GET"])
def clients_ids():
    source = request.args.get("source", "score")
    df = load_full_data() if source == "full" else load_scoring_data()
    return jsonify({"ids": df[ID_COL].dropna().astype(int).tolist()})


@app.route("/clients", methods=["GET"])
def clients():
    """Retourne seulement les colonnes utiles au tableau clientèle."""
    df = parse_filters(load_scoring_data())
    limit = request.args.get("limit", default=1000, type=int)
    cols = [col for col in DISPLAY_COLS if col in df.columns]
    data = df[cols].head(limit).to_dict(orient="records")
    return jsonify({"total": int(len(df)), "data": data})


@app.route("/credit/", methods=["GET"])
def credit():
    id_client = request.args.get("id_client", type=int)
    if id_client is None:
        return jsonify({"error": "Paramètre id_client manquant"}), 400

    df = load_scoring_data()
    try:
        row = get_client_row(df, id_client)
        x_client = build_model_input(row, df)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404

    model = load_model()
    proba = model.predict_proba(x_client)
    prediction = model.predict(x_client)

    return jsonify({
        "id_client": id_client,
        "prediction": int(prediction[0]),
        "proba": float(proba[0][0]),
    })


@app.route("/comparison", methods=["GET"])
def comparison():
    """Retourne uniquement les valeurs nécessaires aux graphiques de comparaison."""
    id_client = request.args.get("id_client", type=int)
    size = request.args.get("size", default=500, type=int)
    profile = request.args.get("profile", default="global")

    if id_client is None:
        return jsonify({"error": "Paramètre id_client manquant"}), 400

    df = load_full_data()
    try:
        row = get_client_row(df, id_client)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404

    if profile == "neighbors":
        features = feature_columns(df)
        n_neighbors = max(2, min(size, len(df)))
        neigh = NearestNeighbors(n_neighbors=n_neighbors).fit(df[features])
        idxs = neigh.kneighbors(row[features], return_distance=False)[0]
        group = df.iloc[idxs]
    else:
        group = df

    cols = [col for col in COMPARISON_COLS if col in df.columns]
    return jsonify({
        "id_client": id_client,
        "group_size": int(len(group)),
        "client": row[cols].iloc[0].to_dict(),
        "group": {col: group[col].dropna().tolist() for col in cols},
    })


@app.route("/shap", methods=["GET"])
def shap_plot():
    """Calcule SHAP côté API et renvoie une image PNG encodée en base64."""
    id_client = request.args.get("id_client", type=int)
    if id_client is None:
        return jsonify({"error": "Paramètre id_client manquant"}), 400

    df = load_scoring_data()
    try:
        row = get_client_row(df, id_client)
        x_client = build_model_input(row, df)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404

    model = load_model()
    x_shap, _ = transform_for_final_estimator(model, x_client)
    explainer = load_shap_explainer()
    shap_values = explainer.shap_values(x_shap)
    values_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(8, 5))
    shap.summary_plot(values_to_plot, x_shap, plot_type="bar", show=False)
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    plt.close()
    buffer.seek(0)

    return jsonify({
        "id_client": id_client,
        "image_base64": base64.b64encode(buffer.read()).decode("utf-8"),
    })


if __name__ == "__main__":
    app.run(debug=True)
