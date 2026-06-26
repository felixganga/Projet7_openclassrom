#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Streamlit optimisé.

Le dashboard ne charge plus de CSV, ne charge plus le modèle et ne lance plus
shap_service.py. Il interroge l'API Flask, qui centralise les traitements lourds.
"""

import base64
import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import requests
import seaborn as sns
import streamlit as st
from matplotlib.image import imread

st.set_page_config(page_title="Dashboard Crédit", layout="wide")

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:5000")
LOGO_PATH = "./Data/logo.png"
ID_COL = "Unnamed: 0.1"
FILTER_LABELS = {
    "CODE_GENDER": "Sexe",
    "YEARS_BIRTH": "Âge",
    "FLAG_OWN_CAR": "Possession voiture",
    "FLAG_OWN_REALTY": "Possession immobilier",
}


@st.cache_data(ttl=300)
def api_get(endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Appel API cacheable pour éviter les appels répétés inutiles."""
    url = f"{API_BASE_URL}{endpoint}"
    response = requests.get(url, params=params, timeout=60)

    if response.status_code != 200:
        try:
            detail = response.json().get("error", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"Erreur API {response.status_code} sur {endpoint} : {detail}")

    return response.json()


def get_filters() -> dict[str, list[str]]:
    return api_get("/clients/filters")


def get_client_ids(source: str = "score") -> list[int]:
    return api_get("/clients/ids", {"source": source})["ids"]


def tab_client():
    st.markdown("### Tableau clientèle")

    filters = get_filters()
    selected_filters = {}
    for col, label in FILTER_LABELS.items():
        if col in filters:
            selected_filters[col] = st.selectbox(
                label,
                ["All"] + filters[col],
                key=f"filter_{col}",
            )

    params = {col: value for col, value in selected_filters.items() if value != "All"}
    payload = api_get("/clients", params)
    df_display = pd.DataFrame(payload["data"])

    st.dataframe(df_display, use_container_width=True)
    st.write(f"**Total clients :** {payload['total']}")


def comparaison_client(client_id: int):
    st.markdown("### Comparaison clientèle")

    profile_global = st.checkbox("Profil global (tous clients)", value=True)
    size = st.slider("Taille du groupe similaire", 2, 1000, value=500)
    profile = "global" if profile_global else "neighbors"

    payload = api_get(
        "/comparison",
        {"id_client": client_id, "size": size, "profile": profile},
    )

    group = payload["group"]
    client = payload["client"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_config = [
        ("YEARS_BIRTH", "Âge (années)"),
        ("AMT_CREDIT", "Montant du crédit"),
    ]

    for ax, (col, title) in zip(axes, plot_config):
        if col not in group or col not in client:
            ax.set_visible(False)
            continue
        sns.kdeplot(group[col], ax=ax, label="Groupe")
        ax.axvline(client[col], linestyle="--", label="Client")
        ax.set_title(title)
        ax.legend()

    st.pyplot(fig, use_container_width=True)
    st.caption(f"Groupe comparé : {payload['group_size']} clients")


def show_score_and_shap(client_id: int):
    st.markdown("### Score & explication SHAP")

    score = api_get("/credit/", {"id_client": client_id})
    pred, proba = score["prediction"], score["proba"]
    label = "✅ Accordé" if pred == 0 else "❌ Refusé"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=proba,
        title={"text": label},
        gauge={"axis": {"range": [0, 1]}},
    ))
    st.plotly_chart(fig, use_container_width=True)

    with st.spinner("Calcul de l'explication SHAP..."):
        shap_payload = api_get("/shap", {"id_client": client_id})
        image_bytes = base64.b64decode(shap_payload["image_base64"])
        st.image(image_bytes, caption=f"SHAP client {client_id}")


if os.path.exists(LOGO_PATH):
    st.sidebar.image(imread(LOGO_PATH))

st.sidebar.title("Navigation")
pages = ["Tableau clientèle", "Comparaison clientèle", "Visualisation score"]
choice = st.sidebar.radio("Aller à", pages)

try:
    if choice == pages[0]:
        tab_client()
    elif choice == pages[1]:
        client_ids = get_client_ids(source="full")
        cid = st.sidebar.selectbox("Client", client_ids)
        if st.sidebar.button("Montrer comparaison"):
            comparaison_client(int(cid))
    else:
        client_ids = get_client_ids(source="score")
        cid = st.sidebar.selectbox("Client", client_ids)
        if st.sidebar.button("Montrer score & SHAP"):
            show_score_and_shap(int(cid))
except Exception as exc:
    st.error(str(exc))
