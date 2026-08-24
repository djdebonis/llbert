import os

import numpy as np
import pandas as pd
import streamlit as st

from eval import get_device, load_model, predict
from train import haversine_km


MODEL_PATH = os.getenv("LLBERT_MODEL_PATH", "output")
DATA_PATH = os.getenv("LLBERT_DATA_PATH", "training.csv")


@st.cache_resource
def get_predictor():
    device = get_device(os.getenv("LLBERT_DEVICE"))
    loaded = load_model(MODEL_PATH, device)
    return loaded[2], loaded[3], loaded[4], loaded[5], device


@st.cache_data
def load_rounds():
    return pd.read_csv(DATA_PATH).dropna(
        subset=["text", "latitude", "longitude"]
    ).reset_index(drop=True)


def predict_coordinates(text):
    encoder, head, coord_mean, coord_std, device = get_predictor()
    coordinates = predict(
        encoder, head, [text.strip()], coord_mean, coord_std, device, batch_size=1
    )[0]
    return float(coordinates[0]), float(coordinates[1])


def score_round(round_id, latitude, longitude):
    row = load_rounds().iloc[round_id]
    predicted_latitude, predicted_longitude = predict_coordinates(str(row["text"]))
    actual = np.array([[float(row["latitude"]), float(row["longitude"])]])
    guess = np.array([[latitude, longitude]])
    model = np.array([[predicted_latitude, predicted_longitude]])
    return {
        "guess_error_km": float(haversine_km(guess, actual)[0]),
        "model_error_km": float(haversine_km(model, actual)[0]),
        "actual_latitude": float(row["latitude"]),
        "actual_longitude": float(row["longitude"]),
    }


st.set_page_config(page_title="LLBert", page_icon="🌍", layout="centered")
st.title("LLBert")
st.caption("Predict the location suggested by sign text.")

predict_tab, round_tab = st.tabs(["Predict", "Challenge round"])

with predict_tab:
    text = st.text_area("Sign text", placeholder="market | bakery | church")
    if st.button("Predict coordinates", type="primary"):
        if not text.strip():
            st.warning("Enter some sign text first.")
        else:
            latitude, longitude = predict_coordinates(text)
            st.metric("Latitude", f"{latitude:.5f}")
            st.metric("Longitude", f"{longitude:.5f}")
            st.map(pd.DataFrame({"latitude": [latitude], "longitude": [longitude]}))

with round_tab:
    rounds = load_rounds()
    if "round_id" not in st.session_state:
        st.session_state.round_id = int(np.random.randint(len(rounds)))

    row = rounds.iloc[st.session_state.round_id]
    st.subheader("Where is this sign likely to be?")
    st.write(str(row["text"]))
    guess_latitude = st.number_input("Your latitude", min_value=-90.0, max_value=90.0, value=0.0)
    guess_longitude = st.number_input(
        "Your longitude", min_value=-180.0, max_value=180.0, value=0.0
    )
    score, next_round = st.columns(2)
    if score.button("Score guess", type="primary"):
        result = score_round(
            st.session_state.round_id, guess_latitude, guess_longitude
        )
        st.metric("Your distance", f"{result['guess_error_km']:.1f} km")
        st.metric("Model distance", f"{result['model_error_km']:.1f} km")
        st.write(
            "Actual coordinates: "
            f"{result['actual_latitude']:.5f}, {result['actual_longitude']:.5f}"
        )
    if next_round.button("New round"):
        st.session_state.round_id = int(np.random.randint(len(rounds)))
        st.rerun()