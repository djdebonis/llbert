import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from eval import get_device, load_model, predict
from train import haversine_km


ROOT = Path(__file__).parent
MODEL_PATH = Path(os.getenv("LLBERT_MODEL_PATH", ROOT / "output"))
DATA_PATH = Path(os.getenv("LLBERT_DATA_PATH", ROOT / "training.csv"))
WEB_PATH = ROOT / "web"

app = FastAPI(title="LLBert API", version="1.0.0")
origins = [origin.strip() for origin in os.getenv("LLBERT_ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}"},
    )


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class ScoreRequest(PredictionRequest):
    round_id: int = Field(ge=0)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


@lru_cache(maxsize=1)
def get_predictor():
    device = get_device(os.getenv("LLBERT_DEVICE"))
    loaded = load_model(str(MODEL_PATH), device)
    return loaded[2], loaded[3], loaded[4], loaded[5], device


def predict_coordinates(text):
    encoder, head, coord_mean, coord_std, device = get_predictor()
    coordinates = predict(
        encoder, head, [text.strip()], coord_mean, coord_std, device, batch_size=1
    )[0]
    return float(coordinates[0]), float(coordinates[1])


def load_rounds():
    if not DATA_PATH.exists():
        raise HTTPException(status_code=503, detail="training.csv is not available")
    rounds = pd.read_csv(DATA_PATH).dropna(subset=["text", "latitude", "longitude"])
    if rounds.empty:
        raise HTTPException(status_code=503, detail="No playable rounds are available")
    return rounds.reset_index(drop=True)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_PATH / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model_path": str(MODEL_PATH)}


@app.post("/api/predict")
def predict_text(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Sign text cannot be blank")
    latitude, longitude = predict_coordinates(request.text)
    return {"latitude": latitude, "longitude": longitude}


@app.get("/api/round")
def new_round():
    rounds = load_rounds()
    round_id = int(np.random.randint(len(rounds)))
    return {"round_id": round_id, "text": str(rounds.iloc[round_id]["text"])}


@app.post("/api/round/score")
def score_round(request: ScoreRequest):
    rounds = load_rounds()
    if request.round_id >= len(rounds):
        raise HTTPException(status_code=404, detail="Round not found")
    row = rounds.iloc[request.round_id]
    predicted_latitude, predicted_longitude = predict_coordinates(str(row["text"]))
    actual = np.array([[float(row["latitude"]), float(row["longitude"])]])
    guess = np.array([[request.latitude, request.longitude]])
    model = np.array([[predicted_latitude, predicted_longitude]])
    return {
        "guess_error_km": float(haversine_km(guess, actual)[0]),
        "model_error_km": float(haversine_km(model, actual)[0]),
        "predicted_latitude": predicted_latitude,
        "predicted_longitude": predicted_longitude,
        "actual_latitude": float(row["latitude"]),
        "actual_longitude": float(row["longitude"]),
    }