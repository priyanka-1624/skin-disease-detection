from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.inference import InferenceEngine

app = FastAPI(title="Intelligent Skin Disease Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: InferenceEngine | None = None
root = Path(__file__).resolve().parents[1]
artifacts_dir = root / "logs" / "artifacts"
# plots directory used for serving training curve images; disabled since plots are no longer generated
# plots_dir = root / "logs" / "plots"
# plots_dir.mkdir(parents=True, exist_ok=True)
# app.mount("/plots", StaticFiles(directory=str(plots_dir)), name="plots")


@app.on_event("startup")
def on_startup() -> None:
    global engine
    engine = InferenceEngine()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/training-summary")
def training_summary() -> dict:
    metrics_path = artifacts_dir / "metrics.json"
    metadata_path = artifacts_dir / "metadata.json"

    if not metrics_path.exists() or not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Training artifacts are not available yet")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    # no plot files available since plotting is disabled
    # plots: dict[str, str] = {}
    # backbone_curve = plots_dir / "backbone_training_curves.png"
    # softmax_curve = plots_dir / "softmax_training_curves.png"
    #
    # if backbone_curve.exists():
    #     plots["backbone_curve"] = f"/plots/backbone_training_curves.png?v={int(backbone_curve.stat().st_mtime)}"
    # if softmax_curve.exists():
    #     plots["softmax_curve"] = f"/plots/softmax_training_curves.png?v={int(softmax_curve.stat().st_mtime)}"
    #
    return {"metadata": metadata, "metrics": metrics}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized")

    filename = file.filename or ""
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        raise HTTPException(status_code=400, detail="Unsupported image format")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image upload")

    try:
        return engine.predict(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
