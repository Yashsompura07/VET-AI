"""Loads the trained model and turns user input into a prediction + explanation."""

import json
import os
import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "symptom_model.joblib")
DISEASES_PATH = os.path.join(HERE, "diseases.json")

# Loaded once when the app starts
_bundle = joblib.load(MODEL_PATH)
with open(DISEASES_PATH, encoding="utf-8") as f:
    _kb = json.load(f)

_disease_by_id = {d["id"]: d for d in _kb["diseases"]}
_symptom_label = {s["id"]: s["label"] for s in _kb["symptoms"]}


def get_symptoms():
    return _kb["symptoms"]


def get_diseases():
    return _kb["diseases"]


def get_disease(disease_id):
    return _disease_by_id.get(disease_id)


def get_model_info():
    return {"model_name": _bundle["model_name"], "metrics": _bundle["metrics"]}


def _row_from_input(animal_type, age_months, weight_kg, symptoms,
                    temp_c=None, heart_rate=None):
    """Build a single feature row matching the training schema."""
    feature_cols = _bundle["feature_cols"]
    symptom_ids = _bundle["symptom_ids"]
    selected = set(symptoms or [])

    row = {col: 0 for col in feature_cols}
    for sid in symptom_ids:
        if sid in selected:
            row[sid] = 1
    if "age_months" in row:
        row["age_months"] = age_months or 0
    if "weight_kg" in row:
        row["weight_kg"] = weight_kg or 0
    # Vital signs. If the farmer did not measure them, use 0 as an "unknown"
    # sentinel (the model is trained to treat 0 as missing and rely on symptoms),
    # rather than a normal value that would wrongly imply "no fever".
    if "temp_c" in row:
        row["temp_c"] = float(temp_c) if temp_c else 0.0
    if "heart_rate" in row:
        row["heart_rate"] = int(heart_rate) if heart_rate else 0
    animal_col = f"animal_{animal_type}"
    if animal_col in row:
        row[animal_col] = 1

    return pd.DataFrame([row], columns=feature_cols)


def predict(animal_type, age_months, weight_kg, symptoms, top_k=3,
            temp_c=None, heart_rate=None):
    """Return ranked disease predictions with confidence, explanation and risk."""
    model = _bundle["model"]
    label_encoder = _bundle["label_encoder"]

    X = _row_from_input(animal_type, age_months, weight_kg, symptoms,
                        temp_c, heart_rate)
    proba = model.predict_proba(X)[0]

    ranked_idx = proba.argsort()[::-1][:top_k]
    predictions = []
    for idx in ranked_idx:
        disease_id = label_encoder.classes_[idx]
        disease = _disease_by_id[disease_id]
        predictions.append({
            "disease_id": disease_id,
            "name": disease["name"],
            "confidence": round(float(proba[idx]) * 100, 1),
            "risk_level": disease["risk_level"],
        })

    top = predictions[0]
    top_disease = _disease_by_id[top["disease_id"]]

    try:
        from app.chatbot import DISEASES_HI
        knowledge_hi = DISEASES_HI.get(top["disease_id"], {})
    except Exception:
        knowledge_hi = {}

    # Explanation: which reported symptoms are known indicators of the top disease
    selected = set(symptoms or [])
    matched = [s for s in top_disease["key_symptoms"] if s in selected]
    explanation = {
        "matched_symptoms": [
            {"id": s, "label": _symptom_label.get(s, s)} for s in matched
        ],
        "summary": _build_summary(top, matched),
    }

    return {
        "predictions": predictions,
        "top": top,
        "explanation": explanation,
        "knowledge": top_disease,
        "knowledge_hi": knowledge_hi,
    }


def predict_proba_full(animal_type, age_months, weight_kg, symptoms,
                       temp_c=None, heart_rate=None):
    """Return the full {disease_id: probability} distribution over all diseases.
    Used by the hybrid engine to fuse with the image model."""
    model = _bundle["model"]
    label_encoder = _bundle["label_encoder"]
    X = _row_from_input(animal_type, age_months, weight_kg, symptoms,
                        temp_c, heart_rate)
    proba = model.predict_proba(X)[0]
    return {label_encoder.classes_[i]: float(proba[i]) for i in range(len(proba))}


def _build_summary(top, matched):
    if matched:
        reasons = ", ".join(_symptom_label.get(s, s).lower() for s in matched)
        return (f"Most likely {top['name']} ({top['confidence']}% confidence), based on the "
                f"reported signs: {reasons}.")
    return (f"Most likely {top['name']} ({top['confidence']}% confidence) based on the overall "
            f"symptom pattern.")
