"""
VetAI — Symptom-based disease model trainer (v2, "clean merged" dataset).

WHAT CHANGED FROM v1 (and why it matters for your report):

1. 15 diseases instead of 10. The five added ones (Bovine TB, Bluetongue,
   Johne's, CAE, Scrapie) are real livestock conditions identified in the public
   Kaggle livestock symptom dataset, which our original knowledge base lacked.

2. Vital signs are now features: body temperature (C) and heart rate (bpm).
   The public dataset showed these are clinically informative, and our knowledge
   base now stores a realistic range per disease based on veterinary reference
   values (e.g. milk fever runs SUBnormal temperature, bloat runs a high heart
   rate from distress).

3. Real data is blended in. Rows from the Kaggle dataset that genuinely map to
   our diseases are converted to our schema and added as real training examples,
   rather than merging two mismatched tables wholesale.

WHY NOT A PURE MERGE: only 4 of our original diseases appear in the Kaggle file,
6 have zero rows, and its 431 records span 139 disease labels (~3 rows each) with
inconsistent naming. Trained alone on its livestock subset it reaches only ~52%
cross-validated accuracy. So we use it to ENRICH a curated dataset, and report
the benchmark honestly.

Run:  python train_symptom_model.py
"""

import json
import os
import re
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
DISEASES_PATH = os.path.join(HERE, "app", "ml", "diseases.json")
REAL_CSV = os.path.join(HERE, "data", "kaggle_animal_disease.csv")
MODEL_OUT = os.path.join(HERE, "app", "ml", "symptom_model.joblib")

SAMPLES_PER_DISEASE = 600
KEY_SYMPTOM_PROB = 0.82
NOISE_SYMPTOM_PROB = 0.06
RANDOM_STATE = 42

REAL_LABEL_MAP = {
    "foot and mouth disease": "fmd",
    "foot-and-mouth disease": "fmd",
    "mastitis": "mastitis",
    "bovine mastitis": "mastitis",
    "bovine respiratory disease": "brd",
    "bovine respiratory disease complex": "brd",
    "bovine respiratory syncytial virus": "brd",
    "bovine pneumonia": "brd",
    "pneumonia": "brd",
    "bovine viral diarrhea": "bvd_enteritis",
    "bovine tuberculosis": "bovine_tb",
    "tuberculosis": "bovine_tb",
    "bluetongue": "bluetongue",
    "bluetongue virus": "bluetongue",
    "blue tongue": "bluetongue",
    "blue tongue virus": "bluetongue",
    "blue tongue disease": "bluetongue",
    "johne's disease": "johnes",
    "bovine johne's disease": "johnes",
    "caprine arthritis encephalitis": "cae",
    "caprine arthritis encephalitis virus": "cae",
    "caprine arthritis": "cae",
    "caprine viral arthritis": "cae",
    "scrapie": "scrapie",
    "scrapie disease": "scrapie",
}

REAL_SYMPTOM_MAP = {
    "Appetite_Loss": "loss_of_appetite",
    "Diarrhea": "diarrhea",
    "Coughing": "cough",
    "Labored_Breathing": "difficulty_breathing",
    "Lameness": "lameness",
    "Skin_Lesions": "skin_lesions",
    "Nasal_Discharge": "nasal_discharge",
}


def load_kb():
    with open(DISEASES_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_curated(kb, symptom_ids):
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for disease in kb["diseases"]:
        key = set(disease["key_symptoms"])
        animals = disease["animals"]
        v = disease["vitals"]
        for _ in range(SAMPLES_PER_DISEASE):
            row = {}
            for sid in symptom_ids:
                p = KEY_SYMPTOM_PROB if sid in key else NOISE_SYMPTOM_PROB
                row[sid] = int(rng.random() < p)
            if sum(row[k] for k in key) == 0:
                row[str(rng.choice(list(key)))] = 1
            row["animal_type"] = str(rng.choice(animals))
            row["age_months"] = int(rng.integers(3, 156))
            row["weight_kg"] = int(rng.integers(15, 650))
            row["temp_c"] = round(float(rng.uniform(*v["temp_c"])), 1)
            row["heart_rate"] = int(rng.integers(*v["hr_bpm"]))
            # ~35% of farmers won't measure vitals: teach the model that 0 = unknown,
            # so at inference it relies on symptoms alone rather than being misled.
            if rng.random() < 0.35:
                row["temp_c"] = 0.0
                row["heart_rate"] = 0
            row["disease"] = disease["id"]
            row["origin"] = "curated"
            rows.append(row)
    return pd.DataFrame(rows)


def load_real(symptom_ids):
    if not os.path.exists(REAL_CSV):
        return pd.DataFrame()
    df = pd.read_csv(REAL_CSV)
    df = df[df["Animal_Type"].isin(["Cow", "Goat", "Sheep", "Buffalo"])].copy()

    def norm(s):
        return re.sub(r"\s+", " ", str(s).lower().strip())

    df["target"] = df["Disease_Prediction"].apply(lambda s: REAL_LABEL_MAP.get(norm(s)))
    df = df[df["target"].notna()]
    if df.empty:
        return pd.DataFrame()

    rows = []
    for _, r in df.iterrows():
        row = {sid: 0 for sid in symptom_ids}
        for col, sid in REAL_SYMPTOM_MAP.items():
            if sid in row and str(r.get(col, "")).strip().lower() == "yes":
                row[sid] = 1
        text = " ".join(str(r.get(f"Symptom_{i}", "")).lower() for i in range(1, 5))
        for kw, sid in [("fever", "fever"), ("lethargy", "weakness"),
                        ("weight loss", "weight_loss"), ("salivation", "salivation"),
                        ("drooling", "salivation"), ("swollen", "joint_swelling"),
                        ("blister", "mouth_lesions"), ("ulcer", "mouth_ulcers"),
                        ("milk", "milk_reduction"), ("dehydration", "dehydration")]:
            if kw in text and sid in row:
                row[sid] = 1

        temp = re.findall(r"[\d.]+", str(r.get("Body_Temperature", "")))
        animal = str(r["Animal_Type"]).lower()
        row["animal_type"] = animal
        row["age_months"] = int(float(r.get("Age", 3)) * 12)
        row["weight_kg"] = int(float(r.get("Weight", 200)))
        row["temp_c"] = float(temp[0]) if temp else 38.8
        row["heart_rate"] = int(r.get("Heart_Rate", 75))
        row["disease"] = r["target"]
        row["origin"] = "real"
        rows.append(row)
    return pd.DataFrame(rows)


def prepare(df, symptom_ids):
    dummies = pd.get_dummies(df["animal_type"], prefix="animal")
    base = df[symptom_ids + ["age_months", "weight_kg", "temp_c", "heart_rate"]]
    X = pd.concat([base, dummies], axis=1)
    return X, list(X.columns)


def main():
    kb = load_kb()
    symptom_ids = [s["id"] for s in kb["symptoms"]]
    print(f"Knowledge base: {len(kb['diseases'])} diseases, {len(symptom_ids)} symptoms")

    curated = build_curated(kb, symptom_ids)
    real = load_real(symptom_ids)
    print(f"Curated (generated) rows: {len(curated)}")
    print(f"Real rows blended in    : {len(real)}"
          + (f"  covering {real['disease'].nunique()} diseases" if len(real) else ""))

    df = pd.concat([curated, real], ignore_index=True).fillna(0)
    X, feature_cols = prepare(df, symptom_ids)
    le = LabelEncoder()
    y = le.fit_transform(df["disease"])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)
    rf_acc, rf_f1 = accuracy_score(y_te, rf_pred), f1_score(y_te, rf_pred, average="macro")
    print(f"  Random Forest -> accuracy={rf_acc:.3f}  macro-F1={rf_f1:.3f}")

    print("Training XGBoost...")
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        subsample=0.9, colsample_bytree=0.9,
                        objective="multi:softprob", num_class=len(le.classes_),
                        random_state=RANDOM_STATE, n_jobs=-1, eval_metric="mlogloss")
    xgb.fit(X_tr, y_tr)
    xgb_pred = xgb.predict(X_te)
    xgb_acc, xgb_f1 = accuracy_score(y_te, xgb_pred), f1_score(y_te, xgb_pred, average="macro")
    print(f"  XGBoost       -> accuracy={xgb_acc:.3f}  macro-F1={xgb_f1:.3f}")

    if xgb_f1 >= rf_f1:
        best_name, best, best_pred = "XGBoost", xgb, xgb_pred
    else:
        best_name, best, best_pred = "Random Forest", rf, rf_pred
    print(f"\nSelected: {best_name}")

    cv = cross_val_score(
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        X, y, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE))
    print(f"5-fold CV accuracy: {cv.mean():.3f} (+/- {cv.std():.3f})")

    print("\nPer-disease report (best model):")
    print(classification_report(y_te, best_pred, target_names=le.classes_, zero_division=0))

    joblib.dump({
        "model": best, "model_name": best_name,
        "feature_cols": feature_cols, "symptom_ids": symptom_ids,
        "label_encoder": le,
        "metrics": {
            "random_forest": {"accuracy": rf_acc, "macro_f1": rf_f1},
            "xgboost": {"accuracy": xgb_acc, "macro_f1": xgb_f1},
            "cv_accuracy": float(cv.mean()), "selected": best_name,
            "n_diseases": len(le.classes_),
            "n_real_rows": int(len(real)), "n_curated_rows": int(len(curated)),
        },
    }, MODEL_OUT)
    print(f"\nSaved -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
