"""Unit & Integration tests for VetAI ML models and hybrid fusion engine."""
import pytest
from app.ml import predict as sym_ml
from app.ml import hybrid as hybrid_ml
from app.ml import image_predict as img_ml


def test_kb_loaded():
    """Verify that the knowledge base contains all 29 symptoms and 15 diseases."""
    symptoms = sym_ml.get_symptoms()
    diseases = sym_ml.get_diseases()
    assert len(symptoms) == 29, f"Expected 29 symptoms, got {len(symptoms)}"
    assert len(diseases) == 15, f"Expected 15 diseases, got {len(diseases)}"

    # Check key diseases exist
    disease_ids = {d["id"] for d in diseases}
    for expected_id in ["fmd", "lsd", "mastitis", "bloat", "hs"]:
        assert expected_id in disease_ids, f"Missing disease: {expected_id}"


def test_model_info():
    """Verify ML model metadata reports valid accuracy and model family."""
    info = sym_ml.get_model_info()
    assert "model_name" in info
    assert "metrics" in info
    metrics = info["metrics"]
    assert metrics["selected"] in ["XGBoost", "Random Forest"]
    assert metrics["n_diseases"] == 15
    assert metrics["xgboost"]["accuracy"] > 0.90


def test_fmd_prediction():
    """Verify that classic FMD signs in cattle yield FMD as the top diagnosis."""
    res = sym_ml.predict(
        animal_type="cow",
        age_months=36,
        weight_kg=420,
        symptoms=["mouth_lesions", "salivation", "hoof_lesions", "lameness", "fever"],
        top_k=3,
        temp_c=40.5
    )
    assert res["top"]["disease_id"] == "fmd"
    assert res["top"]["confidence"] >= 60.0
    assert "fmd" in [p["disease_id"] for p in res["predictions"]]
    assert len(res["explanation"]["matched_symptoms"]) >= 2


def test_mastitis_prediction():
    """Verify that udder swelling and abnormal milk yield Mastitis."""
    res = sym_ml.predict(
        animal_type="cow",
        age_months=48,
        weight_kg=460,
        symptoms=["udder_swelling", "milk_reduction", "fever"],
        top_k=3
    )
    assert res["top"]["disease_id"] == "mastitis"
    assert res["top"]["confidence"] >= 60.0


def test_bloat_prediction():
    """Verify that bloating and respiratory distress yield Bloat."""
    res = sym_ml.predict(
        animal_type="buffalo",
        age_months=30,
        weight_kg=500,
        symptoms=["bloating", "difficulty_breathing", "reduced_rumination"],
        top_k=3
    )
    assert res["top"]["disease_id"] == "bloat"
    assert res["top"]["confidence"] >= 60.0


def test_predict_proba_full_sums_to_one():
    """Verify that predict_proba_full outputs valid probability distributions across all 15 diseases."""
    dist = sym_ml.predict_proba_full(
        animal_type="cow",
        age_months=24,
        weight_kg=380,
        symptoms=["fever", "nasal_discharge"]
    )
    assert len(dist) == 15
    total = sum(dist.values())
    assert abs(total - 1.0) < 1e-4, f"Sum of probabilities was {total}, expected 1.0"


def test_hybrid_fuse_mathematics():
    """Verify that hybrid late-fusion combines symptom and image probability distributions properly."""
    sym_proba = {d["id"]: 1.0 / 15 for d in sym_ml.get_diseases()}
    img_proba = {"fmd": 0.85, "lsd": 0.05, "healthy": 0.10}

    fused = hybrid_ml._fuse(sym_proba, img_proba)
    assert len(fused) == 15
    assert abs(sum(fused.values()) - 1.0) < 1e-4
    # FMD probability should be amplified by the image model
    assert fused["fmd"] > fused["lsd"]
    assert fused["fmd"] > fused["mastitis"]


def test_reject_blank_or_dark_image():
    """Verify that blank/uniform or completely dark images are rejected by validation."""
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color=(0, 0, 0)).save(buf, format="JPEG")
    dark_bytes = buf.getvalue()

    with pytest.raises(ValueError) as exc:
        img_ml.predict_image(dark_bytes)
    assert "अस्पष्ट या खाली फोटो" in str(exc.value) or "Blank" in str(exc.value)


def test_reject_human_face_if_present():
    """Verify that human face images are rejected with bilingual alert message."""
    import os
    user_face_img = r"C:/Users/bharg/.gemini/antigravity-ide/brain/526badf3-cc59-42ce-9cdb-66f8b22e5932/.user_uploaded/media_1791486594653.png"
    if os.path.exists(user_face_img):
        with open(user_face_img, "rb") as f:
            face_bytes = f.read()
        with pytest.raises(ValueError) as exc:
            img_ml.predict_image(face_bytes)
        assert "मानव चेहरा पहचाना गया" in str(exc.value) or "Human Face" in str(exc.value)
