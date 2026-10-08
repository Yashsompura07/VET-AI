"""VetAI Hybrid Engine — late fusion of the symptom model and the image model.

This is the core novelty of the project. It takes two independent predictions:
  - the symptom model's probability over ALL 15 diseases
  - the image model's probability over {healthy, fmd, lsd}  (cattle only)
and fuses them into one combined diagnosis.

FUSION METHOD (weighted late fusion):
  1. Build a full-length image distribution over the 15 diseases:
       - fmd  -> image P(fmd)
       - lsd  -> image P(lsd)
       - the image's "healthy" probability mass is spread across the remaining
         diseases in proportion to the symptom model's belief, because a "healthy
         skin" photo means "not FMD/LSD" and the symptoms decide what else it is.
  2. combined = w_sym * P_symptom + w_img * P_image_full, then normalise.

SPECIES GUARD:
  The image model was trained on cattle only, so it is fused ONLY for cow/buffalo.
  For goat/sheep the result is symptom-only, clearly reported as such.
"""

from app.ml import predict as sym_ml
from app.ml import image_predict as img_ml

CATTLE = {"cow", "buffalo"}
IMAGE_CLASSES_AS_DISEASE = {"fmd": "fmd", "lsd": "lsd"}  # 'healthy' handled separately

W_SYM = 0.6   # weight on the symptom model
W_IMG = 0.4   # weight on the image model


def _fuse(sym_proba, img_proba):
    """Combine a symptom distribution (over 15 diseases) with an image
    distribution (over healthy/fmd/lsd). Returns a fused {disease_id: prob}."""
    diseases = list(sym_proba.keys())
    others = [d for d in diseases if d not in IMAGE_CLASSES_AS_DISEASE]
    sum_others = sum(sym_proba[d] for d in others) or 1.0

    img_full = {}
    for d in diseases:
        if d in IMAGE_CLASSES_AS_DISEASE:
            img_full[d] = img_proba.get(d, 0.0)
        else:
            # channel the "healthy" mass to non-FMD/LSD diseases per the symptoms
            img_full[d] = img_proba.get("healthy", 0.0) * (sym_proba[d] / sum_others)

    combined = {d: W_SYM * sym_proba[d] + W_IMG * img_full[d] for d in diseases}
    total = sum(combined.values()) or 1.0
    return {d: v / total for d, v in combined.items()}


def _rank(proba, top_k=3):
    ranked = sorted(proba.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    out = []
    for disease_id, p in ranked:
        d = sym_ml.get_disease(disease_id)
        out.append({
            "disease_id": disease_id,
            "name": d["name"],
            "confidence": round(p * 100, 1),
            "risk_level": d["risk_level"],
        })
    return out


def predict_hybrid(animal_type, age_months, weight_kg, symptoms,
                   temp_c=None, heart_rate=None, image_bytes=None):
    have_symptoms = bool(symptoms and len(symptoms) > 0)
    is_cattle = animal_type in CATTLE
    have_image = image_bytes is not None and len(image_bytes) > 0
    image_usable = have_image and is_cattle and img_ml.is_available()

    # Case A: No symptoms provided -> image-only diagnosis
    if not have_symptoms:
        if not have_image:
            raise ValueError("Please select at least one symptom or upload a photo.")
        if not is_cattle:
            raise ValueError("Photo detection is trained on cattle (cow/buffalo) only. For sheep and goats, please select symptoms.")
        if not img_ml.is_available():
            raise ValueError("Photo model is not available. Please install dependencies or select symptoms.")

        img = img_ml.predict_image(image_bytes)
        mode = "image_only"
        top_class = img["top_class"]
        is_healthy = img["is_healthy"]

        predictions = []
        for item in img["ranked"]:
            cls = item["class"]
            conf = item["confidence"]
            if cls == "healthy":
                predictions.append({
                    "disease_id": "healthy",
                    "name": "Healthy (No visible lesions detected)",
                    "confidence": conf,
                    "risk_level": "low",
                })
            else:
                d = sym_ml.get_disease(cls)
                predictions.append({
                    "disease_id": cls,
                    "name": d["name"] if d else cls.upper(),
                    "confidence": conf,
                    "risk_level": d["risk_level"] if d else "medium",
                })

        top = predictions[0]
        if is_healthy:
            top_disease = {
                "id": "healthy",
                "name": "Healthy (No visible lesions detected)",
                "risk_level": "low",
                "key_symptoms": [],
                "definition": "Visual inspection did not identify blisters, nodules, or sores characteristic of Foot and Mouth Disease (FMD) or Lumpy Skin Disease (LSD).",
                "causes": "N/A — Animal appears visibly free of key skin/mucosal lesions.",
                "transmission": "N/A",
                "prevention": "Maintain regular biosecurity, clean water, and standard herd vaccination schedules.",
                "diet": "Provide standard balanced nutritional feed and clean drinking water.",
                "supportive_care": "Routine herd health monitoring. Watch for general changes in appetite or milk yield.",
                "treatment_overview": "No medical treatment necessary. Animal appears visibly healthy.",
                "recovery_time": "N/A",
            }
            summary = f"Photo analysis indicates healthy skin/mucosa ({top['confidence']}% confidence). No visual lesions detected."
        else:
            top_disease = sym_ml.get_disease(top_class)
            summary = f"Visual detection identified signs of {top['name']} ({top['confidence']}% confidence) from the uploaded photo."

        components = {
            "symptom": None,
            "image": {
                "class": top_class,
                "confidence": img["confidence"],
                "is_healthy": is_healthy,
            }
        }
        explanation = {
            "matched_symptoms": [],
            "summary": summary,
        }

        try:
            from app.chatbot import DISEASES_HI
            k_hi = DISEASES_HI.get(top.get("id") or top.get("disease_id"), {})
        except Exception:
            k_hi = {}

        return {
            "mode": mode,
            "predictions": predictions,
            "top": top,
            "knowledge": top_disease,
            "knowledge_hi": k_hi,
            "explanation": explanation,
            "components": components,
            "agreement": None,
            "inputs_used": ["photo"],
            "heatmap": img.get("heatmap"),
        }

    # Case B: Symptoms provided (+ optional image)
    # 1. Always get the symptom distribution
    sym_proba = sym_ml.predict_proba_full(
        animal_type, age_months, weight_kg, symptoms, temp_c, heart_rate)
    sym_top_id = max(sym_proba, key=sym_proba.get)

    # 2. Decide the mode
    if image_usable:
        try:
            img = img_ml.predict_image(image_bytes)
            fused = _fuse(sym_proba, img["proba"])
            mode = "hybrid"
        except ValueError as e:
            err_str = str(e)
            # Re-raise hard validation violations (human face, non-livestock objects)
            if "मानव चेहरा" in err_str or "Human Face" in err_str or "गैर-पशु" in err_str or "Non-livestock" in err_str:
                raise
            # If photo lesion features were unclear or ambiguous, fall back safely to symptoms
            fused = sym_proba
            img = None
            mode = "symptoms_only_unclear_image"
    else:
        fused = sym_proba
        img = None
        if have_image and not is_cattle:
            mode = "symptoms_only_non_cattle"
        elif have_image and not img_ml.is_available():
            mode = "symptoms_only_no_image_model"
        else:
            mode = "symptoms_only"

    predictions = _rank(fused, top_k=3)
    top = predictions[0]
    top_disease = sym_ml.get_disease(top["disease_id"])

    # 3. Agreement analysis (for the explanation and the report)
    agreement = None
    components = {"symptom": None, "image": None}
    sym_d = sym_ml.get_disease(sym_top_id)
    components["symptom"] = {
        "name": sym_d["name"],
        "confidence": round(sym_proba[sym_top_id] * 100, 1),
    }
    if img is not None:
        img_label = img["top_class"]
        components["image"] = {
            "class": img_label,
            "confidence": img["confidence"],
            "is_healthy": img["is_healthy"],
        }
        if img_label == "healthy":
            agreement = ("agree" if sym_top_id not in ("fmd", "lsd")
                         else "conflict")
        else:
            agreement = "agree" if img_label == sym_top_id else "conflict"

    # 4. Explanation
    selected = set(symptoms or [])
    matched = [s for s in top_disease["key_symptoms"] if s in selected]
    explanation = {
        "matched_symptoms": [
            {"id": s, "label": sym_ml._symptom_label.get(s, s)} for s in matched
        ],
        "summary": _summary(mode, top, components, agreement),
    }

    try:
        from app.chatbot import DISEASES_HI
        k_hi = DISEASES_HI.get(top.get("id") or top.get("disease_id"), {})
    except Exception:
        k_hi = {}

    result = {
        "mode": mode,
        "predictions": predictions,
        "top": top,
        "knowledge": top_disease,
        "knowledge_hi": k_hi,
        "explanation": explanation,
        "components": components,
        "agreement": agreement,
        "inputs_used": (["symptoms", "photo"] if mode == "hybrid" else ["symptoms"]),
    }
    if img is not None:
        result["heatmap"] = img["heatmap"]
    return result


def _summary(mode, top, components, agreement):
    if mode == "image_only":
        return f"{top['name']} ({top['confidence']}% confidence), identified directly from the uploaded photo."
    if mode == "hybrid":
        base = (f"Combined diagnosis: {top['name']} ({top['confidence']}% confidence), "
                f"fusing symptoms and the photo.")
        if agreement == "agree":
            return base + " Both the symptom and image analyses point the same way, which raises confidence."
        if agreement == "conflict":
            return (base + f" Note: the symptom analysis suggested {components['symptom']['name']} "
                    f"while the photo suggested '{components['image']['class']}' — the combined result "
                    f"weighs both. Consider a veterinary check.")
        return base
    if mode == "symptoms_only_unclear_image":
        return (f"{top['name']} ({top['confidence']}% confidence), from symptoms. "
                "Uploaded photo had ambiguous features or low confidence, so diagnosis relied on reported symptoms.")
    if mode == "symptoms_only_non_cattle":
        return (f"{top['name']} ({top['confidence']}% confidence), from symptoms. "
                f"Photo detection is currently trained on cattle only, so it was not used for this animal.")
    if mode == "symptoms_only_no_image_model":
        return (f"{top['name']} ({top['confidence']}% confidence), from symptoms. "
                f"(The photo model is not installed, so only symptoms were used.)")
    return f"{top['name']} ({top['confidence']}% confidence), based on the reported symptoms."
