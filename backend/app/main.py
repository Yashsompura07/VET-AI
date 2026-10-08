"""VetAI backend — FastAPI app serving the prediction API and the web page."""

import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional

from app.ml import predict as ml
from app.ml import image_predict as img_ml
from app.ml import hybrid as hybrid_ml
from app.ml import report as report_ml
from app import db, accounts, chatbot, vets, health_tracker

db.init_db()

HERE = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(HERE, "..", "..", "frontend")

app = FastAPI(title="VetAI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    animal_type: str = Field(..., examples=["cow"])
    age_months: Optional[int] = 0
    weight_kg: Optional[int] = 0
    symptoms: List[str] = Field(default_factory=list)
    temp_c: Optional[float] = None
    heart_rate: Optional[int] = None


@app.get("/api/health")
def health():
    return {"status": "ok", **ml.get_model_info()}


@app.get("/api/symptoms")
def symptoms():
    return ml.get_symptoms()


@app.get("/api/diseases")
def diseases():
    return ml.get_diseases()


@app.get("/api/diseases/{disease_id}")
def disease(disease_id: str):
    d = ml.get_disease(disease_id)
    if not d:
        raise HTTPException(status_code=404, detail="Disease not found")
    return d


@app.post("/api/predict")
@app.post("/api/predict/symptoms")
def predict_symptoms(req: PredictRequest):
    if not req.symptoms:
        raise HTTPException(status_code=400, detail="Please select at least one symptom.")
    return ml.predict(req.animal_type, req.age_months, req.weight_kg, req.symptoms,
                      temp_c=req.temp_c, heart_rate=req.heart_rate)


@app.get("/api/image-status")
def image_status():
    return {"available": img_ml.is_available()}


@app.post("/api/predict/image")
async def predict_image(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image file.")
    return img_ml.predict_image(data)


@app.post("/api/predict/hybrid")
async def predict_hybrid(
    animal_type: str = Form(...),
    symptoms: str = Form(""),
    age_months: Optional[int] = Form(0),
    weight_kg: Optional[int] = Form(0),
    temp_c: Optional[float] = Form(None),
    heart_rate: Optional[int] = Form(None),
    animal_id: Optional[int] = Form(None),
    file: Optional[UploadFile] = File(None),
    user=Depends(accounts.optional_user),
):
    symptom_list = [s for s in symptoms.split(",") if s.strip()]
    image_bytes = None
    if file is not None:
        try:
            image_bytes = await file.read()
        except Exception:
            image_bytes = None

    if not symptom_list and (not image_bytes or len(image_bytes) == 0):
        raise HTTPException(
            status_code=400, 
            detail="कृपया कम से कम एक लक्षण चुनें या घाव/त्वचा की फोटो अपलोड करें। (Please select at least one symptom or upload a photo.)"
        )
    try:
        result = hybrid_ml.predict_hybrid(
            animal_type, age_months, weight_kg, symptom_list,
            temp_c=temp_c, heart_rate=heart_rate, image_bytes=image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"नैदानिक विश्लेषण में त्रुटि (Diagnostic error): {str(e)}"
        )

    # If the user is logged in, save this diagnosis to their history safely.
    if user:
        try:
            db.save_prediction(user["id"], animal_id, result)
            result["saved"] = True
        except Exception:
            pass
    return result


# ---------- Chatbot ----------
class ChatReq(BaseModel):
    message: str
    context: Optional[dict] = None
    history: Optional[List[dict]] = None


@app.post("/api/chat")
def chat_endpoint(req: ChatReq):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    return chatbot.ask_vetbot(req.message.strip(), context=req.context, history=req.history)


@app.get("/api/chat/engine-status")
def chat_engine_status():
    return chatbot.check_ollama_status()


# ---------- Veterinary Directory & Maps ----------
@app.get("/api/vets")
@app.get("/api/vets/nearby")
def get_vets(q: Optional[str] = None, lat: Optional[float] = None, lon: Optional[float] = None, emergency: bool = False):
    return vets.search_vets(query=q, user_lat=lat, user_lon=lon, emergency_only=emergency)


# ---------- Accounts ----------
class RegisterReq(BaseModel):
    name: str
    phone: str
    password: str = Field(min_length=4)


class LoginReq(BaseModel):
    phone: str
    password: str


def _public_user(u):
    return {"id": u["id"], "name": u["name"], "phone": u["phone"]}


@app.post("/api/auth/register")
def register(req: RegisterReq):
    if db.get_user_by_phone(req.phone):
        raise HTTPException(status_code=409, detail="That phone number is already registered.")
    pw_hash, salt = accounts.hash_password(req.password)
    uid = db.create_user(req.name.strip(), req.phone.strip(), pw_hash, salt)
    u = db.get_user(uid)
    return {"token": accounts.make_token(uid), "user": _public_user(u)}


@app.post("/api/auth/login")
def login(req: LoginReq):
    u = db.get_user_by_phone(req.phone.strip())
    if not u or not accounts.verify_password(req.password, u["pw_hash"], u["salt"]):
        raise HTTPException(status_code=401, detail="Wrong phone number or password.")
    return {"token": accounts.make_token(u["id"]), "user": _public_user(u)}


class ResetPasswordReq(BaseModel):
    phone: str
    new_password: str = Field(min_length=4)


@app.post("/api/auth/reset-password")
def reset_password(req: ResetPasswordReq):
    u = db.get_user_by_phone(req.phone.strip())
    if not u:
        raise HTTPException(status_code=404, detail="इस फोन नंबर से कोई खाता पंजीकृत नहीं है। (No account found with this phone number.)")
    pw_hash, salt = accounts.hash_password(req.new_password)
    with db._lock, db._conn() as c:
        c.execute("UPDATE users SET pw_hash = ?, salt = ? WHERE id = ?", (pw_hash, salt, u["id"]))
    updated_user = db.get_user(u["id"])
    return {
        "status": "ok",
        "message": "पासवर्ड सफलतापूर्वक अपडेट हो गया। (Password reset successful.)",
        "token": accounts.make_token(u["id"]),
        "user": _public_user(updated_user)
    }


@app.get("/api/me")
def me(user=Depends(accounts.current_user)):
    return _public_user(user)


# ---------- Animals ----------
class AnimalReq(BaseModel):
    name: str
    animal_type: str
    tag_number: Optional[str] = None
    breed: Optional[str] = None
    age_months: Optional[int] = None
    weight_kg: Optional[int] = None
    gender: Optional[str] = None


@app.post("/api/animals")
def create_animal(req: AnimalReq, user=Depends(accounts.current_user)):
    data = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    aid = db.add_animal(user["id"], data)
    return db.get_animal(user["id"], aid)


@app.get("/api/animals")
def get_animals(user=Depends(accounts.current_user)):
    return db.list_animals(user["id"])


@app.delete("/api/animals/{animal_id}")
def remove_animal(animal_id: int, user=Depends(accounts.current_user)):
    db.delete_animal(user["id"], animal_id)
    return {"deleted": True}


@app.get("/api/animals/{animal_id}/history")
def animal_history(animal_id: int, user=Depends(accounts.current_user)):
    return db.animal_history(user["id"], animal_id)


# ---------- Animal Health Tracker (Phase 3) ----------
class VaccinationReq(BaseModel):
    vaccine_name: str
    dose_number: Optional[str] = "Routine"
    administered_date: Optional[str] = None
    next_due_date: Optional[str] = None
    batch_number: Optional[str] = None
    veterinarian: Optional[str] = None
    notes: Optional[str] = None


class DewormingReq(BaseModel):
    medicine_name: str
    dose_amount: Optional[str] = None
    administered_date: Optional[str] = None
    next_due_date: Optional[str] = None
    veterinarian: Optional[str] = None
    notes: Optional[str] = None


@app.get("/api/vaccines/standard")
def get_standard_vaccines(animal_type: str = "cow"):
    return {
        "vaccines": health_tracker.get_standard_schedules(animal_type),
        "dewormers": health_tracker.STANDARD_DEWORMERS
    }


@app.get("/api/animals/{animal_id}/health")
def get_animal_health(animal_id: int, user=Depends(accounts.current_user)):
    animal = db.get_animal(user["id"], animal_id)
    if not animal:
        raise HTTPException(status_code=404, detail="Animal not found")
    vacs = db.list_vaccinations(user["id"], animal_id)
    dews = db.list_deworming(user["id"], animal_id)
    vacs_annotated = [health_tracker.compute_health_status(v, "next_due_date") for v in vacs]
    dews_annotated = [health_tracker.compute_health_status(d, "next_due_date") for d in dews]

    overdue_vac = sum(1 for v in vacs_annotated if v.get("status_badge") == "overdue")
    due_soon_vac = sum(1 for v in vacs_annotated if v.get("status_badge") == "due_soon")

    return {
        "animal": animal,
        "vaccinations": vacs_annotated,
        "deworming": dews_annotated,
        "overdue_count": overdue_vac,
        "due_soon_count": due_soon_vac,
        "standard_schedules": health_tracker.get_standard_schedules(animal["animal_type"]),
        "standard_dewormers": health_tracker.STANDARD_DEWORMERS
    }


@app.post("/api/animals/{animal_id}/vaccinations")
def record_vaccination(animal_id: int, req: VaccinationReq, user=Depends(accounts.current_user)):
    animal = db.get_animal(user["id"], animal_id)
    if not animal:
        raise HTTPException(status_code=404, detail="Animal not found")
    data = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    vid = db.add_vaccination(user["id"], animal_id, data)
    return db.get_vaccination(user["id"], vid)


@app.delete("/api/vaccinations/{vac_id}")
def delete_vaccination(vac_id: int, user=Depends(accounts.current_user)):
    db.delete_vaccination(user["id"], vac_id)
    return {"deleted": True}


@app.post("/api/animals/{animal_id}/deworming")
def record_deworming(animal_id: int, req: DewormingReq, user=Depends(accounts.current_user)):
    animal = db.get_animal(user["id"], animal_id)
    if not animal:
        raise HTTPException(status_code=404, detail="Animal not found")
    data = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    did = db.add_deworming(user["id"], animal_id, data)
    return {"id": did, "message": "Deworming record logged"}


@app.delete("/api/deworming/{dew_id}")
def delete_deworming(dew_id: int, user=Depends(accounts.current_user)):
    db.delete_deworming(user["id"], dew_id)
    return {"deleted": True}


@app.get("/api/dashboard")
def dashboard(user=Depends(accounts.current_user)):
    return db.dashboard_stats(user["id"])


@app.post("/api/report/pdf")
async def report_pdf(payload: dict):
    try:
        pdf_bytes = report_ml.build_pdf(payload)
    except ModuleNotFoundError:
        raise HTTPException(status_code=503,
                            detail="PDF support not installed (pip install fpdf2).")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

    fname = "vetai_rog_nidan_report.pdf" if payload.get("lang") == "hi" else "vetai_clinical_report.pdf"
    return Response(
        content=pdf_bytes, media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={fname}"})


# ---- Serve the frontend & PWA ----
@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/manifest.json")
def manifest():
    p = os.path.join(FRONTEND_DIR, "manifest.json")
    if os.path.exists(p):
        return FileResponse(p, media_type="application/manifest+json")
    raise HTTPException(status_code=404, detail="Manifest not found")


@app.get("/sw.js")
def service_worker():
    p = os.path.join(FRONTEND_DIR, "sw.js")
    if os.path.exists(p):
        return FileResponse(
            p,
            media_type="application/javascript",
            headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"}
        )
    raise HTTPException(status_code=404, detail="Service worker not found")


if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
