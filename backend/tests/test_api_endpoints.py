"""Full integration test suite for VetAI HTTP API endpoints using FastAPI TestClient."""
import uuid
import random
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_api_health():
    """Verify GET /api/health returns ok status and model info."""
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "metrics" in data


def test_api_symptoms():
    """Verify GET /api/symptoms returns all 29 symptoms."""
    r = client.get("/api/symptoms")
    assert r.status_code == 200
    symptoms = r.json()
    assert len(symptoms) == 29
    assert all("id" in s and "label" in s for s in symptoms)


def test_api_diseases():
    """Verify GET /api/diseases returns all 15 diseases and /api/diseases/{id} works."""
    r = client.get("/api/diseases")
    assert r.status_code == 200
    diseases = r.json()
    assert len(diseases) == 15

    # Test single disease
    r_fmd = client.get("/api/diseases/fmd")
    assert r_fmd.status_code == 200
    assert r_fmd.json()["id"] == "fmd"

    # Test 404
    r_bad = client.get("/api/diseases/unknown_disease_id")
    assert r_bad.status_code == 404


def test_api_predict():
    """Verify POST /api/predict returns ranked diagnosis."""
    payload = {
        "animal_type": "cow",
        "age_months": 36,
        "weight_kg": 400,
        "symptoms": ["mouth_lesions", "salivation", "hoof_lesions", "lameness", "fever"]
    }
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "predictions" in data
    assert len(data["predictions"]) > 0
    assert data["top"]["disease_id"] == "fmd"
    assert "explanation" in data


def test_api_chat():
    """Verify POST /api/chat returns conversational advisory."""
    # English query
    r_en = client.post("/api/chat", json={"message": "What should I feed a cow with bloat?"})
    assert r_en.status_code == 200
    d_en = r_en.json()
    assert "reply" in d_en
    assert len(d_en["reply"]) > 20

    # Hindi query
    r_hi = client.post("/api/chat", json={"message": "गाय को अफारा है क्या करें"})
    assert r_hi.status_code == 200
    d_hi = r_hi.json()
    assert "reply" in d_hi
    assert d_hi.get("disease_identified") == "bloat"


def test_api_chat_engine_status():
    """Verify GET /api/chat/engine-status reports LLM mode and device info."""
    r = client.get("/api/chat/engine-status")
    assert r.status_code == 200
    data = r.json()
    assert "available" in data
    assert "mode" in data
    assert "device" in data


def test_api_vets_nearby():
    """Verify GET /api/vets/nearby returns clinics and emergency hotlines."""
    r = client.get("/api/vets/nearby?lat=28.6139&lon=77.2090")
    assert r.status_code == 200
    data = r.json()
    assert "clinics" in data
    assert "hotlines" in data
    assert len(data["clinics"]) > 0
    assert any(h["phone"] == "1962" for h in data["hotlines"])


def test_api_pwa_routes():
    """Verify PWA manifest and service worker routes."""
    # Manifest
    r_m = client.get("/manifest.json")
    assert r_m.status_code == 200
    assert "application/manifest+json" in r_m.headers["content-type"]
    manifest = r_m.json()
    assert manifest["short_name"] == "VetAI"
    assert len(manifest["icons"]) >= 2

    # Service Worker
    r_sw = client.get("/sw.js")
    assert r_sw.status_code == 200
    assert "application/javascript" in r_sw.headers["content-type"]
    assert "Service-Worker-Allowed" in r_sw.headers
    assert r_sw.headers["Service-Worker-Allowed"] == "/"


def test_user_lifecycle_and_pashu_aadhaar():
    """Verify registration, login, Pashu Aadhaar tag, vaccination & health summary."""
    rand_phone = f"98{random.randint(10000000, 99999999)}"
    password = "TestPassword123!"

    # 1. Register with phone
    r_reg = client.post("/api/auth/register", json={
        "name": "Ramesh Kumar",
        "phone": rand_phone,
        "password": password
    })
    assert r_reg.status_code == 200
    token = r_reg.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Add animal with 12-digit Pashu Aadhaar tag
    tag_num = f"1200{random.randint(10000000, 99999999)}"
    r_animal = client.post("/api/animals", headers=headers, json={
        "name": "Kamdhenu",
        "animal_type": "cow",
        "breed": "Gir",
        "age_months": 42,
        "weight_kg": 440,
        "tag_number": tag_num
    })
    assert r_animal.status_code == 200
    animal = r_animal.json()
    animal_id = animal["id"]
    assert animal["tag_number"] == tag_num

    # 3. Add vaccination record
    r_vac = client.post(f"/api/animals/{animal_id}/vaccinations", headers=headers, json={
        "vaccine_name": "FMD Vaccine (Raksha-Ovac)",
        "administered_date": "2026-05-15",
        "next_due_date": "2026-11-15",
        "batch_number": "ROV-8821",
        "veterinarian": "Dr. Sharma"
    })
    assert r_vac.status_code == 200
    vac_id = r_vac.json()["id"]

    # 4. Add deworming record
    r_dew = client.post(f"/api/animals/{animal_id}/deworming", headers=headers, json={
        "medicine_name": "Albendazole Suspension",
        "administered_date": "2026-06-01",
        "next_due_date": "2026-09-01",
        "dose_amount": "60 ml oral",
        "veterinarian": "Self"
    })
    assert r_dew.status_code == 200

    # 5. Fetch health summary
    r_health = client.get(f"/api/animals/{animal_id}/health", headers=headers)
    assert r_health.status_code == 200
    h_data = r_health.json()
    assert h_data["animal"]["tag_number"] == tag_num
    assert len(h_data["vaccinations"]) >= 1
    assert len(h_data["deworming"]) >= 1

    # 6. Fetch dashboard stats
    r_dash = client.get("/api/dashboard", headers=headers)
    assert r_dash.status_code == 200
    dash = r_dash.json()
    assert dash["total_animals"] >= 1
    assert dash["total_vaccines"] >= 1
    assert "vaccines_overdue" in dash

    # 7. Cleanup
    client.delete(f"/api/vaccinations/{vac_id}", headers=headers)
    client.delete(f"/api/animals/{animal_id}", headers=headers)


def test_api_report_pdf():
    """Verify POST /api/report/pdf generates valid binary PDF with magic header %PDF."""
    payload = {
        "animal": {
            "name": "Nandi",
            "animal_type": "cow",
            "breed": "Sahiwal",
            "age_months": 36,
            "weight_kg": 450,
            "tag_number": "120045829104"
        },
        "diagnosis": {
            "disease_id": "fmd",
            "name": "Foot and Mouth Disease",
            "confidence": 94.5,
            "risk_level": "high",
            "explanation": "Blisters on mouth and hooves, high fever and salivation."
        }
    }
    r = client.post("/api/report/pdf", json=payload)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")
