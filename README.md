# VetAI — Intelligent Livestock Disease Assistant & Dairy Health Platform

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PWA](https://img.shields.io/badge/PWA-Ready-orange.svg?logo=pwa&logoColor=white)](https://web.dev/progressive-web-apps/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest](https://img.shields.io/badge/Tests-34%20Passing-brightgreen.svg?logo=pytest&logoColor=white)](https://pytest.org)

VetAI is an intelligent, multi-modal clinical advisory platform designed for dairy farmers, rural veterinary workers, and livestock caretakers. It combines tabular symptom inference with computer vision lesion detection, real-time Grad-CAM explainability, out-of-distribution (OOD) human-face filtering, conversational advisory (VetBot), an interactive 48-clinic national veterinary hospital locator, digital herd health rosters with 12-digit Pashu Aadhaar tagging, and bilingual clinical PDF report generation.

---

## Key Capabilities

1. **Multi-Modal Hybrid Diagnostic Engine**
   - **Tabular Symptom Classifier:** Evaluates 29 clinical signs across 15 ruminant diseases using a tuned XGBoost / Random Forest ensemble (95.5% accuracy).
   - **Computer Vision Model:** Fine-tuned MobileNetV2 for lesion detection (Foot and Mouth Disease, Lumpy Skin Disease, and Healthy skin/mucosa).
   - **Late Fusion:** Weighted Bayesian late-fusion ($w_{\text{sym}} = 0.6, w_{\text{img}} = 0.4$) combining symptom probabilities with visual predictions.
   - **Explainable AI (XAI):** Generates Grad-CAM visual heatmaps highlighting affected tissue lesions.

2. **Strict Face & Out-of-Distribution (OOD) Guard**
   - Integrated **OpenCV YuNet** deep learning face detector (232 KB ONNX model) that detects human selfies in ~35ms with zero false positives on cattle coats.
   - Rejects non-livestock objects, blank/dark pictures, or low-confidence ambiguous inputs with polite, clear bilingual guidance.

3. **Interactive Veterinary Hospital Directory & Maps**
   - Leaflet/OpenStreetMap interface plotting **48 verified veterinary hospitals, polyclinics, and dairy union emergency centers** across India (including 20 major hubs in Gujarat: Anand/Amul, Dudhsagar, Sumul, Banas, Sabar, Junagadh, etc.).
   - Geolocation-based Haversine distance sorting, emergency filter, and direct 1962 Animal Ambulance integration.

4. **Digital Herd Roster & Pashu Aadhaar Health Tracker**
   - Register individual cattle with official 12-digit Pashu Aadhaar ear tags.
   - Log vaccination histories (FMD, HS, BQ, Theileriosis, Brucellosis) and deworming schedules with automated overdue alerts.

5. **VetBot AI Assistant & Offline PWA**
   - Natural language conversational assistant in Hindi and English for feeding, calf care, first-aid, and seasonal prevention.
   - Full Progressive Web App (PWA) with service worker caching, offline shell resilience, and installability on mobile devices.

6. **Bilingual Clinical PDF Reports**
   - Generates downloadable clinical PDF diagnostic reports with embedded Devanagari typography (Mukta / Noto Sans Devanagari), triage alerts, and veterinarian notes.

---

## System Architecture

```
vetai/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI server & route handlers
│   │   ├── accounts.py              # JWT authentication & password hashing
│   │   ├── db.py                    # Thread-safe SQLite persistence
│   │   ├── chatbot.py               # VetBot AI conversational knowledge engine
│   │   ├── health_tracker.py        # Vaccination & deworming tracker
│   │   ├── vets.py                  # 48-clinic veterinary hospital directory
│   │   ├── ml/
│   │   │   ├── hybrid.py            # Late-fusion multi-modal diagnostic engine
│   │   │   ├── image_predict.py     # Vision classifier, Grad-CAM & YuNet face guard
│   │   │   ├── predict.py           # Symptom ML classifier (XGBoost/Random Forest)
│   │   │   ├── report.py            # Devanagari bilingual PDF generation
│   │   │   ├── diseases.json        # 15-disease clinical knowledge base
│   │   │   ├── face_detection_yunet.onnx  # Fast YuNet face detector model
│   │   │   ├── image_model.keras    # Fine-tuned MobileNetV2 vision weights
│   │   │   └── symptom_model.joblib # Trained XGBoost symptom classifier
│   │   └── static/fonts/            # Unicode Devanagari font files for PDF
│   ├── data/
│   │   └── kaggle_animal_disease.csv # Clinical tabular dataset
│   ├── scripts/
│   │   └── generate_pwa_icons.py    # PWA icon generator
│   ├── tests/                       # 34 integration and ML unit tests
│   ├── requirements.txt             # Core backend dependencies
│   ├── requirements-image.txt       # TensorFlow & computer vision dependencies
│   ├── train_symptom_model.py       # Symptom ML training & benchmark script
│   └── train_image_model.py         # Cattle vision model training script
├── frontend/
│   ├── index.html                   # Responsive Single-Page Application (HTML5/CSS3/Vanilla JS)
│   ├── sw.js                        # Progressive Web App Service Worker (v2)
│   ├── manifest.json                # Web App Manifest
│   └── icon-192.png, icon-512.png   # PWA app icons
├── run.bat                          # One-click Windows startup script
├── run.sh                           # One-click Linux/Mac startup script
├── PROJECT_REPORT.md                # Comprehensive academic project report
├── PROJECT_FACULTY_PRESENTATION.md  # Viva & faculty defense presentation guide
└── README.md
```

---

## Quick Start Guide

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Web browser (Chrome, Edge, Firefox, or Safari)

### Windows (One-Click)
Double-click `run.bat` or open PowerShell in the project directory:
```powershell
.\run.bat
```
Open **http://127.0.0.1:8000** in your browser.

### Linux / macOS
```bash
chmod +x run.sh
./run.sh
```
Open **http://127.0.0.1:8000** in your browser.

---

## Running Automated Tests

VetAI includes a complete automated test suite covering API endpoints, authentication, health tracking, conversational logic, and ML validation:

```bash
pytest backend/tests
```
All **34 test cases** pass with zero regressions.

---

## Disease Coverage

VetAI covers 15 key ruminant diseases impacting Indian dairy and livestock:

| Disease ID | English Name | हिंदी नाम | Risk Level | Primary Species |
|---|---|---|---|---|
| `fmd` | Foot and Mouth Disease | खुरपका-मुँहपका (FMD) | **High** | Cow, Buffalo, Sheep, Goat |
| `lsd` | Lumpy Skin Disease | लंपी त्वचा रोग (LSD) | **High** | Cow, Buffalo |
| `mastitis` | Mastitis | थनैला रोग | Medium | Cow, Buffalo, Goat |
| `bloat` | Bloat / Tympany | अफारा (पेट फूलना) | **High** | Cow, Buffalo |
| `hs` | Haemorrhagic Septicaemia | गलघोंटू (HS) | **High** | Cow, Buffalo |
| `bq` | Black Quarter | लंगड़ा बुखार (BQ) | **High** | Cow, Buffalo, Sheep |
| `bvd` | Bovine Viral Diarrhea | गोजातीय वायरल डायरिया | Medium | Cattle |
| `brd` | Bovine Respiratory Disease | श्वसन रोग (BRD) | Medium | Cattle |
| `anthrax` | Anthrax | गिल्टी रोग / एंथ्रेक्स | **High** | All Livestock |
| `enterotoxemia` | Enterotoxemia | फिड़किया रोग | **High** | Sheep, Goat |
| `ppr` | Peste des Petits Ruminants | बकरी प्लेग (PPR) | **High** | Goat, Sheep |
| `ketosis` | Ketosis | कीटोसिस (उपापचयी रोग) | Medium | Dairy Cattle |
| `milk_fever` | Milk Fever | दुग्ध ज्वर (हाइपोकैल्सीमिया) | Medium | Fresh Milking Cows |
| `parasites` | Internal Parasites | पेट के कीड़े (परजीवी) | Low | All Livestock |
| `coccidiosis` | Coccidiosis | खूनी पेचिश (कॉक्सीडियोसिस) | Medium | Calves, Kids, Lambs |

---

## Contributing & License

Contributions, bug reports, and suggestions are welcome.
This project is licensed under the **MIT License**.

> **Clinical Disclaimer:** VetAI is intended as an informational, educational, and first-line clinical triage decision-support tool. It does not replace the diagnosis and treatment prescribed by a registered veterinary practitioner. Always consult a licensed veterinarian for emergency animal treatment.
