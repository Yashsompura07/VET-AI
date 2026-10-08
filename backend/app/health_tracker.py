"""Animal Health Tracker Engine (Phase 3)
Standard Indian & Global Veterinary Vaccination Schedules & Deworming Calendars
Compliant with ICAR (Indian Council of Agricultural Research) & DAHD Guidelines.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional

# Standard ICAR & Government of India Livestock Vaccination Calendar
STANDARD_VACCINES: Dict[str, List[Dict[str, Any]]] = {
    "cattle_buffalo": [
        {
            "id": "fmd",
            "name": "Foot and Mouth Disease (FMD / Khurpaka-Muhpaka)",
            "disease_id": "fmd",
            "target_species": ["cow", "buffalo"],
            "initial_age": "3 to 4 months",
            "frequency": "Biannual (Every 6 months — Pre-monsoon & Post-monsoon)",
            "interval_days": 180,
            "season": "May/June (Pre-monsoon) & November/December (Post-monsoon)",
            "description": "Essential under National Animal Disease Control Programme (NADCP). Protects against O, A, Asia-1 strains.",
            "booster": "Required 4 weeks after primary dose in calves, then every 6 months."
        },
        {
            "id": "hs",
            "name": "Haemorrhagic Septicaemia (HS / Galghotu)",
            "disease_id": "hs",
            "target_species": ["cow", "buffalo"],
            "initial_age": "6 months onwards",
            "frequency": "Annual (Every 12 months)",
            "interval_days": 365,
            "season": "May to June (Strictly before the onset of monsoon rains)",
            "description": "Critical bacterial disease with high mortality. Mandatory before seasonal humidity and monsoon.",
            "booster": "Annual revaccination."
        },
        {
            "id": "bq",
            "name": "Black Quarter (BQ / Chuchia-Langda)",
            "disease_id": "bq",
            "target_species": ["cow", "buffalo"],
            "initial_age": "6 months onwards",
            "frequency": "Annual (Every 12 months)",
            "interval_days": 365,
            "season": "May to June (Often given as combined HS + BQ vaccine)",
            "description": "Severe clostridial muscular infection causing high fever and crepitating thigh swelling.",
            "booster": "Annual revaccination."
        },
        {
            "id": "brucellosis",
            "name": "Brucellosis (Calfhood S19 / Strain 19)",
            "disease_id": "brucellosis",
            "target_species": ["cow", "buffalo"],
            "initial_age": "4 to 8 months (Female calves only)",
            "frequency": "Once in a Lifetime (Permanent Immunity)",
            "interval_days": None,
            "season": "Any time when female calf reaches 4–8 months of age",
            "description": "Strictly for female heifer calves. Prevents mid-to-late abortion storms and protects human handlers from undulant fever.",
            "booster": "None required (Lifelong immunity). Do NOT vaccinate adult or male animals."
        },
        {
            "id": "lsd",
            "name": "Lumpy Skin Disease (LSD / Lumpi-ProVacInd / Goat Pox Vaccine)",
            "disease_id": "lsd",
            "target_species": ["cow", "buffalo"],
            "initial_age": "4 months onwards",
            "frequency": "Annual (Every 12 months)",
            "interval_days": 365,
            "season": "April to May (Before vector fly & mosquito boom)",
            "description": "Prevents cutaneous nodule eruption, severe milk loss, and secondary bacterial infections.",
            "booster": "Annual revaccination."
        },
        {
            "id": "theileriosis",
            "name": "Theileriosis (Raktdhar / Tick-Borne)",
            "disease_id": "theileriosis",
            "target_species": ["cow"],
            "initial_age": "2 to 3 months (Crossbred / Exotic cattle)",
            "frequency": "Once in lifetime / Annual in high-tick endemic regions",
            "interval_days": 365,
            "season": "Spring / Summer before tick peak",
            "description": "Schizont cell culture vaccine highly recommended for high-yielding crossbred HF and Jersey cattle.",
            "booster": "Annual or single dose as per regional veterinary advice."
        },
        {
            "id": "anthrax",
            "name": "Anthrax Spore Vaccine",
            "disease_id": "anthrax",
            "target_species": ["cow", "buffalo"],
            "initial_age": "4 months onwards",
            "frequency": "Annual in endemic zones",
            "interval_days": 365,
            "season": "Pre-monsoon (April/May)",
            "description": "Given only in designated anthrax-endemic districts. Do not vaccinate animals undergoing antibiotic therapy.",
            "booster": "Annual revaccination."
        }
    ],
    "sheep_goat": [
        {
            "id": "ppr",
            "name": "Peste des Petits Ruminants (PPR / Goat Plague)",
            "disease_id": "ppr",
            "target_species": ["goat", "sheep"],
            "initial_age": "3 months onwards",
            "frequency": "Once every 3 years (36 months)",
            "interval_days": 1095,
            "season": "Throughout the year (Avoid pregnant does/ewes in last month)",
            "description": "Highest priority vaccine for small ruminants. Grants robust 3-year protection against viral pneumonia-stomatitis.",
            "booster": "Revaccinate every 3 years."
        },
        {
            "id": "et",
            "name": "Enterotoxaemia (ET / Pulpy Kidney)",
            "disease_id": "et",
            "target_species": ["goat", "sheep"],
            "initial_age": "4 months onwards",
            "frequency": "Annual (Every 12 months)",
            "interval_days": 365,
            "season": "May/June before monsoon and lush green flush",
            "description": "Prevents rapid clostridial epsilon-toxin death triggered by sudden high-protein / lush green grazing.",
            "booster": "Annual revaccination."
        },
        {
            "id": "pox",
            "name": "Sheep Pox & Goat Pox",
            "disease_id": "pox",
            "target_species": ["goat", "sheep"],
            "initial_age": "3 months onwards",
            "frequency": "Annual (Every 12 months)",
            "interval_days": 365,
            "season": "December to January (Winter months)",
            "description": "Prevents severe papular-vesicular pock lesions on wool-less body skin, snout, and teats.",
            "booster": "Annual revaccination."
        },
        {
            "id": "fmd_small",
            "name": "Foot and Mouth Disease (FMD — Sheep/Goat Dose)",
            "disease_id": "fmd",
            "target_species": ["goat", "sheep"],
            "initial_age": "4 months onwards",
            "frequency": "Biannual (Every 6 months)",
            "interval_days": 180,
            "season": "Pre-monsoon and post-monsoon",
            "description": "Essential in mixed-species farms to stop sheep and goats from acting as subclinical FMD viral carriers.",
            "booster": "Biannual revaccination."
        }
    ]
}

# Standard Deworming Protocol
STANDARD_DEWORMERS: List[Dict[str, Any]] = [
    {
        "id": "albendazole",
        "name": "Albendazole (Broad-spectrum roundworms & tapeworms)",
        "dosage_guide": "5–10 mg/kg body weight",
        "repeat_interval_days": 90,
        "contraindications": "Avoid in the first 45 days of pregnancy in cattle.",
        "best_season": "Pre-monsoon (June) and Post-monsoon (October)"
    },
    {
        "id": "fenbendazole",
        "name": "Fenbendazole (Safe for pregnant dairy animals)",
        "dosage_guide": "5–7.5 mg/kg body weight",
        "repeat_interval_days": 90,
        "contraindications": "Safe in all trimesters of pregnancy.",
        "best_season": "Mid-summer or rotation after Albendazole"
    },
    {
        "id": "ivermectin",
        "name": "Ivermectin (Internal parasites & external ticks/mange)",
        "dosage_guide": "0.2 mg/kg body weight (Subcutaneous)",
        "repeat_interval_days": 120,
        "contraindications": "Follow milk withdrawal period as directed by veterinarian.",
        "best_season": "Spring and autumn tick peaks"
    },
    {
        "id": "oxyclozanide",
        "name": "Oxyclozanide + Levamisole (Liver fluke & amphistomes)",
        "dosage_guide": "10–15 mg/kg body weight",
        "repeat_interval_days": 90,
        "contraindications": "Crucial in canal-irrigated, waterlogged, or marshy grazing lands.",
        "best_season": "Post-monsoon (October/November)"
    }
]


def get_standard_schedules(animal_type: str = "cow") -> List[Dict[str, Any]]:
    """Returns official recommended vaccination schedule for species."""
    t = (animal_type or "cow").lower()
    if t in ["goat", "sheep"]:
        return STANDARD_VACCINES["sheep_goat"]
    return STANDARD_VACCINES["cattle_buffalo"]


def compute_health_status(record: Dict[str, Any], date_key: str = "next_due_date") -> Dict[str, Any]:
    """Annotates record with real-time status: overdue, due_soon, or up_to_date."""
    rec = dict(record)
    due_str = rec.get(date_key)
    if not due_str:
        rec["status_badge"] = "completed"
        rec["status_label"] = "Completed (No Expiry)"
        rec["days_diff"] = None
        return rec

    try:
        today = datetime.now(timezone.utc).date()
        due_date = datetime.strptime(due_str, "%Y-%m-%d").date()
        diff = (due_date - today).days

        if diff < 0:
            rec["status_badge"] = "overdue"
            rec["status_label"] = f"Overdue by {abs(diff)} day(s)"
            rec["days_diff"] = diff
        elif diff <= 14:
            rec["status_badge"] = "due_soon"
            rec["status_label"] = f"Due in {diff} day(s)"
            rec["days_diff"] = diff
        else:
            rec["status_badge"] = "up_to_date"
            rec["status_label"] = f"Up to date ({diff} days left)"
            rec["days_diff"] = diff
    except Exception:
        rec["status_badge"] = "scheduled"
        rec["status_label"] = "Scheduled"
        rec["days_diff"] = None

    return rec


def calculate_next_due(administered_date: str, interval_days: Optional[int]) -> Optional[str]:
    """Calculates next due date given administered date and frequency interval."""
    if not interval_days or not administered_date:
        return None
    try:
        d = datetime.strptime(administered_date, "%Y-%m-%d")
        return (d + timedelta(days=interval_days)).strftime("%Y-%m-%d")
    except ValueError:
        return None
