"""VetAI Chatbot Engine ("VetBot") — AI Veterinary Advisory Assistant.

Supports dual-mode operation:
1. OFFLINE / LOCAL (Default, Zero-cost, 100% Reliable):
   - Semantic retrieval engine grounded in diseases.json.
   - Comprehensive veterinary advisory for:
     • 15 specific diseases (FMD, LSD, Mastitis, Bloat, HS, BVD, etc.)
     • Milk Yield & Nutritional Rations (TMR, Mineral Mixtures, Bypass Fat)
     • Calf Care & Colostrum Protocols
     • Breeding, Gestation Periods & Calving Signs
     • Heat Stress Management in Cattle & Buffaloes
     • Deworming & Vaccination Schedules
     • On-Farm Emergency First Aid (Trocarization, Choking, Downer Cows)
   - Resolves disease context from conversation history or prior diagnostic scan.

2. ONLINE LLM (Optional Enhancement via API Key):
   - If GEMINI_API_KEY or OPENAI_API_KEY is configured in the environment,
     it enhances the response with conversational RAG.
"""

import json
import os
import re
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
DISEASES_PATH = os.path.join(HERE, "ml", "diseases.json")

# Load disease knowledge base once
with open(DISEASES_PATH, encoding="utf-8") as f:
    _kb = json.load(f)

_diseases = {d["id"]: d for d in _kb["diseases"]}
_symptoms = {s["id"]: s["label"] for s in _kb["symptoms"]}

# Disease Aliases & Common Farmer Terms (English, Hinglish & Hindi)
DISEASE_ALIASES = {
    "fmd": ["fmd", "foot and mouth", "foot-and-mouth", "mouth blister", "hoof blister", "khurpaka", "muhpaka", "khurha", "खुरपका", "मुंहपका", "मुंहपका-खुरपका", "खुरहा"],
    "lsd": ["lsd", "lumpy skin", "lumpy", "skin nodules", "skin lumps", "lumpy virus", "gaanth", "लम्पी", "लंपी", "गांठदार त्वचा"],
    "mastitis": ["mastitis", "udder swelling", "swollen teat", "milk clot", "thanela", "thanail", "blood in milk", "थनैला", "थनैल", "थन में सूजन"],
    "brd": ["brd", "respiratory", "bovine respiratory", "pneumonia", "coughing cow", "chest infection", "निमोनिया", "खांसी", "सांस फूलना"],
    "bloat": ["bloat", "bloating", "swollen belly", "tympany", "rumen gas", "afara", "pait phoolna", "अफारा", "पेट फूलना", "गैस"],
    "hs": ["hs", "haemorrhagic septicaemia", "hemorrhagic septicemia", "shipping fever", "throat swelling", "galghotu", "gharke", "गलघोंटू", "घुरका"],
    "bvd_enteritis": ["bvd", "diarrhea", "enteritis", "viral diarrhea", "scours", "loose dung", "dast", "दस्त", "पतला गोबर"],
    "ketosis": ["ketosis", "acetonemia", "sweet breath", "acetone breath", "low blood sugar", "ketone", "कीटोसिस"],
    "milk_fever": ["milk fever", "hypocalcemia", "calcium deficiency", "downer cow", "cannot stand after calving", "sutika", "दुग्ध ज्वर", "सूतिका ज्वर", "कैल्शियम की कमी"],
    "parasites": ["parasite", "parasites", "worms", "worm", "fluke", "deworming", "stomach worms", "pet ke kide", "पेट के कीड़े", "कृमि", "कीड़ा"],
    "bovine_tb": ["bovine tb", "tb", "tuberculosis", "chronic cough", "wasting disease", "टीबी", "तपेदिक"],
    "bluetongue": ["bluetongue", "blue tongue", "swollen tongue", "cyanosis", "sheep fever", "ब्लूटंग", "नीली जीभ"],
    "johnes": ["johnes", "johne's", "paratuberculosis", "chronic wasting", "pipe stream diarrhea"],
    "cae": ["cae", "caprine arthritis", "swollen knee goat", "goat arthritis", "गठिया"],
    "scrapie": ["scrapie", "prion", "itching sheep", "wool scraping", "sheep incoordination"]
}

# Intent detection keywords (English & Hindi)
INTENTS = {
    "diet": ["diet", "food", "feed", "feeding", "eat", "eating", "fodder", "nutrition", "drink", "water", "gruel", "grass", "hay", "आहार", "चारा", "खुराक", "दाना", "घास"],
    "prevention": ["prevent", "prevention", "protect", "avoid", "vaccine", "vaccination", "biosecurity", "hygiene", "stop spread", "quarantine", "रोकथाम", "बचाव", "टीका", "टीकाकरण"],
    "supportive_care": ["supportive", "care", "manage", "nursing", "bedding", "isolate", "isolation", "clean", "wound", "first aid", "comfort", "देखभाल", "सफाई", "मरहम", "पट्टी"],
    "treatment": ["treat", "treatment", "medicine", "medication", "cure", "drug", "antibiotic", "injection", "remedy", "therapy", "salve", "उपचार", "दवा", "इलाज", "दवाई", "इंजेक्शन"],
    "symptoms": ["symptom", "symptoms", "signs", "identify", "how to know", "look like", "spot", "manifestation", "लक्षण", "पहचान", "निशानी"],
    "transmission": ["cause", "causes", "spread", "transmission", "contagious", "catch", "vector", "flies", "ticks", "airborne", "फैलाव", "संक्रमण", "कारण"],
    "recovery": ["recovery", "recover", "how long", "time", "timeline", "prognosis", "survive", "death", "fatal", "mortality", "ठीक होना", "आरोग्य", "समय"],
    "emergency": ["emergency", "critical", "dying", "urgent", "cannot stand", "downer", "choking", "severe", "save", "convulsion", "आपातकाल", "खतरा", "गंभीर", "तुरंत"]
}

# General Veterinary Domain Modules (English & Hindi)
TOPICS = {
    "milk_yield": [
        "increase milk", "more milk", "milk yield", "milk production", "doodh badhana", "mineral mixture",
        "dairy ration", "calcium supplement", "bypass fat", "lactation", "दूध बढ़ाना", "दूध कैसे बढ़ाएं", "दूध वृद्धि", "बायपास फैट"
    ],
    "calf_care": [
        "calf", "calves", "newborn", "colostrum", "bacchada", "navel", "calf scours", "calf diarrhea", "weaning", "बछड़ा", "बछिया", "खीस", "नाल"
    ],
    "breeding": [
        "pregnant", "pregnancy", "gestation", "calving", "delivery", "heat cycle", "insemination", "artificial insemination",
        "dry period", "biyana", "delivery signs", "gestation period", "गाभिन", "गर्भाधान", "मद", "गर्भावस्था", "ब्यांत"
    ],
    "heat_stress": [
        "heat stress", "summer", "hot weather", "cooling", "panting", "buffalo wallowing", "dehydration in cow", "गर्मी", "लू", "हाफना"
    ],
    "deworming": [
        "deworm", "deworming schedule", "how often to deworm", "albendazole", "deworming medicine", "कीड़े की दवा", "कृमिनाशक", "पेट के कीड़े"
    ],
    "emergency_first_aid": [
        "first aid", "choking", "snake bite", "poisoning", "acute bloat", "trocar", "puncture belly", "fracture", "प्राथमिक उपचार", "दम घुटना", "जहर"
    ]
}


def _detect_disease(text: str, context: Optional[dict] = None) -> Optional[Dict[str, Any]]:
    """Identifies the disease being asked about from text or diagnosis context."""
    text_lower = text.lower()
    for d_id, aliases in DISEASE_ALIASES.items():
        for alias in aliases:
            if re.search(r"\b" + re.escape(alias) + r"\b", text_lower) or (any(ord(c) > 127 for c in alias) and alias in text_lower):
                return _diseases.get(d_id)
                
    if context and context.get("disease_id"):
        cid = context["disease_id"]
        if cid in _diseases:
            return _diseases[cid]
            
    if context and context.get("name"):
        cname = context["name"].lower()
        for d_id, aliases in DISEASE_ALIASES.items():
            if any(a in cname for a in aliases):
                return _diseases.get(d_id)

    return None


def _detect_topic(text: str) -> Optional[str]:
    """Detects general veterinary topics if no specific disease matches."""
    text_lower = text.lower()
    for topic, keywords in TOPICS.items():
        if any(kw in text_lower for kw in keywords):
            return topic
    return None


def _is_hindi(text: str) -> bool:
    """Detects if query contains Devanagari Hindi or explicitly requests Hindi."""
    return any("\u0900" <= c <= "\u097F" for c in text) or any(w in text.lower() for w in ["in hindi", "hindi me", "हिंदी"])


def _detect_intent(text: str) -> str:
    """Classifies clinical intent (supporting English and Hindi keywords)."""
    text_lower = text.lower()
    if any((k in text_lower) if any(ord(c) > 127 for c in k) else bool(re.search(r"\b" + re.escape(k) + r"\b", text_lower)) for k in INTENTS["emergency"]):
        return "emergency"
        
    scores = {}
    for intent, kws in INTENTS.items():
        if intent == "emergency":
            continue
        count = 0
        for kw in kws:
            if any(ord(c) > 127 for c in kw):
                if kw in text_lower:
                    count += 1
            else:
                if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                    count += 1
        scores[intent] = count
        
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"



def _generate_topic_reply(topic: str) -> Dict[str, Any]:
    """Generates comprehensive veterinary guidance for broad livestock queries."""
    if topic == "milk_yield":
        return {
            "reply": (
                "🥛 **How to Optimize Milk Yield & Dairy Nutrition**\n\n"
                "**1. High-Efficiency Balanced Feeding (2:1 Fodder Ratio):**\n"
                "• Green Fodder: 25–30 kg/day (berseem, lucerne, maize, sorghum).\n"
                "• Dry Fodder: 5–7 kg/day (wheat straw, paddy straw) to ensure rumen fibrous scratch.\n"
                "• Concentrate Feed: 1 kg concentrate for every 2.5 kg of milk produced + 1.5 kg for body maintenance.\n\n"
                "**2. Mineral Mixture & Calcium:**\n"
                "• Feed **50–60 grams of chelated mineral mixture daily** to prevent silent heat, mastitis, and trace mineral deficiencies.\n"
                "• Supplement with bypass fat (100–150g/day) during early lactation peak.\n\n"
                "**3. Water Management:**\n"
                "• A high-yielding cow requires **80–120 liters of clean, fresh water** daily. Milk is 87% water — water deprivation instantly drops milk yield.\n\n"
                "**4. Milking Hygiene:**\n"
                "• Always complete milking within 6–7 minutes (oxytocin surge window). Pre-dip and post-dip teats to prevent subclinical mastitis."
            ),
            "suggestions": [
                "What is the best diet for Bloat?",
                "How to prevent Mastitis in dairy cows?",
                "What are the symptoms of Milk Fever?",
                "What is the recommended deworming schedule?"
            ],
            "source": "local_kb",
            "disease_identified": None
        }

    elif topic == "calf_care":
        return {
            "reply": (
                "🍼 **Newborn Calf Management & Care Protocol**\n\n"
                "**1. Critical Colostrum Window (First 2 Hours):**\n"
                "• Feed maternal colostrum at **10% of the calf's body weight** (approx. 2.5 to 3.5 liters) within the first 2 hours of life.\n"
                "• The calf gut can only absorb maternal antibodies (IgG) during the first 24 hours to build systemic immunity.\n\n"
                "**2. Navel Cord Disinfection:**\n"
                "• Immediately dip the naval cord in **7% Tincture of Iodine** to prevent joint ill and navel ill bacteria.\n\n"
                "**3. Bedding & Temperature:**\n"
                "• Keep bedding dry, warm, and clean. Prevent drafts in winter and provide shade in summer.\n\n"
                "**4. Deworming & Starter Feed:**\n"
                "• First deworming at day 7 to 10 for roundworms (Toxocara vitulorum).\n"
                "• Introduce dry calf starter grain and green grass nibbling from week 2 to stimulate rumen papillae growth."
            ),
            "suggestions": [
                "How to treat diarrhea in calves?",
                "What are the signs of Foot and Mouth Disease?",
                "When should I deworm adult cattle?",
                "How to increase milk yield?"
            ],
            "source": "local_kb",
            "disease_identified": None
        }

    elif topic == "breeding":
        return {
            "reply": (
                "🐄 **Livestock Gestation, Breeding & Calving Protocol**\n\n"
                "**1. Typical Gestation Periods:**\n"
                "• **Cows:** ~283 days (approx. 9 months 9 days)\n"
                "• **Buffaloes:** ~310 days (approx. 10 months 10 days)\n"
                "• **Goats & Sheep:** ~150 days (approx. 5 months)\n\n"
                "**2. Dry Period Management:**\n"
                "• Stop milking pregnant cows **60 days before calving** (Dry Period) to regenerate udder tissue and ensure high colostrum quality.\n"
                "• Avoid excessive calcium during the last 2–3 weeks before calving to prime parathyroid hormone and avoid Milk Fever (hypocalcemia).\n\n"
                "**3. Signs of Imminent Calving (24–48 Hours Before):**\n"
                "• Relaxation/sinking of pelvic ligaments near the tailhead.\n"
                "• Udder engorgement with teat waxing.\n"
                "• Restlessness, isolation from the herd, and vaginal mucus discharge.\n\n"
                "*If labor contractions exceed 2 hours with no calf presentation, call a licensed veterinarian immediately to assist dystocia.*"
            ),
            "suggestions": [
                "What are the symptoms of Milk Fever?",
                "How to care for a newborn calf?",
                "What to feed cows after calving?",
                "What are signs of Ketosis in fresh cows?"
            ],
            "source": "local_kb",
            "disease_identified": None
        }

    elif topic == "heat_stress":
        return {
            "reply": (
                "☀️ **Summer Heat Stress Management in Livestock**\n\n"
                "**1. Buffalo Wallowing & Misting:**\n"
                "• Buffaloes have fewer sweat glands and black skin that absorbs heat. Provide clean water wallowing twice daily or install rooftop sprinklers.\n"
                "• Run fans and foggers during peak daytime heat (11 AM to 4 PM).\n\n"
                "**2. Electrolytes & Feeding Schedule:**\n"
                "• Shift major feeding hours to the cooler morning (before 7 AM) and evening/night (after 8 PM).\n"
                "• Add **sodium bicarbonate (baking soda, 50–100g)** and electrolytes to maintain rumen pH and prevent subacute rumen acidosis.\n\n"
                "**3. Signs of Severe Heat Stroke:**\n"
                "• Rapid open-mouth breathing, drooling, rectal temp > 40.5°C (105°F), refusal to stand. Sponge with cool (not freezing) water on the head and neck."
            ),
            "suggestions": [
                "What feed is best for Bloat?",
                "How to protect cows from tick fever?",
                "How to increase milk production in summer?",
                "What are the symptoms of Haemorrhagic Septicaemia?"
            ],
            "source": "local_kb",
            "disease_identified": None
        }

    elif topic == "deworming":
        return {
            "reply": (
                "🪱 **Livestock Deworming Schedule & Best Practices**\n\n"
                "**1. Routine Frequency:**\n"
                "• Adult Dairy Cattle & Buffaloes: **Every 3 to 4 months** (especially pre-monsoon and post-monsoon).\n"
                "• Sheep & Goats: Every 2 to 3 months (very susceptible to Haemonchus / wireworm).\n\n"
                "**2. Rotation of Drug Classes:**\n"
                "• Rotate anthelmintic classes (Benzimidazoles, Levamisole, Ivermectin/Moxidectin) to prevent parasite drug resistance.\n"
                "• Weigh the animal or tape measurement to ensure accurate dosing — underdosing breeds resistant worms.\n\n"
                "**3. Grazing Management:**\n"
                "• Avoid early morning grazing while dew is present on pasture tips where parasite larvae migrate."
            ),
            "suggestions": [
                "What are the symptoms of Gastrointestinal Parasites?",
                "How often to vaccinate for FMD?",
                "What to feed a bloated animal?",
                "How to care for a newborn calf?"
            ],
            "source": "local_kb",
            "disease_identified": "parasites"
        }

    elif topic == "emergency_first_aid":
        return {
            "reply": (
                "🚨 **Livestock Emergency On-Farm First Aid**\n\n"
                "**1. Severe Acute Bloat (Tympany):**\n"
                "• If the left flank is ballooned tight and the cow is gasping for air, position head upright. Keep mouth open with a wooden gag.\n"
                "• Administer 200–300 ml vegetable oil (mustard or peanut oil) with 30g ginger/baking soda.\n"
                "• In immediate life-threatening suffocation: a trained vet performs trocar and cannula puncture at the left paralumbar fossa.\n\n"
                "**2. Downer Cow (Cannot Stand After Calving):**\n"
                "• Suspect **Milk Fever (acute hypocalcemia)**. Do NOT force the animal to stand violently to prevent hip dislocation.\n"
                "• Provide deep soft straw bedding. Requires slow IV Calcium Borogluconate by a veterinarian.\n\n"
                "**3. Heavy Bleeding:**\n"
                "• Apply firm, clean direct pressure with a sterile towel or cloth for at least 10 minutes. Call an emergency vet immediately."
            ),
            "suggestions": [
                "What is the treatment for Bloat?",
                "What is the supportive care for Milk Fever?",
                "Find emergency veterinary clinic near me",
                "What are signs of Haemorrhagic Septicaemia?"
            ],
            "source": "local_kb",
            "disease_identified": None
        }

    return None


def _generate_local_reply(query: str, disease: Optional[Dict[str, Any]], intent: str, context: Optional[dict]) -> Dict[str, Any]:
    """Generates an accurate, structured clinical advisory response offline."""
    query_lower = query.lower().strip()
    
    # Greetings / Meta queries
    if any(g in query_lower for g in ["hi", "hello", "hey", "namaste", "help", "who are you", "what can you do"]):
        return {
            "reply": (
                "👋 **Hello! I am VetBot**, your AI Veterinary Advisory Assistant.\n\n"
                "I can assist you with:\n"
                "- 🩺 **Livestock Diseases & Symptoms** (FMD, Lumpy Skin, Mastitis, Bloat, HS, etc.)\n"
                "- 🌾 **Dietary & Nutritional Management** (Rumen rations, milk yield boost, mineral mixtures)\n"
                "- 🍼 **Calf Care & Colostrum Protocols**\n"
                "- 🐄 **Breeding, Gestation & Calving Support**\n"
                "- ☀️ **Heat Stress & Summer Buffalo Management**\n"
                "- 🪱 **Vaccination & Deworming Calendars**\n"
                "- 🚨 **On-Farm Emergency First Aid Guidance**\n\n"
                "*Tip: Tap any suggestion pill below or ask any question directly!*"
            ),
            "suggestions": [
                "How to increase milk yield in dairy cows?",
                "What diet is best for bloat?",
                "How to prevent Foot & Mouth Disease?",
                "What are newborn calf care steps?",
                "What are signs of Milk Fever?"
            ],
            "source": "local_kb",
            "disease_identified": disease["id"] if disease else None
        }

    # Check for broader veterinary topics if no disease was matched
    if not disease:
        topic = _detect_topic(query)
        if topic:
            topic_reply = _generate_topic_reply(topic)
            if topic_reply:
                return topic_reply

        # Fallback general query guidance
        return {
            "reply": (
                "I'm here to help with livestock healthcare and dairy farm management!\n\n"
                "You can ask me about:\n"
                "• **Diseases:** Foot & Mouth, Lumpy Skin, Mastitis, Bloat, Blackleg, Milk Fever, Ketosis, Respiratory Complex.\n"
                "• **Herd Management:** How to boost milk yield, calf colostrum feeding, heat stress in buffaloes, deworming cycles.\n"
                "• **First Aid:** What to do when an animal is bloated, downer cow, or choking.\n\n"
                "What specific symptom, animal, or question can I assist you with?"
            ),
            "suggestions": [
                "How to increase milk yield?",
                "What to do if a cow has bloat?",
                "Newborn calf care tips",
                "How does Lumpy Skin spread?",
                "Emergency first aid for downer cows"
            ],
            "source": "local_kb",
            "disease_identified": None
        }

    # Disease-specific responses
    name = disease["name"]
    risk = disease["risk_level"].upper()
    animals = ", ".join(a.title() for a in disease.get("animals", []))

    reply_parts = []
    
    if intent == "emergency":
        reply_parts.append(f"🚨 **EMERGENCY ADVISORY: {name} (Risk Tier: {risk})**")
        reply_parts.append(
            f"If your animal is severely distressed, recumbent (unable to rise), or having difficulty breathing, "
            f"**contact an emergency licensed veterinarian immediately**.\n"
        )
        reply_parts.append(f"**Immediate Supportive Actions:**\n{disease.get('supportive_care')}\n")
        reply_parts.append(f"**Emergency Dietary Note:**\n{disease.get('diet')}")
        suggestions = [
            f"What is the supportive care for {name}?",
            f"What medication is used for {name}?",
            f"How long does recovery take for {name}?",
            "Find emergency veterinary clinic near me"
        ]

    elif intent == "diet":
        reply_parts.append(f"🌾 **Dietary & Feeding Protocol for {name}**")
        reply_parts.append(f"**Recommended Diet:**\n{disease.get('diet')}\n")
        reply_parts.append(f"**Key Supportive Care:**\n{disease.get('supportive_care')}\n")
        reply_parts.append(f"**Nutritional Strategy:** Proper hydration and fibrous roughage maintain rumen motility while supporting immune recovery.")
        suggestions = [
            f"What is the supportive care for {name}?",
            f"How to prevent {name} in the herd?",
            f"How long does recovery take for {name}?"
        ]

    elif intent == "supportive_care":
        reply_parts.append(f"🩹 **Supportive Care & Farm Management: {name}**")
        reply_parts.append(f"**Step-by-Step Care:**\n{disease.get('supportive_care')}\n")
        reply_parts.append(f"**Recommended Diet:**\n{disease.get('diet')}\n")
        reply_parts.append(f"**Typical Recovery Duration:** {disease.get('recovery_time')}")
        suggestions = [
            f"What diet should I give for {name}?",
            f"What veterinary treatment is needed for {name}?",
            f"How to stop {name} from spreading?"
        ]

    elif intent == "treatment":
        reply_parts.append(f"💊 **Treatment Overview: {name}**")
        reply_parts.append(f"**Clinical Overview:**\n{disease.get('treatment_overview')}\n")
        reply_parts.append(f"**Supportive Measures:**\n{disease.get('supportive_care')}\n")
        reply_parts.append(
            f"> ⚠️ **Prescription Warning:** Antibiotics, antiprotozoals, and prescription drugs must only be administered "
            f"under the direct guidance of a licensed veterinarian to prevent drug resistance and milk/meat residues."
        )
        suggestions = [
            f"What supportive care can I provide at home for {name}?",
            f"What is the recommended diet for {name}?",
            f"What is the vaccination schedule for {name}?"
        ]

    elif intent == "prevention":
        reply_parts.append(f"🛡️ **Prevention & Biosecurity: {name}**")
        reply_parts.append(f"**Farm Prevention Guidelines:**\n{disease.get('prevention')}\n")
        if disease.get("vaccination"):
            reply_parts.append(f"**Vaccination Schedule:**\n{disease.get('vaccination')}\n")
        reply_parts.append(f"**Transmission Vectors:** {disease.get('transmission')}")
        suggestions = [
            f"What are the early symptoms of {name}?",
            f"What should I feed an animal with {name}?",
            f"How does {name} spread?"
        ]

    elif intent == "transmission":
        reply_parts.append(f"🔬 **Causes & Disease Transmission: {name}**")
        reply_parts.append(f"**Pathogen/Cause:** {disease.get('causes')}\n")
        reply_parts.append(f"**Transmission Mode:**\n{disease.get('transmission')}\n")
        reply_parts.append(f"**Prevention Measures:**\n{disease.get('prevention')}")
        suggestions = [
            f"How to prevent {name}?",
            f"What are the symptoms of {name}?",
            f"What care is needed for {name}?"
        ]

    elif intent == "recovery":
        reply_parts.append(f"⏱️ **Prognosis & Recovery Timeline: {name}**")
        reply_parts.append(f"**Estimated Recovery Time:**\n{disease.get('recovery_time')}\n")
        reply_parts.append(f"**Supportive Nursing:**\n{disease.get('supportive_care')}\n")
        reply_parts.append(f"**Risk Level:** {risk} — Monitor vitals closely during this recovery period.")
        suggestions = [
            f"What diet helps recovery from {name}?",
            f"What are the symptoms of {name}?",
            f"When should I call the vet for {name}?"
        ]

    elif intent == "symptoms":
        sym_names = [_symptoms.get(s, s.replace('_', ' ').title()) for s in disease.get("key_symptoms", [])]
        reply_parts.append(f"🔍 **Key Clinical Signs of {name}**")
        reply_parts.append(f"**Susceptible Animals:** {animals}\n")
        reply_parts.append(f"**Typical Symptoms:**\n• " + "\n• ".join(sym_names) + "\n")
        v = disease.get("vitals", {})
        if v:
            reply_parts.append(f"**Expected Vitals Range:** Temp: {v.get('temp_c', [0,0])[0]}–{v.get('temp_c', [0,0])[1]}°C, HR: {v.get('hr_bpm', [0,0])[0]}–{v.get('hr_bpm', [0,0])[1]} bpm\n")
        reply_parts.append(f"**Condition Overview:** {disease.get('definition')}")
        suggestions = [
            f"What is the treatment for {name}?",
            f"What feed should I give for {name}?",
            f"How to prevent {name}?"
        ]

    else:
        # General comprehensive overview
        reply_parts.append(f"📋 **Clinical Overview: {name}**")
        reply_parts.append(f"**Definition:** {disease.get('definition')}\n")
        reply_parts.append(f"**Risk Tier:** `{risk}` | **Susceptible Species:** {animals}\n")
        reply_parts.append(f"**Immediate Supportive Care:**\n{disease.get('supportive_care')}\n")
        reply_parts.append(f"**Dietary Recommendation:**\n{disease.get('diet')}\n")
        reply_parts.append(f"**Expected Recovery Window:** {disease.get('recovery_time')}")
        suggestions = [
            f"What feed is best for {name}?",
            f"What is the medical treatment for {name}?",
            f"How does {name} spread?",
            f"How to prevent {name}?"
        ]

    reply_parts.append("\n\n*VetAI provides educational veterinary guidance. Always confirm treatment plans with a licensed veterinarian.*")
    
    return {
        "reply": "\n".join(reply_parts),
        "suggestions": suggestions,
        "source": "local_kb",
        "disease_identified": disease["id"]
    }


# ==========================================
# PHASE 4: HINDI VETERINARY KNOWLEDGE BASE
# ==========================================
DISEASES_HI = {
    "fmd": {
        "name": "खुरपका-मुंहपका रोग (Foot & Mouth Disease - FMD)",
        "definition": "यह एक अत्यंत संक्रामक विषाणुजनित रोग है, जो गाय, भैंस, बकरी और भेड़ जैसे खुर वाले पशुओं में मुंह और खुरों पर दर्दनाक छाले पैदा करता है।",
        "supportive_care": "• मुंह के छालों को 2% पोटेशियम परमैंगनेट (लाल दवा) या फिटकरी के पानी से धोएं।\n• खुरों के घावों पर बोरिक एसिड और ग्लिसरीन का लेप या नीम का तेल लगाएं।\n• पीड़ित पशु को स्वस्थ पशुओं से तुरंत अलग रखें और छायादार सूखे बाड़े में रखें।",
        "diet": "• मुलायम, पतला और आसानी से पचने वाला दलिया, पतली कांजी और मुलायम हरा चारा दें। सूखा भूसा या कठोर दाना बिल्कुल न दें।",
        "treatment_overview": "• वायरल रोग होने के कारण इसका कोई सीधा एंटीवायरल नहीं है।\n• द्वितीयक जीवाणु संक्रमण और दर्द रोकने हेतु पशुचिकित्सक एंटीबायोटिक और एनाल्जेसिक इंजेक्शन देते हैं।",
        "prevention": "• राष्ट्रीय पशुरोग नियंत्रण कार्यक्रम (FMD-CP) के तहत साल में दो बार (हर 6 महीने) नियमित टीका लगवाएं।\n• बाड़े में चूने का छिड़काव करें।",
        "recovery_time": "10 से 15 दिन (उचित देखभाल के साथ)"
    },
    "lsd": {
        "name": "लंपी त्वचा रोग (Lumpy Skin Disease - LSD)",
        "definition": "कैप्रिपॉक्स वायरस जनित संक्रामक रोग, जिसमें पशु के पूरे शरीर की त्वचा पर सख्त गांठें या ढेले बन जाते हैं, तेज बुखार आता है और दूध घट जाता है।",
        "supportive_care": "• पशु को मक्खी, मच्छर और किलनी (चींचड़ों) से सुरक्षित साफ बाड़े में रखें।\n• गांठों और घावों पर नीम का पानी और हल्दी का लेप लगाएं।\n• हवादार छायादार स्थान दें और पशु को धूप से बचाएं।",
        "diet": "• गुड़, हल्दी, काली मिर्च और गिलोय का काढ़ा दें।\n• पानी में ग्लूकोज व इलेक्ट्रोल मिलाकर पिलाएं और सुपाच्य हरा चारा दें।",
        "treatment_overview": "• बुखार और दर्द कम करने हेतु पैरासिटामोल/एंटी-इंफ्लेमेटरी और एंटीबायोटिक पशुचिकित्सक की देखरेख में दी जाती है।",
        "prevention": "• गोट पॉक्स (Goat Pox) वैक्सीन 3ml सब-क्यूटेनियस लगाकर टीकाकरण कराएं।\n• बाड़े में नीम के पत्तों का धुआं करें।",
        "recovery_time": "2 से 4 सप्ताह"
    },
    "mastitis": {
        "name": "थनैला रोग (Mastitis)",
        "definition": "अयन (लेवा) या थनों में जीवाणु संक्रमण के कारण होने वाला रोग, जिसमें थन में सूजन, गर्माहट, दर्द और दूध में थक्के, मवाद या खून आता है।",
        "supportive_care": "• रोगग्रस्त थन को दिन में 3-4 बार पूरी तरह खाली (दुह) करें और यह दूध नष्ट कर दें।\n• तीव्र सूजन में ठंडे पानी या बर्फ की सिंकाई करें।",
        "diet": "• दाने की मात्रा थोड़ी कम करें और विटामिन ई, सेलेनियम और त्रिसोडियम साइट्रेट पाउडर (50g/दिन) दें।",
        "treatment_overview": "• पशुचिकित्सक से तुरंत थन में इंट्रा-मेमरी एंटीबायोटिक ट्यूब डलवाएं और दर्द-सूजन रोधी इंजेक्शन लगवाएं।",
        "prevention": "• दुहाने के बाद थनों को 0.5% पोविडोन आयोडीन घोल में डुबोएं (Post-milking teat dip)।\n• दुहाने के तुरंत बाद 30 मिनट तक पशु को बैठने न दें।",
        "recovery_time": "5 से 7 दिन (समय पर इलाज होने पर)"
    },
    "bloat": {
        "name": "अफारा / पेट में गैस (Bloat / Tympany)",
        "definition": "रूमेन (पेट) में किण्वन से अत्यधिक गैस भर जाने के कारण बायां पेट ढोलक की तरह फूल जाना, बेचैनी और सांस लेने में कठिनाई होना।",
        "supportive_care": "• पशु के मुंह में लकड़ी की खपच्ची (गैग) बांधें ताकि मुंह खुला रहे और डकार आ सके।\n• पशु को आगे से ऊंचा खड़ा रखें और बाईं कोख की मालिश करें।",
        "diet": "• तुरंत सारा दाना और रसीला चारा (बरसीम) हटा लें। केवल सूखा भूसा दें।",
        "treatment_overview": "• 250-300 मिली सरसों या अलसी का तेल, 25 ग्राम हींग और 50 ग्राम मीठा सोडा मिलाकर नाल से पिलाएं।\n• दम घुटने की आपात स्थिति में पशुचिकित्सक से बाईं कोख में ट्रोकार-कैन्युला लगवाएं।",
        "prevention": "• भूखे पशु को ओस वाला गीला बरसीम या दलहनी चारा न खिलाएं। पहले सूखा भूसा खिलाएं।",
        "recovery_time": "1 से 6 घंटे"
    },
    "hs": {
        "name": "गलघोंटू रोग (Haemorrhagic Septicaemia - HS)",
        "definition": "पाश्चुरेला जीवाणु जनित अति-घातक रोग, जिसमें तेज बुखार, गले व जबड़े के नीचे गर्म दर्दनाक सूजन और सांस लेते समय घर्र-घर्र की आवाज आती है।",
        "supportive_care": "• यह अति-आपातकालीन स्थिति है! बिना देर किए सरकारी पशुचिकित्सक को तुरंत बुलाएं।\n• पशु को बिल्कुल शांत रखें और तनाव न दें।",
        "diet": "• निगलने में कठिनाई होने पर जबरदस्ती नाल से कुछ न पिलाएं (फेफड़ों में जाने का खतरा रहता है)।",
        "treatment_overview": "• शुरुआती घंटों में ही सल्फाडेमिडीन या शक्तिशाली एंटीबायोटिक का बड़ा डोज नस (IV) द्वारा दिया जाना आवश्यक है।",
        "prevention": "• मानसून से पहले (मई-जून में) HS का वार्षिक टीका अवश्य लगवाएं।",
        "recovery_time": "त्वरित इलाज पर 3 से 5 दिन, अन्यथा जानलेवा"
    },
    "bvd_enteritis": {
        "name": "विषाणुजनित दस्त व आंत्रशोथ (BVD / Enteritis)",
        "definition": "आंतों में विषाणु या जीवाणु संक्रमण से होने वाला गंभीर पतला दस्त, जिससे शरीर में पानी और लवणों की तीव्र कमी हो जाती है।",
        "supportive_care": "• ओआरएस (ORS) या नमक-चीनी का पानी और छाछ बार-बार पिलाएं ताकि डिहाइड्रेशन न हो।",
        "diet": "• उबले चावल का मांड, बेल का गूदा और हल्का दलिया दें। भारी व चिकना दाना बंद रखें।",
        "treatment_overview": "• पशुचिकित्सक की देखरेख में एंटी-डायरियल दवाएं और डिहाइड्रेशन में रिंगर लेक्टेट (RL) नस से चढ़वाई जाती है।",
        "prevention": "• बासी व फफूंद लगा चारा न दें। बाड़े में पानी का जमाव न होने दें।",
        "recovery_time": "3 से 7 दिन"
    },
    "ketosis": {
        "name": "कीटोसिस / एसिटोनीमिया (Ketosis)",
        "definition": "ब्याने के बाद दुधारू पशुओं में ऊर्जा (ग्लूकोज) की कमी से सांस और दूध में मीठी एसिटोन गंध आना, भूख न लगना और दूध गिरना।",
        "supportive_care": "• पशु को आराम दें और तुरंत ऊर्जा वर्धक खुराक दें।",
        "diet": "• प्रोपलीन ग्लाइकोल (200-300 ग्राम) या गुड़, शीरा और मक्का दाना खिलाएं।",
        "treatment_overview": "• पशुचिकित्सक द्वारा 20% से 50% डेक्सट्रोज (Dextrose IV) नस के जरिए चढ़ाया जाता है।",
        "prevention": "• गाभिन अवस्था में पशु को संतुलित आहार दें और ब्याने के बाद दाने की मात्रा धीरे-धीरे बढ़ाएं।",
        "recovery_time": "2 से 4 दिन"
    },
    "milk_fever": {
        "name": "दुग्ध ज्वर / सूतिका ज्वर (Milk Fever - Hypocalcemia)",
        "definition": "ब्याने के 24 से 72 घंटे में खून में कैल्शियम कम होने से पशु का बैठ जाना और उठ न पाना (Downer Cow)।",
        "supportive_care": "• पशु को जबरदस्ती खड़ा न करें। पेट के बल सीधा बैठाएं और दोनों तरफ भूसे की बोरियां लगा दें।",
        "diet": "• ब्याने के तुरंत बाद ओरल कैल्शियम जेल दें। पशु लेटा हो तो तरल जबरदस्ती न पिलाएं।",
        "treatment_overview": "• पशुचिकित्सक द्वारा कैल्शियम बोरो-ग्लूकोनेट (CBG) नस (IV) में धीरे-धीरे चढ़ाया जाता है।",
        "prevention": "• ब्याने से 2-3 हफ्ते पहले अधिक कैल्शियम न दें। ब्याने के बाद तुरंत कैल्शियम सप्लीमेंट दें।",
        "recovery_time": "उपचार के कुछ ही घंटों में"
    },
    "parasites": {
        "name": "पेट के कीड़े / आंतरिक कृमि (Internal Parasites)",
        "definition": "पेट और आंतों में कीड़ों के कारण पशु का कमजोर होना, बाल खुरदरे होना, बदबूदार गोबर और जबड़े के नीचे पानी की सूजन (बॉटल जॉ)।",
        "supportive_care": "• पशु को स्वच्छ जल दें और दलदली घास चरने से रोकें।",
        "diet": "• खनिज मिश्रण (50 ग्राम प्रतिदिन) और प्रोटीन युक्त संतुलित आहार दें।",
        "treatment_overview": "• पशु के वजन अनुसार अल्बेंडाजोल, फेनबेंडाजोल या आइवरमेक्टिन की उचित खुराक खाली पेट दें।",
        "prevention": "• साल में 3-4 बार कृमिनाशक दवा बदल-बदल कर दें। सुबह ओस के समय चराई न कराएं।",
        "recovery_time": "1 से 2 सप्ताह"
    },
    "brd": {
        "name": "श्वसन रोग / निमोनिया (BRD / Pneumonia)",
        "definition": "फेफड़ों का संक्रमण जिससे तेज बुखार, लगातार खांसी, नाक से मवाद और सांस फूलने की समस्या होती है।",
        "supportive_care": "• पशु को ठंडी हवा के सीधे झोंकों से बचाएं और सूखा बिछावन दें। नीलगिरी तेल की भाप दें।",
        "diet": "• गुनगुना पानी और हल्का गर्म दलिया दें।",
        "treatment_overview": "• पशुचिकित्सक से उपयुक्त एंटीबायोटिक और सांस नली खोलने वाली दवाएं लगवाएं।",
        "prevention": "• बाड़े में अच्छी हवा का आवागमन रखें और नमी न रहने दें।",
        "recovery_time": "7 से 14 दिन"
    },
    "bovine_tb": {
        "name": "गोजातीय टीबी / तपेदिक (Bovine TB)",
        "definition": "एक जीर्ण जीवाणुजनित रोग जिसमें लगातार सूखी खांसी, शरीर का सूखना और ग्रंथियों में सूजन होती है।",
        "supportive_care": "• बीमार पशु को झुंड से अलग रखें क्योंकि यह इंसानों में भी फैल सकता है।",
        "diet": "• पौष्टिक दाना और हरा चारा दें।",
        "treatment_overview": "• पशुओं में टीबी का इलाज प्रतिबंधित है; सरकारी पशु अस्पताल से ट्यूबरकुलिन जांच कराएं।",
        "prevention": "• केवल जांचे-परखे स्वस्थ पशु ही खरीदें।",
        "recovery_time": "जीर्ण रोग"
    },
    "bluetongue": {
        "name": "ब्लूटंग रोग / नीली जीभ (Bluetongue)",
        "definition": "भेड़-बकरियों में मक्खी के काटने से फैलने वाला रोग, जिसमें जीभ व मुंह में सूजन, जीभ का नीला पड़ना और लंगड़ापन होता है।",
        "supportive_care": "• छायादार स्थान में रखें और मुंह के छालों को लाल दवा के घोल से धोएं।",
        "diet": "• मुलायम हरी पत्तियां और पतला दलिया दें।",
        "treatment_overview": "• दर्द निवारक और एंटीबायोटिक दवाएं पशुचिकित्सक द्वारा दी जाती हैं।",
        "prevention": "• मक्खियों-मच्छरों से बचाव करें और मानसून से पूर्व टीका लगवाएं।",
        "recovery_time": "2 से 3 सप्ताह"
    },
    "johnes": {
        "name": "जोन्स रोग (Johne's Disease)",
        "definition": "आंतों का पुराना जीवाणु रोग जिससे पशु खाते-पीते भी लगातार सूखता जाता है और पतला गोबर करता है।",
        "supportive_care": "• गोबर का सुरक्षित निस्तारण करें और बछड़ों को अलग रखें।",
        "diet": "• आसानी से पचने वाला संतुलित आहार दें।",
        "treatment_overview": "• इसका कोई स्थायी इलाज नहीं है; सहायक देखभाल करें।",
        "prevention": "• नए पशु की रक्त/गोबर जांच कराएं।",
        "recovery_time": "असाध्य जीर्ण रोग"
    },
    "cae": {
        "name": "कैप्राइन आर्थराइटिस (CAE in Goats)",
        "definition": "बकरियों का जोड़ों का रोग जिससे घुटनों में सूजन, दर्द और लंगड़ापन होता है।",
        "supportive_care": "• जोड़ों के दर्द से राहत के लिए मुलायम सूखा बिछावन दें।",
        "diet": "• संतुलित आहार और खनिज लवण दें।",
        "treatment_overview": "• पशुचिकित्सक से दर्द व सूजन कम करने वाली दवाएं दिलवाएं।",
        "prevention": "• संक्रमित बकरी का दूध छोटे मेमनों को न पिलाएं।",
        "recovery_time": "आजीवन प्रबंधन"
    },
    "scrapie": {
        "name": "स्क्रेपी रोग (Scrapie)",
        "definition": "भेड़-बकरियों का गंभीर तंत्रिका रोग जिससे अत्यधिक खुजली, ऊन का झड़ना और लड़खड़ाहट होती है।",
        "supportive_care": "• पशु को शांत वातावरण में रखें।",
        "diet": "• सामान्य चारा।",
        "treatment_overview": "• कोई इलाज नहीं है; तुरंत पशुपालन विभाग को सूचित करें।",
        "prevention": "• प्रमाणित रोगमुक्त पशु ही खरीदें।",
        "recovery_time": "घातक तंत्रिका रोग"
    }
}

TOPICS_HI = {
    "milk_yield": {
        "reply": (
            "🥛 **दूध उत्पादन बढ़ाने और संतुलित डेयरी आहार की पूरी जानकारी**\n\n"
            "**1. संतुलित हरा व सूखा चारा (2:1 अनुपात):**\n"
            "• हरा चारा: 25–30 किलो प्रतिदिन (बरसीम, लूसर्न, मक्का, ज्वार या नेपियर घास)।\n"
            "• सूखा चारा: 5–7 किलो प्रतिदिन (गेहूं का भूसा या धान की पुआल) रूमेन पाचन के लिए।\n"
            "• दाना मिश्रण: हर 2.5 किलो दूध पर 1 किलो दाना + शरीर के रखरखाव के लिए 1.5 किलो दाना।\n\n"
            "**2. मिनरल मिक्सचर (खनिज मिश्रण) व बाईपास फैट:**\n"
            "• प्रतिदिन **50–60 ग्राम अच्छी गुणवत्ता वाला कीलेटेड मिनरल मिक्सचर** जरूर दें।\n"
            "• ब्याने के बाद शुरुआती 90 दिनों तक 100–150 ग्राम बाईपास फैट दें ताकि दूध में फैट व उत्पादन बढ़े।\n\n"
            "**3. स्वच्छ पानी की व्यवस्था:**\n"
            "• दुधारू पशु को प्रतिदिन **80–120 लीटर ताजा व स्वच्छ पानी** चाहिए। दूध में 87% पानी होता है — पानी की कमी से दूध तुरंत गिर जाता है।\n\n"
            "**4. सही दोहन तकनीक:**\n"
            "• दुहाई 6–7 मिनट के अंदर पूरी कर लें (ऑक्सीटोसिन का असर रहने तक)। थनों को दुहाई के बाद लाल दवा या आयोडीन से साफ करें।"
        ),
        "suggestions": [
            "अफारा में क्या आहार दें?",
            "थनैला रोग से बचाव कैसे करें?",
            "दुग्ध ज्वर (Milk Fever) के क्या लक्षण हैं?",
            "पेट के कीड़े की दवा कब देनी चाहिए?"
        ]
    },
    "calf_care": {
        "reply": (
            "🍼 **नवजात बछड़े/बछिया की देखभाल के 4 सुनहरे नियम**\n\n"
            "**1. खीस (Colostrum) पिलाना (पहले 2 घंटे में):**\n"
            "• जन्म के 2 घंटे के भीतर बछड़े के वजन का **10% खीस** (लगभग 2 से 2.5 लीटर) अवश्य पिलाएं।\n"
            "• खीस से ही बछड़े को मां की रोग प्रतिरोधक क्षमता (एंटीबॉडीज) मिलती है।\n\n"
            "**2. नाल की सफाई (Navel Dipping):**\n"
            "• नाल को शरीर से 2 इंच छोड़कर साफ धागे से बांधें और **7% टिंचर आयोडीन** में डुबोएं ताकि नाल का संक्रमण न हो।\n\n"
            "**3. पहली डीवर्मिंग (कीड़े की दवा):**\n"
            "• जन्म के 7 से 10 दिन के अंदर पेट के गोलकीड़ों (Toxocara) की दवा (पिपरैजीन सिरप) पिलाएं।\n\n"
            "**4. सूखा बिछावन व गर्माहट:**\n"
            "• बछड़े को सूखी पराली पर रखें और ठंडी हवा या बारिश के पानी से बचाएं। 2 सप्ताह बाद थोड़ा काफ स्टार्टर दाना शुरू करें।"
        ),
        "suggestions": [
            "बछड़े के दस्त का घरेलू उपचार क्या है?",
            "खुरपका रोग के क्या लक्षण हैं?",
            "दूध उत्पादन कैसे बढ़ाएं?",
            "गाय के हीट में आने के क्या लक्षण हैं?"
        ]
    },
    "breeding": {
        "reply": (
            "🐄 **पशु प्रजनन, हीट (मद) चक्र और गर्भावस्था प्रबंधन**\n\n"
            "**1. विभिन्न पशुओं का औसत गर्भकाल:**\n"
            "• **गाय:** ~283 दिन (लगभग 9 महीने 9 दिन)\n"
            "• **भैंस:** ~310 दिन (लगभग 10 महीने 10 दिन)\n"
            "• **बकरी व भेड़:** ~150 दिन (लगभग 5 महीने)\n\n"
            "**2. ड्राई पीरियड (दूध बंद करने का समय):**\n"
            "• ब्याने से **60 दिन पहले** दूध दुहना बंद कर दें ताकि लेवा (थन) की कोशिकाएं दोबारा स्वस्थ हो सकें।\n"
            "• ब्याने से 2-3 हफ्ते पहले अधिक कैल्शियम न दें ताकि दुग्ध ज्वर (Milk Fever) से बचाव हो सके।\n\n"
            "**3. ब्याने के प्रमुख लक्षण (24–48 घंटे पहले):**\n"
            "• पूंछ के दोनों तरफ की हड्डियां (लिगामेंट्स) धंस जाना।\n"
            "• लेवा (थन) पूरा फूल जाना और थनों में खीस उतर आना।\n"
            "• पशु का बेचैन होना, बार-बार उठना-बैठना और योनि से गाढ़ा लसदार स्राव आना।"
        ),
        "suggestions": [
            "दुग्ध ज्वर (Milk Fever) के लक्षण क्या हैं?",
            "नवजात बछड़े की देखभाल कैसे करें?",
            "ब्याने के बाद गाय को क्या खिलाएं?",
            "कीटोसिस के क्या लक्षण हैं?"
        ]
    },
    "heat_stress": {
        "reply": (
            "☀️ **गर्मियों में लू व हीट स्ट्रेस से पशुओं का बचाव**\n\n"
            "**1. भैंसों को नहलाना व पानी की व्यवस्था:**\n"
            "• भैंसों में पसीने की ग्रंथियां कम होती हैं और काली त्वचा धूप सोखती है। दिन में 2-3 बार नहलाएं या तालाब में छोड़ें।\n"
            "• बाड़े की छत पर पराली डालें या दोपहर 11 से 4 बजे तक पंखे/फॉगर्स चलाएं।\n\n"
            "**2. खान-पान का समय बदलें:**\n"
            "• भारी दाना और चारा सुबह 7 बजे से पहले और रात 8 बजे के बाद ठंडे समय में दें।\n"
            "• पीने के पानी में **मीठा सोडा (50-100 ग्राम)** और इलेक्ट्रोलाइट्स मिलाएं ताकि रूमेन में एसिडिटी न बने।\n\n"
            "**3. हीट स्ट्रोक के लक्षण:**\n"
            "• जीभ बाहर निकाल कर तेजी से हांफना, लार टपकना, तापमान 105°F से ऊपर जाना। सिर पर तुरंत सामान्य ठंडा पानी डालें।"
        ),
        "suggestions": [
            "गर्मी में दूध की पैदावार कैसे बनाए रखें?",
            "अफारा होने पर क्या करें?",
            "गलघोंटू (HS) रोग के क्या लक्षण हैं?",
            "पशुओं में टीकाकरण का समय"
        ]
    },
    "deworming": {
        "reply": (
            "🪱 **पशुओं में पेट के कीड़े मारने (डीवर्मिंग) का सही समय व नियम**\n\n"
            "**1. दवा देने की आवृत्ति:**\n"
            "• वयस्क गाय-भैंस: **साल में 3 से 4 बार** (विशेषकर मानसून से ठीक पहले और मानसून के बाद)।\n"
            "• भेड़-बकरियां: हर 2 से 3 महीने में (कृमि के प्रति बहुत संवेदनशील होती हैं)।\n\n"
            "**2. दवा बदल-बदल कर दें:**\n"
            "• हमेशा एक ही दवा न दें। अल्बेंडाजोल, फेनबेंडाजोल और आइवरमेक्टिन को बदल-बदल कर दें ताकि कीड़ों में प्रतिरोधक क्षमता न बने।\n"
            "• दवा हमेशा सुबह खाली पेट दें और पशु के सही वजन के अनुसार दें।\n\n"
            "**3. चराई के नियम:**\n"
            "• सुबह जब घास पर ओस हो, तब पशु को चरने न भेजें क्योंकि ओस की बूंदों में कृमि के लार्वा ऊपर आ जाते हैं।"
        ),
        "suggestions": [
            "पेट में कीड़े होने के क्या लक्षण हैं?",
            "खुरपका (FMD) का टीका कब लगवाएं?",
            "बछड़े को पहली डीवर्मिंग कब दें?",
            "दूध बढ़ाने के उपाय क्या हैं?"
        ]
    },
    "emergency_first_aid": {
        "reply": (
            "🚨 **फार्म पर आपातकालीन प्राथमिक उपचार (Emergency First Aid)**\n\n"
            "**1. तीव्र अफारा (पेट में गैस भर जाना):**\n"
            "• यदि बायां पेट फूलकर टाइट हो जाए और पशु सांस न ले पाए: मुंह में लकड़ी की खपच्ची बांधें।\n"
            "• 250 मिली सरसों का तेल, 25 ग्राम हींग और 50 ग्राम मीठा सोडा मिलाकर तुरंत पिलाएं।\n"
            "• दम घुटने पर आपात स्थिति में डॉक्टर बाईं कोख में ट्रोकार-कैन्युला लगाकर गैस निकालते हैं।\n\n"
            "**2. गाय ब्याने के बाद बैठ जाए (उठ न सके - Downer Cow):**\n"
            "• यह **दुग्ध ज्वर (कैल्शियम की कमी)** हो सकता है। पशु को जबरदस्ती खड़ा न करें।\n"
            "• मुलायम सूखा बिछावन दें और तुरंत पशुचिकित्सक से नस में कैल्शियम बोरो-ग्लूकोनेट (CBG) चढ़वाएं।\n\n"
            "**3. गंभीर चोट या खून बहना:**\n"
            "• साफ कपड़े या तौलिए से घाव को 10 मिनट तक दबाकर रखें और तुरंत डॉक्टर को बुलाएं।"
        ),
        "suggestions": [
            "अफारा का घरेलू उपचार क्या है?",
            "दुग्ध ज्वर (Milk Fever) के लक्षण क्या हैं?",
            "नजदीकी आपातकालीन पशु अस्पताल खोजें",
            "गलघोंटू (HS) रोग के क्या लक्षण हैं?"
        ]
    }
}


def _generate_hindi_reply(query: str, disease: Optional[Dict[str, Any]], intent: str, context: Optional[dict]) -> Dict[str, Any]:
    """Generates an accurate, structured veterinary advisory response in Hindi."""
    query_lower = query.lower().strip()

    # Greetings / Meta queries in Hindi
    if any(g in query_lower for g in ["hi", "hello", "namaste", "नमस्ते", "नमस्कार", "मदद", "help", "कौन हो", "सहायता"]):
        return {
            "reply": (
                "👋 **नमस्ते! मैं वेटबॉट (VetBot) हूँ**, आपका एआई पशु चिकित्सा सलाहकार।\n\n"
                "मैं आपकी इन विषयों पर सहायता कर सकता हूँ:\n"
                "- 🩺 **पशु रोग व लक्षण** (खुरपका, लंपी, थनैला, अफारा, गलघोंटू आदि)\n"
                "- 🌾 **आहार व पोषण प्रबंधन** (संतुलित चारा, दूध वृद्धि, मिनरल मिक्सचर)\n"
                "- 🍼 **नवजात बछड़ा देखभाल व खीस पिलाने के नियम**\n"
                "- 🐄 **प्रजनन, मद (हीट) चक्र व ब्याने के लक्षण**\n"
                "- ☀️ **गर्मियों में लू व हीट स्ट्रेस से बचाव**\n"
                "- 🪱 **टीकाकरण व पेट के कीड़े (डीवर्मिंग) का कैलेंडर**\n"
                "- 🚨 **फार्म पर आपातकालीन प्राथमिक उपचार**\n\n"
                "*सुझाव: नीचे दिए गए किसी भी सुझाव पर टैप करें या अपना प्रश्न सीधा पूछें!*"
            ),
            "suggestions": [
                "गाय-भैंस का दूध कैसे बढ़ाएं?",
                "गाय को अफारा है क्या करें?",
                "खुरपका रोग से बचाव कैसे करें?",
                "बछड़े को खीस कब और कितना पिलाएं?",
                "दुग्ध ज्वर (Milk Fever) के क्या लक्षण हैं?"
            ],
            "source": "local_kb_hi",
            "disease_identified": disease["id"] if disease else None
        }

    # If no disease identified, check topic or fallback
    if not disease:
        topic = _detect_topic(query)
        if topic and topic in TOPICS_HI:
            return {
                "reply": TOPICS_HI[topic]["reply"],
                "suggestions": TOPICS_HI[topic]["suggestions"],
                "source": "local_kb_hi",
                "disease_identified": None
            }

        return {
            "reply": (
                "पशु स्वास्थ्य और डेयरी फार्म प्रबंधन में आपकी मदद के लिए मैं यहाँ हूँ!\n\n"
                "आप मुझसे इनके बारे में पूछ सकते हैं:\n"
                "• **पशु रोग:** खुरपका-मुंहपका, लंपी स्किन, थनैला, अफारा, गलघोंटू, दुग्ध ज्वर, निमोनिया।\n"
                "• **डेयरी प्रबंधन:** दूध कैसे बढ़ाएं, बछड़े की देखभाल, गर्मी से बचाव, पेट के कीड़े की दवा।\n"
                "• **प्राथमिक उपचार:** पशु का पेट फूलने, ब्याने के बाद बैठ जाने या सांस फूलने पर क्या करें।\n\n"
                "कृपया अपने पशु, लक्षण या समस्या के बारे में बताएं।"
            ),
            "suggestions": [
                "दूध उत्पादन कैसे बढ़ाएं?",
                "गाय को अफारा होने पर क्या करें?",
                "नवजात बछड़े की देखभाल कैसे करें?",
                "लंपी वायरस कैसे फैलता है?",
                "ब्याने के बाद गाय बैठ जाए तो क्या करें?"
            ],
            "source": "local_kb_hi",
            "disease_identified": None
        }

    # Disease-specific Hindi Response
    d_id = disease["id"]
    hi = DISEASES_HI.get(d_id)
    d_name = hi["name"] if hi else disease["name"]
    risk = disease.get("risk_level", "medium").upper()
    risk_label = "उच्च (HIGH) ⚠️" if risk == "HIGH" else ("मध्यम (MEDIUM)" if risk == "MEDIUM" else "सामान्य (LOW)")

    reply_parts = []

    if intent == "emergency":
        reply_parts.append(f"🚨 **आपातकालीन सलाह: {d_name} (जोखिम स्तर: {risk_label})**")
        reply_parts.append(
            "यदि पशु अत्यधिक कष्ट में है, लेटा हुआ है (उठ नहीं पा रहा) या सांस लेने में भारी तकलीफ है, "
            "तो **बिना समय गंवाए तुरंत अधिकृत पशुचिकित्सक से संपर्क करें**।\n"
        )
        if hi:
            reply_parts.append(f"**तुरंत की जाने वाली सहायता:**\n{hi['supportive_care']}\n")
            reply_parts.append(f"**आपातकालीन आहार सावधानी:**\n{hi['diet']}")
        else:
            reply_parts.append(f"**Immediate Supportive Actions:**\n{disease.get('supportive_care')}\n")
        suggestions = [
            f"{d_name} में क्या देखभाल करें?",
            f"{d_name} के लिए क्या इलाज है?",
            "नजदीकी आपातकालीन पशु अस्पताल खोजें"
        ]

    elif intent == "diet":
        reply_parts.append(f"🌾 **खान-पान व आहार सलाह: {d_name}**")
        if hi:
            reply_parts.append(f"**अनुशंसित आहार:**\n{hi['diet']}\n")
            reply_parts.append(f"**घरेलू देखभाल:**\n{hi['supportive_care']}")
        else:
            reply_parts.append(f"**Recommended Diet:**\n{disease.get('diet')}\n")
        suggestions = [
            f"{d_name} में क्या देखभाल करें?",
            f"{d_name} से बचाव कैसे करें?",
            f"{d_name} ठीक होने में कितना समय लगता है?"
        ]

    elif intent == "treatment":
        reply_parts.append(f"💊 **चिकित्सा व उपचार संबंधी सलाह: {d_name}**")
        if hi:
            reply_parts.append(f"**उपचार विवरण:**\n{hi['treatment_overview']}\n")
            reply_parts.append(f"**घरेलू सहायक देखभाल:**\n{hi['supportive_care']}\n")
        else:
            reply_parts.append(f"**Treatment Overview:**\n{disease.get('treatment_overview')}\n")
        reply_parts.append(
            "> ⚠️ **सावधानी:** एंटीबायोटिक या अन्य गंभीर दवाएं केवल पंजीकृत पशुचिकित्सक की देखरेख में ही दें ताकि दवाओं का दुष्प्रभाव न हो।"
        )
        suggestions = [
            f"{d_name} में क्या चारा दें?",
            f"{d_name} से बचाव कैसे करें?",
            f"{d_name} का टीका कब लगवाएं?"
        ]

    elif intent == "prevention":
        reply_parts.append(f"🛡️ **रोकथाम व टीकाकरण: {d_name}**")
        if hi:
            reply_parts.append(f"**बचाव के उपाय:**\n{hi['prevention']}\n")
        else:
            reply_parts.append(f"**Prevention:**\n{disease.get('prevention')}\n")
        suggestions = [
            f"{d_name} के शुरुआती लक्षण क्या हैं?",
            f"{d_name} में क्या आहार दें?",
            f"{d_name} का घरेलू इलाज क्या है?"
        ]

    else:
        # Comprehensive Overview in Hindi
        reply_parts.append(f"📋 **रोग विवरण: {d_name}**")
        if hi:
            reply_parts.append(f"**रोग परिचय:** {hi['definition']}\n")
            reply_parts.append(f"**जोखिम स्तर:** `{risk_label}`\n")
            reply_parts.append(f"**त्वरित घरेलू व सहायक देखभाल:**\n{hi['supportive_care']}\n")
            reply_parts.append(f"**आहार व खान-पान:**\n{hi['diet']}\n")
            reply_parts.append(f"**उपचार संबंधी निर्देश:**\n{hi['treatment_overview']}\n")
            reply_parts.append(f"**रोकथाम व बचाव:**\n{hi['prevention']}\n")
            reply_parts.append(f"**स्वस्थ होने का अनुमानित समय:** {hi['recovery_time']}")
        else:
            reply_parts.append(f"**Definition:** {disease.get('definition')}\n")
            reply_parts.append(f"**Supportive Care:**\n{disease.get('supportive_care')}\n")
            reply_parts.append(f"**Diet:**\n{disease.get('diet')}\n")

        suggestions = [
            f"{d_name} में क्या चारा दें?",
            f"{d_name} का मेडिकल इलाज क्या है?",
            f"{d_name} से बचाव कैसे करें?",
            f"{d_name} के टीके के बारे में बताएं"
        ]

    reply_parts.append("\n\n*वेटएआई (VetAI) शैक्षणिक मार्गदर्शन प्रदान करता है। गंभीर स्थिति में तुरंत अधिकृत पशुचिकित्सक से परामर्श लें।*")

    return {
        "reply": "\n".join(reply_parts),
        "suggestions": suggestions,
        "source": "local_kb_hi",
        "disease_identified": disease["id"]
    }


# ==========================================
# LOCAL OFFLINE LLM (OLLAMA ON NVIDIA GPU)
# ==========================================
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")


def check_ollama_status() -> Dict[str, Any]:
    """Checks if local Ollama daemon is reachable and which models are installed."""
    try:
        url = f"{OLLAMA_HOST}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "VetAI/1.0"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name") for m in data.get("models", [])]
                preferred = OLLAMA_MODEL
                active_model = None
                for m in models:
                    if preferred in m or m.startswith(preferred.split(":")[0]):
                        active_model = m
                        break
                if not active_model and models:
                    active_model = models[0]
                return {
                    "available": True,
                    "model": active_model or preferred,
                    "all_models": models,
                    "device": "NVIDIA GeForce RTX 3050 (CUDA Offload)",
                    "mode": "ollama_gpu"
                }
    except Exception:
        pass
    return {
        "available": False,
        "model": None,
        "all_models": [],
        "device": "CPU / Local Knowledge Engine",
        "mode": "rule_engine"
    }


def _try_ollama_llm(query: str, disease: Optional[Dict[str, Any]], context: Optional[dict] = None, history: Optional[List[dict]] = None) -> Optional[Dict[str, Any]]:
    """Generates an answer using local Ollama LLM accelerated by NVIDIA GPU with RAG grounding."""
    status = check_ollama_status()
    if not status["available"]:
        return None

    model_name = status["model"]
    is_hi = _is_hindi(query)

    # 1. Build RAG Grounding context from local database
    grounding_parts = []
    if disease:
        grounding_parts.append(
            f"CLINICAL KNOWLEDGE BASE FACTS FOR {disease['name']}:\n"
            f"- Definition: {disease.get('definition')}\n"
            f"- Key Symptoms: {', '.join(disease.get('key_symptoms', []))}\n"
            f"- Recommended Diet: {disease.get('diet')}\n"
            f"- Supportive Nursing & Hygiene: {disease.get('supportive_care')}\n"
            f"- Standard Veterinary Treatment Protocol: {disease.get('treatment_overview')}\n"
            f"- Biosecurity & Prevention: {disease.get('prevention')}\n"
            f"- Recovery Timeline: {disease.get('recovery_time')}"
        )

    if context:
        patient_info = []
        if context.get("animal_type"): patient_info.append(f"Species: {context['animal_type']}")
        if context.get("breed"): patient_info.append(f"Breed: {context['breed']}")
        if context.get("age_months"): patient_info.append(f"Age: {context['age_months']} months")
        if context.get("weight_kg"): patient_info.append(f"Weight: {context['weight_kg']} kg")
        if context.get("tag_number"): patient_info.append(f"Pashu Aadhaar Tag: {context['tag_number']}")
        if context.get("symptoms"): patient_info.append(f"Observed Signs: {', '.join(context['symptoms'])}")
        if patient_info:
            grounding_parts.append("PATIENT LIVESTOCK RECORD:\n" + "\n".join(patient_info))

    grounding_text = "\n\n".join(grounding_parts)

    # 2. Safety-guided system prompt
    if is_hi:
        system_prompt = (
            "आप 'वेटबॉट' (VetBot) हैं — भारतीय डेयरी किसानों और पशुपालकों के लिए समर्पित एक विशेषज्ञ, सहानुभूतिपूर्ण पशु चिकित्सा AI सहायक।\n"
            "नियम:\n"
            "1. सरल, स्पष्ट, व्यावहारिक हिंदी (देवनागरी लिपि) में उत्तर दें।\n"
            "2. उत्तर को सुव्यवस्थित शीर्षकों (बुलेट पॉइंट्स) में प्रस्तुत करें।\n"
            "3. पशु चिकित्सा के वैज्ञानिक तथ्यों पर आधारित रहें। बिना जांच के मनगढ़ंत एंटीबायोटिक या जहरीली दवाओं के नाम न सुझाएं।\n"
            "4. गंभीर आपात स्थिति (जैसे पेट में अत्यधिक गैस/अफारा, गर्भपात, जहर) में तुरंत सरकारी पशु एम्बुलेंस हेल्पलाइन '1962' पर कॉल करने की सलाह दें।\n"
            "5. अंत में पंजीकृत पशु चिकित्सक से संपर्क करने की सलाह दें।\n\n"
            + (f"सत्यापित संदर्भ तथ्य:\n{grounding_text}\n" if grounding_text else "")
        )
    else:
        system_prompt = (
            "You are VetBot, an empathetic, highly knowledgeable veterinary AI assistant for livestock farmers and clinicians.\n"
            "Guidelines:\n"
            "1. Provide direct, practical, structured advice in clean Markdown with concise headings.\n"
            "2. Strictly adhere to verified veterinary medicine facts. Do not invent unverified drug dosages or recommend human medicines toxic to animals.\n"
            "3. If critical signs are present (acute bloat, choking, prolapse), instruct the user to immediately call emergency animal services (1962 in India).\n"
            "4. Emphasize hygiene, supportive nutrition, and consulting a local licensed veterinarian for prescription therapeutics.\n\n"
            + (f"VERIFIED CLINICAL GROUNDING FACTS:\n{grounding_text}\n" if grounding_text else "")
        )

    # 3. Message sequence with optional multi-turn history
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        for msg in history[-4:]:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": query})

    try:
        url = f"{OLLAMA_HOST}/api/chat"
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.4,
                "top_p": 0.9,
                "num_predict": 450
            }
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=18) as resp:
            if resp.status == 200:
                result = json.loads(resp.read().decode("utf-8"))
                reply_text = result.get("message", {}).get("content", "").strip()
                if reply_text:
                    if disease and not any(k in reply_text.lower() for k in [disease['name'].lower(), disease['id']]):
                        if is_hi:
                            hi_name = DISEASES_HI.get(disease["id"], {}).get("name", disease["name"])
                            reply_text = f"🩺 **परामर्श: {hi_name}**\n\n" + reply_text
                        else:
                            reply_text = f"🩺 **Clinical Advisory: {disease['name']}**\n\n" + reply_text
                    if is_hi:
                        suggestions = [
                            f"{disease['name']} का आहार क्या हो?" if disease else "दूध बढ़ाने का संतुलित आहार",
                            "बचाव के मुख्य टीके",
                            "पशु चिकित्सक कब बुलाएं?"
                        ]
                    else:
                        suggestions = [
                            f"Diet & feed for {disease['name']}" if disease else "How to increase milk yield",
                            "Vaccination & prevention tips",
                            "Emergency first aid signs"
                        ]
                    return {
                        "reply": reply_text,
                        "suggestions": suggestions,
                        "source": "ollama_gpu",
                        "model": model_name,
                        "disease_identified": disease["id"] if disease else None
                    }
    except Exception:
        # Fallback takes over
        pass

    return None


def _try_online_llm(query: str, disease: Optional[Dict[str, Any]], history: Optional[List[dict]] = None) -> Optional[str]:
    """Optional online LLM RAG enhancement if API keys are set in the environment."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not gemini_key and not openai_key:
        return None

    grounding = ""
    if disease:
        grounding = (
            f"GROUND TRUTH VETERINARY FACTS FOR {disease['name']}:\n"
            f"- Definition: {disease.get('definition')}\n"
            f"- Key Symptoms: {', '.join(disease.get('key_symptoms', []))}\n"
            f"- Diet: {disease.get('diet')}\n"
            f"- Supportive Care: {disease.get('supportive_care')}\n"
            f"- Treatment Overview: {disease.get('treatment_overview')}\n"
            f"- Prevention: {disease.get('prevention')}\n"
            f"- Recovery Time: {disease.get('recovery_time')}\n"
        )

    system_prompt = (
        "You are VetBot, an empathetic, highly knowledgeable veterinary AI assistant for livestock farmers. "
        "Provide direct, practical, structured advice in clear Markdown. "
        "Strictly adhere to verified veterinary medicine facts. Do not invent drug dosages or unverified remedies. "
        "Always include a concise note to consult a licensed local veterinarian for prescription drugs.\n\n"
        + grounding
    )

    if gemini_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            payload = {
                "contents": [
                    {"role": "user", "parts": [{"text": f"{system_prompt}\n\nFarmer's Question: {query}"}]}
                ]
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            pass

    if openai_key:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.3
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai_key}"
                }
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except Exception:
            pass

    return None


def ask_vetbot(query: str, context: Optional[dict] = None, history: Optional[List[dict]] = None) -> Dict[str, Any]:
    """Main entrypoint for chatbot queries with multi-tier routing (Ollama GPU -> Online LLM -> Local Rule Engine)."""
    disease = _detect_disease(query, context)
    intent = _detect_intent(query)
    is_hi = _is_hindi(query)

    # 1. Try Local GPU Ollama first (Fastest, unconstrained, offline on RTX 3050)
    ollama_res = _try_ollama_llm(query, disease, context, history)
    if ollama_res:
        return ollama_res

    # 2. Try Online Cloud LLM if keys provided (Gemini / OpenAI)
    online_reply = _try_online_llm(query, disease, history)
    if online_reply:
        suggestions = [
            f"What is the recommended diet for {disease['name']}?",
            f"What supportive care is needed for {disease['name']}?",
            f"How to prevent {disease['name']}?"
        ] if disease else [
            "How to increase milk yield?",
            "What diet is best for bloat?",
            "Newborn calf care tips",
            "Emergency first aid for downer cows"
        ]
        return {
            "reply": online_reply,
            "suggestions": suggestions,
            "source": "online_llm",
            "disease_identified": disease["id"] if disease else None
        }

    # 3. Instant Deterministic Local Knowledge Engine Fallback (Zero Latency, 100% Reliable)
    if is_hi:
        return _generate_hindi_reply(query, disease, intent, context)

    return _generate_local_reply(query, disease, intent, context)

