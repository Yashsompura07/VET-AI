"""Generate a clean, professional, bilingual (English / Hindi) PDF diagnosis report.
Uses pure-Python fpdf2 with Mukta Unicode font support for full Devanagari script rendering.
"""

import base64
import io
import os
import tempfile
from datetime import datetime
from app.chatbot import DISEASES_HI

PRIMARY = (15, 90, 71)         # VetAI Forest Green #0F5A47
PRIMARY_DARK = (10, 62, 49)    # #0A3E31
INK = (15, 23, 42)             # Slate Ink #0F172A
MUTED = (100, 116, 139)        # Slate Muted #64748B
BORDER = (226, 232, 240)       # Light Border
RISK = {
    "high": (220, 38, 38),     # #DC2626
    "medium": (217, 119, 6),   # #D97706
    "low": (22, 163, 74)       # #16A34A
}

ANIMAL_TYPES_HI = {
    "cow": "गाय (Cow / Cattle)",
    "buffalo": "भैंस (Buffalo)",
    "goat": "बकरी (Goat)",
    "sheep": "भेड़ (Sheep)"
}

RISK_COPY_HI = {
    "high": "उच्च जोखिम (HIGH RISK) — तुरंत आपातकालीन पशुचिकित्सक को बुलाएं",
    "medium": "मध्यम जोखिम (MEDIUM RISK) — निरंतर निगरानी रखें व पशुचिकित्सक से संपर्क करें",
    "low": "सामान्य जोखिम (LOW RISK) — पशु की सामान्य देखरेख व पोषण जारी रखें"
}

FONTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "fonts")
MUKTA_REG = os.path.join(FONTS_DIR, "Mukta-Regular.ttf")
MUKTA_BOLD = os.path.join(FONTS_DIR, "Mukta-Bold.ttf")


def build_pdf(p):
    from fpdf import FPDF

    lang = p.get("lang", "en")
    is_hi = (lang == "hi")

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    W = pdf.w - 2 * pdf.l_margin

    # Register Unicode font if present
    use_unicode = os.path.exists(MUKTA_REG) and os.path.exists(MUKTA_BOLD)
    if use_unicode:
        pdf.add_font("Mukta", "", MUKTA_REG)
        pdf.add_font("Mukta", "B", MUKTA_BOLD)
        font_family = "Mukta"
    else:
        font_family = "Helvetica"

    def clean_txt(t):
        if t is None:
            return ""
        t = str(t)
        if not use_unicode:
            for a, b in [("\u2014", "-"), ("\u2013", "-"), ("\u2019", "'"),
                         ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"'),
                         ("\u2026", "..."), ("\u00b0", " deg ")]:
                t = t.replace(a, b)
            return t.encode("latin-1", "replace").decode("latin-1")
        return t

    def h_sec(txt, size=11, color=PRIMARY, style="B"):
        pdf.set_font(font_family, style, size)
        pdf.set_text_color(*color)
        pdf.cell(0, size * 0.5 + 4, clean_txt(txt), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    def kv(label, value):
        pdf.set_x(pdf.l_margin)
        pdf.set_font(font_family, "B", 9.5)
        pdf.set_text_color(*MUTED)
        pdf.write(5.5, clean_txt(label) + ":  ")
        pdf.set_font(font_family, "", 9.5)
        pdf.set_text_color(*INK)
        pdf.write(5.5, clean_txt(value))
        pdf.ln(6)

    def para(label, value):
        if not value:
            return
        pdf.set_font(font_family, "B", 9.5)
        pdf.set_text_color(*PRIMARY_DARK)
        pdf.cell(0, 5, clean_txt(label), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(font_family, "", 9)
        pdf.set_text_color(*INK)
        pdf.multi_cell(0, 4.5, clean_txt(value))
        pdf.ln(2)

    # 1. Header Banner
    pdf.set_fill_color(*PRIMARY)
    pdf.rect(pdf.l_margin, pdf.get_y(), W, 14, "F")
    pdf.set_xy(pdf.l_margin + 4, pdf.get_y() + 3)
    pdf.set_font(font_family, "B", 14)
    pdf.set_text_color(255, 255, 255)
    header_title = "VetAI — पशु रोग नैदानिक रिपोर्ट (Clinical Assessment Report)" if is_hi else "VetAI — AI Veterinary Clinical Assessment Report"
    pdf.cell(0, 8, clean_txt(header_title))
    pdf.ln(17)

    # Timestamp
    pdf.set_font(font_family, "", 8.5)
    pdf.set_text_color(*MUTED)
    date_str = datetime.now().strftime("%d %b %Y, %I:%M %p")
    pdf.cell(0, 4, clean_txt(f"तारीख व समय (Generated): {date_str}" if is_hi else f"Generated: {date_str}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # 2. Animal Metadata Card
    h_sec("१. पशु विवरण (Animal Profile)" if is_hi else "1. Animal Profile")
    raw_atype = str(p.get("animal_type", "-")).lower()
    disp_type = ANIMAL_TYPES_HI.get(raw_atype, raw_atype.title()) if is_hi else raw_atype.title()
    kv("पशु प्रकार (Species)" if is_hi else "Species / Animal Type", disp_type)

    if p.get("tag_number") or p.get("animal_tag"):
        tag_val = p.get("tag_number") or p.get("animal_tag")
        kv("इयर टैग / पशु आधार (Pashu Aadhaar)" if is_hi else "Ear Tag / Pashu Aadhaar", str(tag_val))

    if p.get("breed"):
        kv("नस्ल (Breed)" if is_hi else "Breed", str(p["breed"]))

    if p.get("age_months"):
        kv("उम्र (Age)" if is_hi else "Age", f"{p['age_months']} महीने (Months)" if is_hi else f"{p['age_months']} months")

    if p.get("weight_kg"):
        kv("वजन (Weight)" if is_hi else "Weight", f"{p['weight_kg']} कि.ग्रा. (kg)")

    vitals = []
    if p.get("temp_c"):
        vitals.append(f"तापमान: {p['temp_c']} °C" if is_hi else f"Temp: {p['temp_c']} °C")
    if p.get("heart_rate"):
        vitals.append(f"हृदय गति: {p['heart_rate']} bpm" if is_hi else f"HR: {p['heart_rate']} bpm")
    if vitals:
        kv("शारीरिक माप (Vitals)" if is_hi else "Vitals", ", ".join(vitals))

    if p.get("symptom_labels"):
        kv("पहचाने गए लक्षण (Observable Symptoms)" if is_hi else "Reported Symptoms", ", ".join(p["symptom_labels"]))
    pdf.ln(3)

    # 3. Diagnosis Assessment
    res = p.get("result", {})
    top = res.get("top", {})
    risk = top.get("risk_level", "low")
    disease_id = top.get("disease_id") or top.get("id")

    # Knowledge content
    kb_en = res.get("knowledge", {})
    kb_hi = res.get("knowledge_hi") or DISEASES_HI.get(disease_id, {})

    h_sec("२. नैदानिक मूल्यांकन (Diagnostic Assessment)" if is_hi else "2. Diagnostic Assessment")
    
    # Disease Name
    disease_name = kb_hi.get("name") if (is_hi and kb_hi.get("name")) else top.get("name", "—")
    pdf.set_font(font_family, "B", 13)
    pdf.set_text_color(*INK)
    pdf.cell(0, 7, clean_txt(disease_name), new_x="LMARGIN", new_y="NEXT")

    # Risk & Confidence
    pdf.set_font(font_family, "B", 10)
    pdf.set_text_color(*RISK.get(risk, INK))
    conf_val = top.get("confidence", 0)
    if is_hi:
        risk_str = f"आत्मविश्वास: {conf_val}%  ·  {RISK_COPY_HI.get(risk, risk.upper())}"
    else:
        risk_str = f"{conf_val}% Confidence  ·  {risk.upper()} RISK TIER"
    pdf.cell(0, 6, clean_txt(risk_str), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Decision Rationale
    summary_text = res.get("explanation", {}).get("summary", "")
    if is_hi:
        summary_text = f"यह निष्कर्ष लक्षणों के पैटर्न, शारीरिक संकेतों और घाव/त्वचा विश्लेषण के आधार पर निकाला गया है। संभावित रोग: {disease_name} ({conf_val}% संभावना)।"
    para("नैदानिक आधार (Clinical Rationale)" if is_hi else "Clinical Rationale", summary_text)

    # Hybrid fusion note
    comp = res.get("components", {})
    if res.get("mode") == "hybrid" and comp.get("image"):
        f_txt = (f"लक्षण मॉडल संकेत: {comp['symptom']['name']} ({comp['symptom']['confidence']}%). "
                 f"कैमरा विज़न विश्लेषण: {comp['image']['class']} ({comp['image']['confidence']}%). "
                 f"भारित संलयन (Weighted Late Fusion) द्वारा संयुक्त शीर्ष परिणाम निर्धारित किया गया।") if is_hi else (
                 f"Symptoms model indicated {comp['symptom']['name']} ({comp['symptom']['confidence']}%). "
                 f"Camera vision indicated '{comp['image']['class']}' ({comp['image']['confidence']}%). "
                 f"Weighted late fusion combined both modalities.")
        para("मॉडल संलयन वास्तुकला (Decision Fusion)" if is_hi else "Decision Fusion Architecture", f_txt)
    elif res.get("mode") == "image_only" and comp.get("image"):
        img_txt = (f"त्वचा व घाव की फोटो का कंप्यूटर विज़न वर्गीकरण (MobileNetV2): {disease_name} ({conf_val}% आत्मविश्वास)।") if is_hi else (
                  f"Visual photo classification on cattle lesion/skin: identified '{comp['image']['class']}' ({conf_val}% confidence).")
        para("कंप्यूटर विज़न फोटो विश्लेषण (Vision Analysis)" if is_hi else "Visual Photo Classification", img_txt)

    # Grad-CAM Heatmap Image
    heat = res.get("heatmap")
    if heat and heat.startswith("data:image"):
        try:
            raw = base64.b64decode(heat.split(",", 1)[1])
            tmp = os.path.join(tempfile.gettempdir(), f"vetai_heat_{os.getpid()}.png")
            with open(tmp, "wb") as f:
                f.write(raw)
            pdf.set_font(font_family, "B", 9)
            pdf.set_text_color(*MUTED)
            pdf.cell(0, 5, clean_txt("फोटो विश्लेषण विज़ुअलाइज़ेशन (GRAD-CAM LESION HEATMAP)" if is_hi else "VISUAL EXPLANATION (GRAD-CAM LESION HEATMAP)"), new_x="LMARGIN", new_y="NEXT")
            pdf.image(tmp, w=50)
            pdf.ln(3)
        except Exception:
            pass

    # 4. Clinical Knowledge Guidance
    h_sec("३. उपचार व प्रबंधन दिशा-निर्देश (Clinical Protocol)" if is_hi else "3. Veterinary Guidance & Farm Protocols")
    active_kb = kb_hi if is_hi else kb_en
    para("रोग परिचय (Disease Overview)" if is_hi else "Pathology & Overview", active_kb.get("definition") or kb_en.get("definition"))
    para("तुरंत घरेलू देखभाल (Supportive Farm Care)" if is_hi else "Immediate Supportive Care", active_kb.get("supportive_care") or kb_en.get("supportive_care"))
    para("अनुशंसित आहार व पोषण (Diet & Nutrition)" if is_hi else "Recommended Diet", active_kb.get("diet") or kb_en.get("diet"))
    para("पशुचिकित्सा उपचार (Veterinary Treatment)" if is_hi else "Veterinary Treatment Overview", active_kb.get("treatment_overview") or kb_en.get("treatment_overview"))
    para("रोकथाम व जैव-सुरक्षा (Prevention & Biosecurity)" if is_hi else "Prevention & Vaccination", active_kb.get("prevention") or kb_en.get("prevention"))
    para("स्वस्थ होने का अनुमानित समय (Estimated Recovery)" if is_hi else "Estimated Recovery Time", active_kb.get("recovery_time") or kb_en.get("recovery_time"))

    # 5. Emergency Contacts & Disclaimer Footer
    pdf.ln(3)
    pdf.set_draw_color(*MUTED)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + W, pdf.get_y())
    pdf.ln(2)

    pdf.set_font(font_family, "B", 8.5)
    pdf.set_text_color(*PRIMARY)
    em_text = "आपातकालीन हेल्पलाइन: राष्ट्रीय पशु एम्बुलेंस: 1962 (टोल-फ्री)  |  किसान कॉल सेंटर: 1800-180-1551" if is_hi else "Emergency Support: National Livestock Ambulance: 1962 (Toll-Free) | Kisan Helpline: 1800-180-1551"
    pdf.cell(0, 4.5, clean_txt(em_text), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font(font_family, "", 8)
    pdf.set_text_color(*MUTED)
    disclaimer = (
        "महत्वपूर्ण सूचना: VetAI एक एआई-आधारित शैक्षणिक एवं प्रारंभिक ट्राइएज सहायक प्रणाली है। "
        "यह किसी अधिकृत पशुचिकित्सक के अंतिम निदान या पर्चे का विकल्प नहीं है। गंभीर स्थिति में तुरंत नजदीकी पशु अस्पताल से संपर्क करें।"
        if is_hi else
        "Legal & Clinical Disclaimer: VetAI provides informational AI guidance and educational triage indications only. "
        "It does not replace professional veterinary diagnosis or prescription. In critical emergencies, consult an authorized veterinarian immediately."
    )
    pdf.multi_cell(0, 4, clean_txt(disclaimer))

    out = pdf.output()
    return bytes(out) if isinstance(out, (bytes, bytearray)) else out.encode("utf-8")
