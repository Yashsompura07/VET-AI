"""Unit & Integration tests for the bilingual VetBot conversational engine."""
import pytest
from app import chatbot


def test_intent_detection_english():
    """Verify English intent detection for varied livestock inquiries."""
    assert chatbot._detect_intent("What is the cure and treatment for mastitis?") == "treatment"
    assert chatbot._detect_intent("When should I give the FMD vaccine for prevention?") == "prevention"
    assert chatbot._detect_intent("My animal is collapsed and dying, emergency!") == "emergency"
    assert chatbot._detect_intent("What are the symptoms and signs of bloat?") == "symptoms"


def test_intent_detection_hindi():
    """Verify Hindi intent detection for Devanagari queries."""
    assert chatbot._is_hindi("गाय को अफारा है क्या करें") is True
    assert chatbot._detect_intent("गाय का इलाज और दवा क्या है") == "treatment"
    assert chatbot._detect_intent("खुरपका रोग से बचाव का टीका कब लगाएं") == "prevention"


def test_disease_entity_extraction_english():
    """Verify disease entity extraction from queries."""
    d_lsd = chatbot._detect_disease("How do I identify lumpy skin disease nodules?")
    assert d_lsd is not None and d_lsd["id"] == "lsd"

    d_bloat = chatbot._detect_disease("What causes acute bloat in calves?")
    assert d_bloat is not None and d_bloat["id"] == "bloat"

    d_hs = chatbot._detect_disease("Signs of hemorrhagic septicemia in buffaloes")
    assert d_hs is not None and d_hs["id"] == "hs"


def test_disease_entity_extraction_hindi():
    """Verify Hindi disease name recognition."""
    d_hs = chatbot._detect_disease("मेरी भैंस को गलघोंटू हो गया है")
    assert d_hs is not None and d_hs["id"] == "hs"

    d_mast = chatbot._detect_disease("गाय के थनों में सूजन और थनैला रोग के लक्षण")
    assert d_mast is not None and d_mast["id"] == "mastitis"

    d_fmd = chatbot._detect_disease("खुरपका-मुंहपका रोग से बचाव")
    assert d_fmd is not None and d_fmd["id"] == "fmd"


def test_ask_vetbot_english_fmd():
    """Verify English VetBot response for FMD."""
    res = chatbot.ask_vetbot("What should I do if my cow has foot and mouth disease?")
    assert res["disease_identified"] == "fmd"
    assert "Foot and Mouth Disease" in res["reply"] or "FMD" in res["reply"]
    assert len(res["suggestions"]) > 0


def test_ask_vetbot_hindi_bloat():
    """Verify Hindi VetBot response for Bloat."""
    res = chatbot.ask_vetbot("गाय को अफारा है क्या करें")
    assert res["disease_identified"] == "bloat"
    assert "अफारा" in res["reply"] or "Bloat" in res["reply"]
    assert res.get("source") in ["ollama_gpu", "local_kb_hi"]
    assert len(res["suggestions"]) > 0


def test_ask_vetbot_hindi_dairy_topics():
    """Verify Hindi general husbandry advice for milk yield and calf care."""
    res_milk = chatbot.ask_vetbot("गाय-भैंस का दूध कैसे बढ़ाएं")
    assert any(w in res_milk["reply"] for w in ["दूध", "चारा", "आहार", "भैंस", "गाय", "खाना", "पशु"])

    res_calf = chatbot.ask_vetbot("नवजात बछड़े की देखभाल और खीस पिलाना")
    assert any(w in res_calf["reply"] for w in ["बछड़े", "बछड़ा", "खीस", "दूध", "पशु", "देखभाल"])


def test_ask_vetbot_with_prior_diagnosis_context():
    """Verify that VetBot leverages prior diagnosis context when answering follow-up questions."""
    ctx = {
        "disease_id": "mastitis",
        "animal_type": "cow",
        "name": "Mastitis"
    }
    res = chatbot.ask_vetbot("Can I still drink or sell the milk?", context=ctx)
    assert res["disease_identified"] == "mastitis"
    assert "milk" in res["reply"].lower()
