"""VetAI Veterinary Clinic & Emergency Directory.

Provides:
- Directory of veterinary hospitals, polyclinics, and emergency dispensaries.
- Geolocation distance sorting (Haversine formula).
- Keyword search by district, city, state, or service.
- Emergency hotlines (e.g. 1962 Animal Ambulance).
"""

import math
from typing import List, Dict, Any, Optional

EMERGENCY_HOTLINES = [
    {
        "name": "1962 National Animal Emergency & Ambulance Helpline",
        "phone": "1962",
        "toll_free": True,
        "hours": "24/7",
        "description": "Government mobile veterinary clinic & emergency livestock helpline."
    },
    {
        "name": "Kisan Call Centre (Animal Husbandry Division)",
        "phone": "18001801551",
        "toll_free": True,
        "hours": "6:00 AM - 10:00 PM",
        "description": "Toll-free expert veterinary advisory and disease alert support."
    }
]

VET_CLINICS: List[Dict[str, Any]] = [
    # --- Gujarat Livestock Hubs ---
    {
        "id": "vet_ahmedabad_1",
        "name": "District Veterinary Center & Polyclinic",
        "city": "Ahmedabad",
        "state": "Gujarat",
        "address": "Opp. Polytechnic, Ambawadi, Ahmedabad, Gujarat 380015",
        "lat": 23.0225,
        "lon": 72.5450,
        "phone": "+91 79 2630 1144",
        "emergency": True,
        "hours": "24 Hours (Emergency)",
        "services": ["Emergency C-Section", "Haemorrhagic Septicaemia Treatment", "Lumpy Skin Isolation", "Vaccination", "Mobile Unit"]
    },
    {
        "id": "vet_anand_1",
        "name": "College of Veterinary Science & Animal Hospital (AAU)",
        "city": "Anand",
        "state": "Gujarat",
        "address": "Anand Agricultural University Campus, Anand, Gujarat 388001",
        "lat": 22.5560,
        "lon": 72.9510,
        "phone": "+91 2692 261 486",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["Bovine ICU", "Ultrasonography", "Bloat Decompression", "FMD Diagnostic Lab", "Livestock Ambulance"]
    },
    {
        "id": "vet_anand_2",
        "name": "Amul Dairy Livestock Emergency Unit",
        "city": "Anand",
        "state": "Gujarat",
        "address": "Amul Dairy Road, Anand, Gujarat 388001",
        "lat": 22.5645,
        "lon": 72.9289,
        "phone": "+91 2692 256 124",
        "emergency": True,
        "hours": "24 Hours (Emergency)",
        "services": ["Emergency Dystocia Relief", "Doorstep Vet Ambulance", "Mastitis Treatment", "Preventive Vaccination"]
    },
    {
        "id": "vet_gandhinagar_1",
        "name": "State Veterinary Polyclinic & Hospital",
        "city": "Gandhinagar",
        "state": "Gujarat",
        "address": "Sector 28, Near GIDC, Gandhinagar, Gujarat 382028",
        "lat": 23.2384,
        "lon": 72.6578,
        "phone": "+91 79 2325 4560",
        "emergency": True,
        "hours": "24 Hours (Emergency)",
        "services": ["Emergency Surgical Unit", "Digital Radiography", "Routine Immunization", "Blood Testing"]
    },
    {
        "id": "vet_mehsana_1",
        "name": "Dudhsagar Dairy Animal Healthcare Center & Polyclinic",
        "city": "Mehsana",
        "state": "Gujarat",
        "address": "Dudhsagar Dairy Highway Campus, Mehsana, Gujarat 384002",
        "lat": 23.5980,
        "lon": 72.3693,
        "phone": "+91 2762 253 201",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["Emergency Rumen Surgery", "Mobile Vet Dispensary", "Lumpy Skin Ward", "Deworming Supplies"]
    },
    {
        "id": "vet_mehsana_2",
        "name": "Government Veterinary Polyclinic",
        "city": "Mehsana",
        "state": "Gujarat",
        "address": "Near District Court, Mehsana, Gujarat 384001",
        "lat": 23.6015,
        "lon": 72.3995,
        "phone": "+91 2762 245 810",
        "emergency": False,
        "hours": "8:30 AM - 6:00 PM (On-call 24x7)",
        "services": ["OPD Consultation", "FMD Vaccination", "Wound Dressing", "Artificial Insemination"]
    },
    {
        "id": "vet_surat_1",
        "name": "Surat District Veterinary Polyclinic",
        "city": "Surat",
        "state": "Gujarat",
        "address": "Athwa Lines, Near District Panchayat, Surat, Gujarat 395001",
        "lat": 21.1702,
        "lon": 72.8095,
        "phone": "+91 261 247 1820",
        "emergency": True,
        "hours": "24 Hours (Emergency)",
        "services": ["Emergency Trauma & Fractures", "Bovine C-Section", "Microbiology Lab", "Ambulance 1962"]
    },
    {
        "id": "vet_surat_2",
        "name": "Sumul Dairy Veterinary Clinical Hospital",
        "city": "Surat",
        "state": "Gujarat",
        "address": "Sumul Dairy Road, Surat, Gujarat 395008",
        "lat": 21.1450,
        "lon": 72.8410,
        "phone": "+91 261 253 7631",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["Round-the-clock Emergency", "Milk Quality & Mastitis Care", "Field Veterinary Service", "Inpatient Sheds"]
    },
    {
        "id": "vet_vadodara_1",
        "name": "Vadodara Veterinary Polyclinic & Hospital",
        "city": "Vadodara",
        "state": "Gujarat",
        "address": "Near Baroda Dairy, Makarpura Road, Vadodara, Gujarat 390009",
        "lat": 22.2920,
        "lon": 73.1950,
        "phone": "+91 265 241 1120",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["Emergency Surgical Suite", "Bloat Treatment", "LSD Care", "Pathology Screening"]
    },
    {
        "id": "vet_rajkot_1",
        "name": "Rajkot District Veterinary Polyclinic",
        "city": "Rajkot",
        "state": "Gujarat",
        "address": "Race Course Road, Rajkot, Gujarat 360001",
        "lat": 22.3039,
        "lon": 70.8022,
        "phone": "+91 281 244 5820",
        "emergency": True,
        "hours": "24 Hours (Emergency)",
        "services": ["Large Animal Emergency", "Sonography", "Emergency Delivery", "Inpatient Housing"]
    },
    {
        "id": "vet_junagadh_1",
        "name": "College of Veterinary Science & Animal Hospital (JAU)",
        "city": "Junagadh",
        "state": "Gujarat",
        "address": "Motibaug, Junagadh Agricultural University, Junagadh, Gujarat 362001",
        "lat": 21.5222,
        "lon": 70.4579,
        "phone": "+91 285 267 0722",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["Veterinary Teaching Hospital", "Gir Cattle Specialist OPD", "Emergency Intensive Care", "X-Ray & Lab"]
    },
    {
        "id": "vet_bhavnagar_1",
        "name": "District Veterinary Dispensary & Animal Hospital",
        "city": "Bhavnagar",
        "state": "Gujarat",
        "address": "Kalanala, Bhavnagar, Gujarat 364001",
        "lat": 21.7645,
        "lon": 72.1519,
        "phone": "+91 278 242 8410",
        "emergency": True,
        "hours": "24 Hours (Emergency)",
        "services": ["Emergency Care", "Vaccination Drives", "Minor Surgery", "Field Veterinary Unit"]
    },
    {
        "id": "vet_bhuj_1",
        "name": "Kutch District Veterinary Polyclinic (Sarhad Dairy)",
        "city": "Bhuj",
        "state": "Gujarat",
        "address": "Near Sarpat Gate, Bhuj, Kutch, Gujarat 370001",
        "lat": 23.2420,
        "lon": 69.6669,
        "phone": "+91 2832 250 310",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["Arid Livestock Care", "Kankrej Cattle OPD", "Bovine Mobile Clinic", "Emergency Heat Stroke Unit"]
    },
    {
        "id": "vet_himatnagar_1",
        "name": "Sabar Dairy Veterinary Hospital & Emergency Clinic",
        "city": "Himatnagar",
        "state": "Gujarat",
        "address": "Sabar Dairy Campus, Subhash Road, Himatnagar, Gujarat 383001",
        "lat": 23.5979,
        "lon": 72.9698,
        "phone": "+91 2772 242 120",
        "emergency": True,
        "hours": "24 Hours (Emergency)",
        "services": ["Emergency Dystocia", "Dairy Herd Health Advisory", "LSD / FMD Isolation", "Ambulance Service"]
    },
    {
        "id": "vet_palanpur_1",
        "name": "Banas Dairy Multi-Specialty Animal Hospital",
        "city": "Palanpur",
        "state": "Gujarat",
        "address": "Post Box No 20, Banas Dairy Road, Palanpur, Gujarat 385001",
        "lat": 24.1719,
        "lon": 72.4346,
        "phone": "+91 2742 253 142",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["24x7 Emergency Surgery", "Kankrej & Mehsani Breed Clinic", "Automated Rumen Lavage", "Inpatient Sheds"]
    },
    {
        "id": "vet_navsari_1",
        "name": "College of Veterinary Science & Animal Hospital (NAU)",
        "city": "Navsari",
        "state": "Gujarat",
        "address": "Eru Char Rasta, NAU Campus, Navsari, Gujarat 396450",
        "lat": 20.9467,
        "lon": 72.9280,
        "phone": "+91 2637 283 160",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["Teaching Hospital ICU", "Digital Diagnostic Imaging", "Haemorrhagic Septicaemia Care", "Mobile Clinic"]
    },
    {
        "id": "vet_patan_1",
        "name": "Patan District Veterinary Polyclinic",
        "city": "Patan",
        "state": "Gujarat",
        "address": "Near Collector Office, Chansma Road, Patan, Gujarat 384265",
        "lat": 23.8493,
        "lon": 72.1266,
        "phone": "+91 2766 221 450",
        "emergency": False,
        "hours": "9:00 AM - 6:00 PM (Emergency on-call)",
        "services": ["Emergency Wound Care", "Routine Vaccination", "Deworming Distribution", "Artificial Insemination"]
    },
    {
        "id": "vet_godhra_1",
        "name": "Panchamrut Dairy Livestock Emergency Center",
        "city": "Godhra",
        "state": "Gujarat",
        "address": "Panchamrut Dairy Road, Godhra, Gujarat 389001",
        "lat": 22.7758,
        "lon": 73.6149,
        "phone": "+91 2672 242 815",
        "emergency": True,
        "hours": "24 Hours (Emergency)",
        "services": ["Round-the-clock Emergency", "Milk Testing & Mastitis Lab", "Surgical Support", "Ambulance Service"]
    },
    {
        "id": "vet_bharuch_1",
        "name": "Bharuch District Veterinary Hospital",
        "city": "Bharuch",
        "state": "Gujarat",
        "address": "Station Road, Bharuch, Gujarat 392001",
        "lat": 21.7051,
        "lon": 72.9959,
        "phone": "+91 2642 241 830",
        "emergency": False,
        "hours": "8:30 AM - 5:30 PM",
        "services": ["Preventive Healthcare", "Vaccination", "Minor Trauma Care", "Deworming"]
    },
    {
        "id": "vet_jamnagar_1",
        "name": "Jamnagar Veterinary Polyclinic & Hospital",
        "city": "Jamnagar",
        "state": "Gujarat",
        "address": "Bedi Road, Jamnagar, Gujarat 361002",
        "lat": 22.4707,
        "lon": 70.0577,
        "phone": "+91 288 255 1290",
        "emergency": True,
        "hours": "24 Hours (Emergency)",
        "services": ["Emergency Surgery", "FMD Quarantine", "Digital X-Ray", "Mobile Unit"]
    },

    # --- Northern Dairy & Livestock Belt ---
    {
        "id": "vet_ludhiana_1",
        "name": "Apex Livestock Care Center & GADVASU Hospital",
        "city": "Ludhiana",
        "state": "Punjab",
        "address": "Ferozepur Road, GADVASU Campus, Ludhiana, Punjab 141004",
        "lat": 30.9010,
        "lon": 75.8080,
        "phone": "+91 161 241 4000",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["Bovine Intensive Care", "Ultrasonography", "Bloat Decompression", "Emergency Surgery", "Deworming"]
    },
    {
        "id": "vet_karnal_1",
        "name": "NDRI Referral Veterinary Polyclinic",
        "city": "Karnal",
        "state": "Haryana",
        "address": "ICAR-National Dairy Research Institute, GT Road, Karnal, Haryana 132001",
        "lat": 29.6857,
        "lon": 76.9905,
        "phone": "+91 184 225 9002",
        "emergency": True,
        "hours": "24 Hours (Emergency)",
        "services": ["Dairy Cattle Specialist OPD", "Murrah Buffalo Health Hub", "Emergency C-Section", "Metabolic Profiling"]
    },
    {
        "id": "vet_hisar_1",
        "name": "LUVAS Veterinary Clinical Complex & Hospital",
        "city": "Hisar",
        "state": "Haryana",
        "address": "Lala Lajpat Rai University of Veterinary Sciences, Sirsa Road, Hisar, Haryana 125004",
        "lat": 29.1492,
        "lon": 75.7217,
        "phone": "+91 1662 256 065",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["Emergency Poisoning Care", "Buffalo Specialist OPD", "Surgical Theatre", "Diagnostic Lab"]
    },
    {
        "id": "vet_jaipur_1",
        "name": "Government Veterinary Polyclinic & Animal Hospital",
        "city": "Jaipur",
        "state": "Rajasthan",
        "address": "Panch Batti, MI Road, Jaipur, Rajasthan 302001",
        "lat": 26.9157,
        "lon": 75.8118,
        "phone": "+91 141 237 2000",
        "emergency": True,
        "hours": "24 Hours (Emergency) / 9 AM - 5 PM (OPD)",
        "services": ["Emergency Surgery", "FMD Vaccination", "Digital X-Ray", "Mobile Unit", "Large Animal Inpatient"]
    },
    {
        "id": "vet_bikaner_1",
        "name": "RAJUVAS Veterinary Clinical Complex",
        "city": "Bikaner",
        "state": "Rajasthan",
        "address": "Bijey Bhavan Palace Complex, Bikaner, Rajasthan 334001",
        "lat": 28.0229,
        "lon": 73.3119,
        "phone": "+91 151 254 0021",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["Desert Livestock Care", "Rathi Cattle Specialist Unit", "Emergency Colic Relief", "Inpatient Housing"]
    },
    {
        "id": "vet_udaipur_1",
        "name": "District Veterinary Polyclinic & Hospital",
        "city": "Udaipur",
        "state": "Rajasthan",
        "address": "Chetak Circle, Udaipur, Rajasthan 313001",
        "lat": 24.5854,
        "lon": 73.7125,
        "phone": "+91 294 242 1890",
        "emergency": True,
        "hours": "24 Hours (Emergency)",
        "services": ["Emergency C-Section", "Vaccine Distribution", "Bloat Treatment", "Livestock Ambulance"]
    },
    {
        "id": "vet_delhi_1",
        "name": "Central Veterinary Hospital & Animal Care",
        "city": "New Delhi",
        "state": "Delhi",
        "address": "Moti Bagh, Ring Road, New Delhi 110021",
        "lat": 28.5880,
        "lon": 77.1680,
        "phone": "+91 11 2467 2150",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["Emergency Trauma", "Vaccination", "Pathology Lab", "Mobile Veterinary Ambulance"]
    },
    {
        "id": "vet_lucknow_1",
        "name": "State Veterinary Hospital & Large Animal Clinic",
        "city": "Lucknow",
        "state": "Uttar Pradesh",
        "address": "Gokaran Nath Road, Badshahnagar, Lucknow, UP 226006",
        "lat": 26.8683,
        "lon": 80.9575,
        "phone": "+91 522 232 4410",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["Emergency Delivery", "Wound Care", "Bloat Treatment", "Routine OPD", "Mobile Hospital"]
    },
    {
        "id": "vet_bareilly_1",
        "name": "ICAR-IVRI Referral Veterinary Polyclinic",
        "city": "Bareilly",
        "state": "Uttar Pradesh",
        "address": "Indian Veterinary Research Institute Campus, Izatnagar, Bareilly, UP 243122",
        "lat": 28.3970,
        "lon": 79.4320,
        "phone": "+91 581 258 6328",
        "emergency": True,
        "hours": "24 Hours (Emergency Referral)",
        "services": ["National Referral Center", "Advanced Bovine Surgery", "Virology & Bacterial Disease Lab", "ICU"]
    },
    {
        "id": "vet_mathura_1",
        "name": "DUVASU Veterinary Clinical Complex & Hospital",
        "city": "Mathura",
        "state": "Uttar Pradesh",
        "address": "UP Pandit Deen Dayal Upadhyaya Pashu Chikitsa Vigyan Vishwavidyalaya, Mathura, UP 281001",
        "lat": 27.4924,
        "lon": 77.6737,
        "phone": "+91 565 247 1178",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["Large Animal Surgery", "Reproductive Disorders Clinic", "Inpatient Sheds", "Emergency Unit"]
    },
    {
        "id": "vet_varanasi_1",
        "name": "District Veterinary Polyclinic & Hospital",
        "city": "Varanasi",
        "state": "Uttar Pradesh",
        "address": "Orderly Bazar, Near District Jail, Varanasi, UP 221002",
        "lat": 25.3356,
        "lon": 82.9890,
        "phone": "+91 542 250 1420",
        "emergency": True,
        "hours": "24 Hours (Emergency)",
        "services": ["Emergency Dystocia", "Haemorrhagic Septicaemia Vaccine", "OPD Care", "Mobile Unit"]
    },
    {
        "id": "vet_dehradun_1",
        "name": "Dehradun Livestock Polyclinic & Hospital",
        "city": "Dehradun",
        "state": "Uttarakhand",
        "address": "Pashudhan Bhawan, Mothrowala Road, Dehradun, Uttarakhand 248001",
        "lat": 30.2912,
        "lon": 78.0345,
        "phone": "+91 135 267 1205",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["Mountain Livestock Care", "Sheep & Goat Care", "Emergency Trauma", "Vaccination Supplies"]
    },
    {
        "id": "vet_shimla_1",
        "name": "State Central Veterinary Hospital",
        "city": "Shimla",
        "state": "Himachal Pradesh",
        "address": "Tutikandi, Shimla, Himachal Pradesh 171004",
        "lat": 31.0980,
        "lon": 77.1520,
        "phone": "+91 177 280 4390",
        "emergency": True,
        "hours": "24 Hours (Emergency)",
        "services": ["High-altitude Animal Care", "Emergency Poisoning Unit", "Vaccination Depot", "Ambulance"]
    },

    # --- Western & Central Livestock Belt ---
    {
        "id": "vet_pune_1",
        "name": "Pune District Veterinary Dispensary & Hospital",
        "city": "Pune",
        "state": "Maharashtra",
        "address": "Aundh Road, Ganeshkhind, Pune, Maharashtra 411007",
        "lat": 18.5529,
        "lon": 73.8188,
        "phone": "+91 20 2565 8920",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["Large Animal Emergency", "Mastitis Treatment & Milk Testing", "Surgery", "Livestock Ambulance"]
    },
    {
        "id": "vet_mumbai_1",
        "name": "Bombay Veterinary College Hospital (MAFSU)",
        "city": "Mumbai",
        "state": "Maharashtra",
        "address": "Parel, Dr. SS Rao Road, Mumbai, Maharashtra 400012",
        "lat": 19.0020,
        "lon": 72.8420,
        "phone": "+91 22 2413 1180",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["Critical Care ICU", "Advanced Bovine Surgery", "Pathology Diagnostics", "Blood Transfusion"]
    },
    {
        "id": "vet_nagpur_1",
        "name": "Nagpur Veterinary College Clinical Complex",
        "city": "Nagpur",
        "state": "Maharashtra",
        "address": "Seminary Hills, Nagpur, Maharashtra 440006",
        "lat": 21.1620,
        "lon": 79.0620,
        "phone": "+91 712 251 1402",
        "emergency": True,
        "hours": "24 Hours (Emergency)",
        "services": ["Vidarbha Livestock Emergency Hub", "C-Section", "Bloat Care", "Mobile Vet Van"]
    },
    {
        "id": "vet_indore_1",
        "name": "Rural Animal Husbandry Veterinary Dispensary & Polyclinic",
        "city": "Indore",
        "state": "Madhya Pradesh",
        "address": "AB Road, Near Agricultural College, Indore, MP 452001",
        "lat": 22.7196,
        "lon": 75.8577,
        "phone": "+91 731 249 2200",
        "emergency": False,
        "hours": "8:00 AM - 6:00 PM (Emergency on-call)",
        "services": ["General Health Check", "Preventive Vaccination", "Deworming", "Minor Wound Dressing"]
    },
    {
        "id": "vet_bhopal_1",
        "name": "State Central Veterinary Hospital",
        "city": "Bhopal",
        "state": "Madhya Pradesh",
        "address": "Jahangirabad, Bhopal, Madhya Pradesh 462008",
        "lat": 23.2420,
        "lon": 77.4126,
        "phone": "+91 755 255 1340",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["Emergency Surgery", "FMD Diagnostic Lab", "Livestock Ambulance 1962", "Inpatient Housing"]
    },

    # --- Southern Livestock Belt ---
    {
        "id": "vet_bengaluru_1",
        "name": "Central Livestock Hospital & Veterinary College (KVAFSU)",
        "city": "Bengaluru",
        "state": "Karnataka",
        "address": "Hebbal, Bellary Road, Bengaluru, Karnataka 560024",
        "lat": 13.0358,
        "lon": 77.5870,
        "phone": "+91 80 2341 1483",
        "emergency": True,
        "hours": "24 Hours (Emergency)",
        "services": ["Emergency ICU", "Rumen Impaction Treatment", "Advanced Diagnostics", "Pathology Lab", "Ambulance"]
    },
    {
        "id": "vet_mysuru_1",
        "name": "District Veterinary Hospital",
        "city": "Mysuru",
        "state": "Karnataka",
        "address": "Dhanvantri Road, Mysuru, Karnataka 570001",
        "lat": 12.3118,
        "lon": 76.6529,
        "phone": "+91 821 242 0890",
        "emergency": True,
        "hours": "24 Hours (Emergency)",
        "services": ["Emergency Obstetrics", "Routine OPD", "Preventive Vaccines", "Mobile Unit"]
    },
    {
        "id": "vet_hyderabad_1",
        "name": "PVNRTVU Veterinary Clinical Complex",
        "city": "Hyderabad",
        "state": "Telangana",
        "address": "Rajendranagar, Hyderabad, Telangana 500030",
        "lat": 17.3200,
        "lon": 78.4050,
        "phone": "+91 40 2400 2150",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["Advanced Surgical Wing", "Deora & Ongole Breed Clinic", "Emergency Trauma", "ICU Sheds"]
    },
    {
        "id": "vet_vijayawada_1",
        "name": "Government Veterinary Polyclinic",
        "city": "Vijayawada",
        "state": "Andhra Pradesh",
        "address": "Labbipet, MG Road, Vijayawada, Andhra Pradesh 520010",
        "lat": 16.5062,
        "lon": 80.6480,
        "phone": "+91 866 247 1820",
        "emergency": True,
        "hours": "24 Hours (Emergency)",
        "services": ["Large Animal Emergency", "Bloat Treatment", "LSD Care", "Vaccination Drives"]
    },
    {
        "id": "vet_chennai_1",
        "name": "Madras Veterinary College Hospital (Large Animal Unit)",
        "city": "Chennai",
        "state": "Tamil Nadu",
        "address": "Vepery High Road, Vepery, Chennai, Tamil Nadu 600007",
        "lat": 13.0850,
        "lon": 80.2642,
        "phone": "+91 44 2530 4000",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["24x7 Critical Care", "Endoscopy", "Emergency Soft Tissue Surgery", "Blood Transfusion", "Inpatient Sheds"]
    },
    {
        "id": "vet_coimbatore_1",
        "name": "District Veterinary Hospital & Clinical Center",
        "city": "Coimbatore",
        "state": "Tamil Nadu",
        "address": "Near Railway Station, Goods Shed Road, Coimbatore, Tamil Nadu 641001",
        "lat": 11.0168,
        "lon": 76.9558,
        "phone": "+91 422 230 0812",
        "emergency": True,
        "hours": "24 Hours (Emergency)",
        "services": ["Emergency Surgical Care", "Bovine Inpatient Facility", "Mobile Veterinary Ambulance", "FMD Ward"]
    },

    # --- Eastern & North-Eastern Belt ---
    {
        "id": "vet_patna_1",
        "name": "Bihar Veterinary College Hospital (BASU)",
        "city": "Patna",
        "state": "Bihar",
        "address": "Bihar Animal Sciences University Campus, Patna, Bihar 800014",
        "lat": 25.6120,
        "lon": 85.0930,
        "phone": "+91 612 222 7715",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["Bovine Emergency Care", "Haemorrhagic Septicaemia Unit", "Surgical Complex", "Free Vaccine Depot"]
    },
    {
        "id": "vet_ranchi_1",
        "name": "Birsa Agricultural University Veterinary Clinical Complex",
        "city": "Ranchi",
        "state": "Jharkhand",
        "address": "Kanke, Ranchi, Jharkhand 834006",
        "lat": 23.4350,
        "lon": 85.3210,
        "phone": "+91 651 245 0830",
        "emergency": True,
        "hours": "24 Hours (Emergency)",
        "services": ["Tribal Livestock Health Unit", "Emergency Delivery", "Deworming Campaigns", "Mobile Clinic"]
    },
    {
        "id": "vet_kolkata_1",
        "name": "WBUAFS Veterinary Clinical Complex",
        "city": "Kolkata",
        "state": "West Bengal",
        "address": "68 Kshudiram Bose Sarani, Belgachia, Kolkata, West Bengal 700037",
        "lat": 22.6050,
        "lon": 88.3810,
        "phone": "+91 33 2556 5021",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["Bovine Intensive Care", "Black Quarter Isolation", "Advanced Surgery", "Diagnostic Pathology"]
    },
    {
        "id": "vet_guwahati_1",
        "name": "College of Veterinary Science Clinical Complex (AAU)",
        "city": "Guwahati",
        "state": "Assam",
        "address": "Khanapara, GS Road, Guwahati, Assam 781022",
        "lat": 26.1158,
        "lon": 91.8175,
        "phone": "+91 361 233 4990",
        "emergency": True,
        "hours": "24 Hours",
        "services": ["Northeast Livestock Referral Hospital", "Flood Trauma Care", "Emergency Surgery", "Mobile Van"]
    }
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance in kilometers between two coordinates."""
    r = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(r * c, 1)


def search_vets(query: Optional[str] = None,
                user_lat: Optional[float] = None,
                user_lon: Optional[float] = None,
                emergency_only: bool = False) -> Dict[str, Any]:
    """Filter and sort veterinary clinics."""
    results = []
    q = (query or "").strip().lower()

    for vet in VET_CLINICS:
        if emergency_only and not vet.get("emergency"):
            continue

        if q:
            searchable = f"{vet['name']} {vet['city']} {vet['state']} {vet['address']} {' '.join(vet['services'])}".lower()
            if q not in searchable:
                continue

        vet_copy = dict(vet)
        if user_lat is not None and user_lon is not None:
            vet_copy["distance_km"] = haversine_km(user_lat, user_lon, vet["lat"], vet["lon"])
        else:
            vet_copy["distance_km"] = None

        results.append(vet_copy)

    # Sort: by distance if coordinates provided, else emergency first
    if user_lat is not None and user_lon is not None:
        results.sort(key=lambda v: (v["distance_km"] if v["distance_km"] is not None else 999999))
    else:
        results.sort(key=lambda v: (not v.get("emergency", False), v["city"]))

    return {
        "hotlines": EMERGENCY_HOTLINES,
        "count": len(results),
        "clinics": results,
        "user_coords": {"lat": user_lat, "lon": user_lon} if (user_lat and user_lon) else None
    }
