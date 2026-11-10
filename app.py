import hashlib
import html
import re

import streamlit as st
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification


# ============================================================
# PLANTSCAN - KONFIGURATION
# ============================================================

MODEL_ID = "domai-tb/OpenPlants-Identification-ViT-Base-Patch16-224"

st.set_page_config(
    page_title="PlantScan | Dein Pflanzenhelfer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "view": None,
    "plant_results": None,
    "image_hash": None,
    "analysis_error": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# DESIGN & CSS (Mit Rand-Blättern im Entwurfs-Look)
# ============================================================

st.markdown("""
<style>
    .stApp {
        background: 
            url('https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?q=80&w=400&auto=format&fit=crop') left center / 22% 100% no-repeat fixed,
            url('https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?q=80&w=400&auto=format&fit=crop') right center / 22% 100% no-repeat fixed,
            linear-gradient(135deg, #f2f7ec 0%, #e1edd8 100%);
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 5px;
    }

    .main-title-box {
        display: flex;
        align-items: center;
        gap: 15px;
    }

    .main-title {
        color: #174b2e;
        font-size: 44px;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin: 0;
        line-height: 1.1;
    }

    .subtitle {
        color: #5c7960;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .hero {
        background: linear-gradient(rgba(23, 75, 46, 0.75), rgba(23, 75, 46, 0.75)), 
                    url('https://images.unsplash.com/photo-1530595467537-0b5996c41f2d?q=80&w=1200&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        padding: 36px 40px;
        border-radius: 24px;
        color: white;
        margin: 15px 0 25px 0;
        box-shadow: 0 10px 30px rgba(23,75,46,0.15);
    }

    .hero h2 {
        color: white !important;
        font-size: 28px;
        margin-top: 0;
        margin-bottom: 10px;
        font-weight: 700;
    }

    .hero p {
        color: #e4f2e6;
        font-size: 15px;
        line-height: 1.6;
        margin: 0;
        max-width: 700px;
    }

    .card-container {
        background: #fbfdf9;
        border: 1px solid #dcebd8;
        border-radius: 24px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(30, 80, 40, 0.05);
    }

    .card-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 8px;
    }

    .card-title {
        color: #174b2e;
        font-size: 22px;
        font-weight: 700;
        margin: 0;
    }

    .card-desc {
        color: #617566;
        font-size: 15px;
        margin-bottom: 20px;
    }

    .step-row {
        display: flex;
        align-items: center;
        gap: 14px;
        margin: 12px 0;
        color: #2c4c33;
        font-size: 15px;
        font-weight: 500;
    }

    .step-badge {
        background-color: #287344;
        color: white;
        width: 26px;
        height: 26px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        font-weight: 700;
        flex-shrink: 0;
    }

    div[data-testid="stFileUploader"] {
        background: #f4f8f1;
        border: 2px dashed #b8d4b0;
        border-radius: 18px;
        padding: 10px;
    }
    div[data-testid="stFileUploader"] section {
        padding: 15px;
    }

    .section-heading {
        color: #174b2e;
        font-size: 24px;
        font-weight: 800;
        margin: 25px 0 15px 0;
    }

    .feature-card {
        background: white;
        padding: 24px;
        border-radius: 18px;
        border: 1px solid #dcebd8;
        min-height: 175px;
        box-shadow: 0 4px 12px rgba(30,80,40,0.04);
        margin-bottom: 10px;
    }

    .feature-card h3 {
        color: #205b36;
        font-size: 20px;
        margin-top: 0;
    }

    .feature-card p {
        color: #617566;
        font-size: 14px;
        line-height: 1.5;
        margin-bottom: 0;
    }

    .result-name {
        color: #174b2e;
        font-size: 26px;
        font-weight: 800;
    }

    .muted {
        color: #718276;
        font-size: 13px;
    }

    div.stButton > button {
        background: #287344;
        color: white;
        border: none;
        border-radius: 12px;
        min-height: 45px;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background: #174b2e;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(22,75,46,0.2);
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #dcebd8;
        border-radius: 14px;
        padding: 14px;
    }

    hr {
        border-color: #dcebd8;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODELL LADEN
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


# ============================================================
# PFLANZEN-DATENBANK & HILFSFUNKTIONEN
# ============================================================

PLANT_INFO = {
    "monstera deliciosa": {
        "name": "Monstera",
        "description": "Eine tropische Zimmerpflanze mit großen, charakteristisch eingeschnittenen Blättern.",
        "care": "Helles, indirektes Licht. Mäßig gießen und Staunässe vermeiden.",
    },
    "sansevieria trifasciata": {
        "name": "Bogenhanf",
        "description": "Eine robuste Zimmerpflanze mit langen, aufrecht wachsenden Blättern.",
        "care": "Heller bis halbschattiger Standort. Nur sparsam gießen.",
    },
    "chlorophytum comosum": {
        "name": "Grünlilie",
        "description": "Eine beliebte Zimmerpflanze mit langen, schmalen Blättern.",
        "care": "Heller Standort ohne starke direkte Mittagssonne. Regelmäßig, aber mäßig gießen.",
    },
    "aloe vera": {
        "name": "Aloe vera",
        "description": "Eine sukkulente Pflanze mit dicken, fleischigen Blättern.",
        "care": "Viel Licht. Sparsam gießen und die Erde zwischen den Wassergaben trocknen lassen.",
    },
    "zamioculcas zamiifolia": {
        "name": "Glücksfeder",
        "description": "Eine robuste Zimmerpflanze mit glänzenden, kräftigen Blättern.",
        "care": "Heller bis halbschattiger Standort. Wenig Wasser und keine Staunässe.",
    },
    "epipremnum aureum": {
        "name": "Efeutute",
        "description": "Eine Kletterpflanze mit herzförmigen Blättern.",
        "care": "Heller Standort ohne starke direkte Sonne. Mäßig gießen.",
    },
    "ficus elastica": {
        "name": "Gummibaum",
        "description": "Eine Zimmerpflanze mit großen, dicken und glänzenden Blättern.",
        "care": "Heller Standort und mäßige Wassergaben.",
    },
    "ficus lyrata": {
        "name": "Geigenfeige",
        "description": "Eine Zimmerpflanze mit großen, geigenförmigen Blättern.",
        "care": "Heller Standort ohne starke direkte Mittagssonne.",
    },
    "spathiphyllum wallisii": {
        "name": "Einblatt",
        "description": "Eine Zimmerpflanze mit dunkelgrünen Blättern und weißen Blüten.",
        "care": "Heller bis halbschattiger Standort. Erde leicht feucht halten.",
    },
    "dracaena marginata": {
        "name": "Drachenbaum",
        "description": "Eine Zimmerpflanze mit schmalen, langen Blättern.",
        "care": "Heller Standort und mäßig gießen.",
    },
    "crassula ovata": {
        "name": "Geldbaum",
        "description": "Eine Sukkulente mit dicken, ovalen Blättern.",
        "care": "Viel Licht und sparsam gießen.",
    },
    "yucca elephantipes": {
        "name": "Yucca-Palme",
        "description": "Eine Zimmerpflanze mit holzigem Stamm und langen, spitzen Blättern.",
        "care": "Heller Standort und mäßige Wassergaben.",
    },
}

TOXICITY_INFO = {
    "monstera deliciosa": {
        "details": "Die Monstera enthält Calciumoxalat-Kristalle. Pflanzenteile können bei Aufnahme Reizungen verursachen.",
        "audience": "Menschen und Haustiere",
    },
    "sansevieria trifasciata": {
        "details": "Der Bogenhanf enthält Saponine und kann bei Aufnahme Beschwerden verursachen.",
        "audience": "Insbesondere Haustiere",
    },
    "aloe vera": {
        "details": "Bestandteile der Aloe vera können bei Aufnahme Beschwerden verursachen.",
        "audience": "Insbesondere Haustiere",
    },
    "zamioculcas zamiifolia": {
        "details": "Die Glücksfeder enthält Calciumoxalat-Kristalle, die Schleimhäute reizen können.",
        "audience": "Menschen und Haustiere",
    },
    "epipremnum aureum": {
        "details": "Die Efeutute enthält Calciumoxalat-Kristalle, die bei Aufnahme Reizungen verursachen können.",
        "audience": "Menschen und Haustiere",
    },
    "ficus elastica": {
        "details": "Der Gummibaum enthält einen reizenden Milchsaft.",
        "audience": "Menschen und Haustiere",
    },
    "spathiphyllum wallisii": {
        "details": "Das Einblatt enthält Calciumoxalat-Kristalle, die bei Kontakt oder Aufnahme reizen können.",
        "audience": "Menschen und Haustiere",
    },
}

PLANT_ALIASES = {
    "dracaena trifasciata": "sansevieria trifasciata",
    "sansevieria": "sansevieria trifasciata",
    "bogenhanf": "sansevieria trifasciata",
    "gruenlilie": "chlorophytum comosum",
    "grünlilie": "chlorophytum comosum",
    "gluecksfeder": "zamioculcas zamiifolia",
    "glücksfeder": "zamioculcas zamiifolia",
    "efeutute": "epipremnum aureum",
    "gummibaum": "ficus elastica",
    "geigenfeige": "ficus lyrata",
    "einblatt": "spathiphyllum wallisii",
    "drachenbaum": "dracaena marginata",
    "geldbaum": "crassula ovata",
    "yucca palme": "yucca elephantipes",
    "yucca": "yucca elephantipes",
}

def normalize_name(name):
    name = str(name).lower().strip()
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"[^a-z0-9äöüß ]", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()

def find_database_entry(label, database):
    normalized = normalize_name(label)
    if normalized in database:
        return database[normalized]
    for alias in sorted(PLANT_ALIASES, key=len, reverse=True):
        if normalized == alias or normalized.startswith(alias + " "):
            canonical_name = PLANT_ALIASES[alias]
            if canonical_name in database:
                return database[canonical_name]
    for key in sorted(database, key=len, reverse=True):
        if normalized == key or normalized.startswith(key + " "):
            return database[key]
    return None

def get_plant_information(label):
    result = find_database_entry(label, PLANT_INFO)
    if result:
        return result
    return {
        "name": label,
        "description": "Das Modell hat diese Pflanzenart erkannt. In unserer Informationsdatenbank liegen dazu noch keine genauen Zusatzinfos vor.",
        "care": "Bitte informiere dich anhand des wissenschaftlichen Pflanzennamens über die individuellen Pflegeansprüche.",
    }

def get_toxicity_information(label):
    result =
