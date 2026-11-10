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
# DESIGN & CSS (Angepasst an deinen Entwurf)
# ============================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f2f7ec 0%, #e1edd8 100%);
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Header / Titel-Bereich */
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

    /* Hero Banner mit Bild-Hintergrund */
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

    /* Container Karten im Stil des Entwurfs */
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

    /* Nummerierte Schritte für "So funktioniert's" */
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

    /* Streamlit File Uploader optisch anpassen */
    div[data-testid="stFileUploader"] {
        background: #f4f8f1;
        border: 2px dashed #b8d4b0;
        border-radius: 18px;
        padding: 10px;
    }
    div[data-testid="stFileUploader"] section {
        padding: 15px;
    }

    /* Allgemeine UI Elemente */
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
        transition: all 0.2s ease;
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
# PFLANZEN-DATENBANK & HILFSFUNKTIONEN (Unverändert stabil)
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
        "description": "Das Modell hat diese Pflanzenart erkannt. Keine genauen Zusatzinfos hinterlegt.",
        "care": "Bitte informiere dich über den wissenschaftlichen Pflanzennamen.",
    }

def get_toxicity_information(label):
    result = find_database_entry(label, TOXICITY_INFO)
    if result:
        return {"known": True, "details": result["details"], "audience": result["audience"]}
    return {"known": False, "details": "Nicht in der Giftigkeitsdatenbank enthalten.", "audience": "Nicht überprüft"}

def classify_image(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)[0]
    count = min(5, probabilities.shape[0])
    top_results = torch.topk(probabilities, k=count)
    results = []
    for score, index in zip(top_results.values, top_results.indices):
        label = model.config.id2label[index.item()]
        results.append({"name": str(label), "score": float(score.item())})
    return results


# ============================================================
# AUSWAHL-POPUP
# ============================================================

@st.dialog("🌿 Was möchtest du entdecken?", width="large")
def choose_mode():
    st.write("Dein Pflanzenfoto ist bereit! Wähle einen Bereich aus.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🌱 Pflanzenpflege</h3>
            <p>Entdecke Pflegehinweise, Standort, Wasserbedarf und mögliche Risiken.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Pflege öffnen", key="care_choice", use_container_width=True):
            st.session_state.view = "care"
            st.rerun()
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Pflanzenerkennung</h3>
            <p>Entdecke den vermuteten Namen, die KI-Ergebnisse und den Giftigkeitscheck.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Erkennung öffnen", key="recognition_choice", use_container_width=True):
            st.session_state.view = "recognition"
            st.rerun()


# ============================================================
# HEADER BEREICH (EXAKT WIE IM ENTWURF)
# ============================================================

st.markdown("""
<div class="header-container">
    <div class="main-title-box">
        <span style="font-size: 42px;">🌿</span>
        <h1 class="main-title">PlantScan</h1>
    </div>
</div>
<div class="subtitle">Dein intelligenter Begleiter für gesunde Pflanzen.</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h2>Entdecke die Welt deiner Pflanzen.</h2>
    <p>Lade ein Foto hoch und entdecke mit künstlicher Intelligenz, welche Pflanze vor dir steht und was sie benötigt.</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# UPLOAD BEREICH (IM DESIGN DER KARTE)
# ============================================================

st.markdown("""
<div class="card-container">
    <div class="card-header">
        <span style="font-size: 26px;">📷</span>
        <h3 class="card-title">Deine Pflanze entdecken</h3>
    </div>
    <div class="card-desc">Lade ein Foto deiner Pflanze hoch.</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload",
    type=["jpg", "jpeg", "png", "webp"],
    key="plant_upload",
    label_visibility="collapsed"
)

st.markdown("</div>", unsafe_allow_html=True)


if uploaded_file is None:
    # So funktioniert PlantScan Karte
    st.markdown("""
    <div class="card-container">
        <div class="card-header">
            <span style="font-size: 26px;">🌱</span>
            <h3 class="card-title">So funktioniert PlantScan</h3>
        </div>
        <div style="margin-top: 15px;">
            <div class="step-row">
                <div class="step-badge">1</div>
                <div>Lade ein Foto deiner Pflanze hoch.</div>
            </div>
            <div class="step-row">
                <div class="step-badge">2</div>
                <div>Wähle zwischen Pflege und Erkennung.</div>
            </div>
            <div class="step-row">
                <div class="step-badge">3</div>
                <div>Entdecke Informationen und mögliche Risiken.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    try:
        image_bytes = uploaded_file.getvalue()
        current_hash = hashlib.sha256(image_bytes).hexdigest()

        if current_hash != st.session_state.image_hash:
            st.session_state.image_hash = current_hash
            st.session_state.view = None
            st.session_state.plant_results = None
            st.session_state.analysis_error = None

        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        st.image(image, caption="Dein hochgeladenes Pflanzenfoto", width=400)

        if st.session_state.view is None:
            choose_mode()

        if st.session_state.view is not None:
            if st.session_state.plant_results is None:
                try:
                    with st.spinner("🌿 Die KI untersucht deine Pflanze..."):
                        processor, model = load_model()
                        results = classify_image(image, processor, model)
                        st.session_state.plant_results = results
                        st.session_state.analysis_error = None
                except Exception as error:
                    st.session_state.analysis_error = str(error)

            if st.session_state.analysis_error:
                st.error("Die Pflanzenanalyse konnte nicht durchgeführt werden.")
                st.code(st.session_state.analysis_error)

            elif st.session_state.plant_results:
                results = st.session_state.plant_results
                best_result = results[0]
                label = best_result["name"]
                confidence = best_result["score"] * 100

                plant = get_plant_information(label)
                toxicity = get_toxicity_information(label)

                st.divider()

                # PFLEGE VIEW
                if st.session_state.view == "care":
                    st.markdown('<div class="section-heading">🌱 Pflanzenpflege</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="card-container">
                        <p class="muted">Erkannte Pflanzenart</p>
                        <div class="result-name">{html.escape(plant["name"])}</div>
                        <p>{html.escape(plant["description"])}</p>
                        <hr>
                        <h3>💧 Pflegehinweise</h3>
                        <p>{html.escape(plant["care"])}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.subheader("⚠️ Sicherheit")
                    if toxicity["known"]:
                        st.warning("Mögliche Giftigkeit: " + toxicity["details"])
                    else:
                        st.info("Giftigkeit nicht überprüft: " + toxicity["details"])
                    st.caption("Betroffene Gruppe laut Datenbank: " + toxicity["audience"])

                # ERKENNUNG VIEW
                elif st.session_state.view == "recognition":
                    st.markdown('<div class="section-heading">🔍 KI-Pflanzenerkennung</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="card-container">
                        <p class="muted">Wahrscheinlichste Pflanzenart</p>
                        <div class="result-name">{html.escape(plant["name"])}</div>
                        <p>{html.escape(label)}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Modell-Konfidenz", f"{confidence:.2f} %")
                    with col2:
                        st.metric("Weitere Ergebnisse", len(results))

                    if confidence < 50:
                        st.warning("Die Erkennung ist unsicher. Bitte überprüfe die Bestimmung.")
                    else:
                        st.info("Das Modell zeigt eine hohe Konfidenz.")

                    st.subheader("⚠️ Giftigkeitscheck")
                    if toxicity["known"]:
                        st.error("Für diese Pflanzenart sind mögliche Risiken bekannt.")
                    else:
                        st.warning("Giftigkeit nicht überprüft – unbekannt bedeutet nicht ungefährlich.")
                    st.write(toxicity["details"])
                    st.caption("Betroffene Gruppe laut Datenbank: " + toxicity["audience"])

                    st.subheader("📊 Weitere KI-Ergebnisse")
                    for result in results:
                        percentage = result["score"] * 100
                        st.write(f"**{result['name']}** – {percentage:.2f} %")
                        st.progress(float(result["score"]))

                st.divider()
                if st.button("↩ Zurück zur Auswahl", use_container_width=True):
                    st.session_state.view = None
                    st.rerun()

    except Exception as error:
        st.error("Das Bild konnte nicht verarbeitet werden.")
        st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown("""
<div style="text-align:center; padding:20px; color:#55745d;">
    <h3 style="color:#174b2e; margin-bottom:5px;">🌿 PlantScan</h3>
    Dein digitaler Pflanzenhelfer<br>
    <small>Ein KI-Schulprojekt</small>
</div>
""", unsafe_allow_html=True)
