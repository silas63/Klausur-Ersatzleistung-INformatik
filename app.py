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
    "view": "care",
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
# MASSIV ERWEITERTE PFLANZEN-DATENBANK
# ============================================================

PLANT_INFO = {
    "monstera deliciosa": {
        "name": "Monstera (Fensterblatt)",
        "description": "Eine tropische Zimmerpflanze mit großen, charakteristisch eingeschnittenen Blättern.",
        "care": "Helles, indirektes Licht. Mäßig gießen und Staunässe unbedingt vermeiden.",
    },
    "sansevieria trifasciata": {
        "name": "Bogenhanf (Sansevieria)",
        "description": "Eine extrem robuste Zimmerpflanze mit langen, aufrecht wachsenden, sukkulenten Blättern.",
        "care": "Heller bis halbschattiger Standort. Nur sparsam gießen, verträgt Trockenheit hervorragend.",
    },
    "chlorophytum comosum": {
        "name": "Grünlilie",
        "description": "Eine sehr beliebte, pflegeleichte Zimmerpflanze mit langen, schmalen, oft weiß-grün gestreiften Blättern.",
        "care": "Heller Standort ohne starke direkte Mittagssonne. Regelmäßig, aber mäßig gießen.",
    },
    "aloe vera": {
        "name": "Aloe vera",
        "description": "Eine sukkulente Heilpflanze mit dicken, fleischigen Blättern, die heilenden Saft enthalten.",
        "care": "Sehr viel Licht und Sonne. Sparsam gießen, die Erde vor dem nächsten Gießen komplett austrocknen lassen.",
    },
    "zamioculcas zamiifolia": {
        "name": "Glücksfeder",
        "description": "Eine unkomplizierte Zimmerpflanze mit glänzenden, kräftigen Blättern und dickem Rhizom.",
        "care": "Kommt auch mit schattigen Ecken klar. Wenig Wasser und keinesfalls Staunässe.",
    },
    "epipremnum aureum": {
        "name": "Efeutute",
        "description": "Eine wüchsige Kletter- und Ampel-Pflanze mit herzförmigen, oft panaschierten Blättern.",
        "care": "Heller bis halbschattiger Standort ohne direkte Sonne. Mäßig feucht halten.",
    },
    "ficus elastica": {
        "name": "Gummibaum",
        "description": "Klassische Zimmerpflanze mit großen, lederartigen, glänzenden Blättern.",
        "care": "Heller Standort, mäßige Wassergaben mit kalkarmem Wasser bei Zimmertemperatur.",
    },
    "ficus lyrata": {
        "name": "Geigenfeige",
        "description": "Imposante Zimmerpflanze mit großen, geigenförmigen und geäderten Blättern.",
        "care": "Sehr hell, keine direkte Sonne, zugluftgeschützt. Gleichmäßig leicht feucht halten.",
    },
    "spathiphyllum wallisii": {
        "name": "Einblatt",
        "description": "Zimmerpflanze mit dunkelgrünem Laub und charakteristischen weißen Blütenkolben.",
        "care": "Halbschattig bis hell, hohe Luftfeuchtigkeit wird geliebt. Die Erde immer leicht feucht halten.",
    },
    "dracaena marginata": {
        "name": "Drachenbaum",
        "description": "Palmenähnliche Zimmerpflanze mit schlankem Stamm und schmalen, rotgeränderten Blättern.",
        "care": "Heller bis halbschattiger Standort. Mäßig gießen, Staunässe führt zu Wurzelfäule.",
    },
    "crassula ovata": {
        "name": "Geldbaum (Kirchbäumchen)",
        "description": "Beliebte Sukkulente mit dicken, ovalen Blättern und baumartigem Wuchs.",
        "care": "Viel Sonne, durchlässige Erde. Sparsam gießen.",
    },
    "yucca elephantipes": {
        "name": "Yucca-Palme (Palmlilie)",
        "description": "Holziger Stamm mit schopfartig angeordneten, langen und spitzen Blättern.",
        "care": "Heller, sonniger Standort. Mäßig gießen, im Winter noch weniger.",
    },
    "quercus robur": {
        "name": "Stiel-Eiche",
        "description": "Einer der majestätischsten heimischen Laubbäume mit gelappten Blättern und Eicheln.",
        "care": "Freistehend im Garten, sonniger Standort, tiefgründiger Boden.",
    },
    "fagus sylvatica": {
        "name": "Rot-Buche",
        "description": "Der häufigste heimische Laubbaum in deutschen Wäldern mit glatter Rinde und welligen Blättern.",
        "care": "Wächst als Wald- und Parkbaum, bevorzugt nährstoffreiche, frische Böden.",
    },
    "betula pendula": {
        "name": "Sand-Birke",
        "description": "Leicht erkennbarer Baum mit auffälliger weißer Rinde und zarten, hängenden Zweigen.",
        "care": "Sehr anspruchslos, liebt sonnige Standorte.",
    },
    "acer pseudoplatanus": {
        "name": "Berg-Ahorn",
        "description": "Großer, heimischer Laubbaum mit typisch fünfzipfeligen Ahornblättern und geflügelten Nüssen.",
        "care": "Robuster Baum für große Gärten und Parks.",
    },
    "pinus sylvestris": {
        "name": "Wald-Kiefer (Föhre)",
        "description": "Nadelbaum mit rötlicher Rinde im oberen Stammbereich und blaugrünen Nadeln.",
        "care": "Sonniger Standort, gedeiht auch auf sandigen, mageren Böden.",
    },
    "picea abies": {
        "name": "Gemeine Fichte (Rotfichte)",
        "description": "Häufigster Nadelbaum in deutschen Wirtschaftswäldern, oft als Weihnachtsbaum genutzt.",
        "care": "Bevorzugt kältere, feuchtere Lagen und saure Böden.",
    },
    "taraxacum officinale": {
        "name": "Wiesen-Löwenzahn",
        "description": "Bekannte Wildpflanze mit leuchtend gelben Blütenkörbchen und gezahnten Blättern.",
        "care": "Wächst auf Wiesen und in Gärten als Wildkraut.",
    },
    "bellis perennis": {
        "name": "Gänseblümchen (Tausendschön)",
        "description": "Kleine, zarte Wiesenblume mit weiß-rosa Blütenblättern und gelber Mitte.",
        "care": "Wächst auf Rasenflächen, mag sonnige bis halbschattige Standorte.",
    },
    "urtica dioica": {
        "name": "Große Brennnessel",
        "description": "Heimische Wildpflanze mit Brennhaaren, aber extrem wertvoll für die Tierwelt (Schmetterlingsraupen).",
        "care": "Wächst wild an stickstoffreichen Standorten.",
    },
    "rosa canina": {
        "name": "Hunds-Rose (Heckenrose)",
        "description": "Wildstrauch mit rosa-weißen Blüten und im Herbst roten Hagebutten.",
        "care": "Sonnig, pflegeleicht, ideal für Hecken im Naturgarten.",
    },
}


# ============================================================
# GIFTIGKEITS-DATENBANK
# ============================================================

TOXICITY_INFO = {
    "monstera deliciosa": {
        "details": "Die Monstera enthält Calciumoxalat-Kristalle in Pflanzensaft und Blättern. Bei Verzehr brennendes Gefühl, Reizungen der Schleimhäute.",
        "audience": "Menschen und Haustiere (Katzen, Hunde)",
    },
    "sansevieria trifasciata": {
        "details": "Enthält Saponine. Kann bei Haustieren zu Speichelfluss, Erbrechen und Durchfall führen.",
        "audience": "Insbesondere Haustiere",
    },
    "aloe vera": {
        "details": "Die Randschicht des Blattes enthält Aloin, welches abführend und reizend wirkt.",
        "audience": "Menschen und Haustiere",
    },
    "zamioculcas zamiifolia": {
        "details": "Enthält Calciumoxalat. Berührung oder Verzehr kann Schleimhautreizungen auslösen.",
        "audience": "Menschen und Haustiere",
    },
    "epipremnum aureum": {
        "details": "Enthält Calciumoxalat-Kristalle. Reizt den Mund- und Rachenraum bei Aufnahme.",
        "audience": "Menschen und Haustiere",
    },
    "ficus elastica": {
        "details": "Der weiße Milchsaft ist haut- und schleimhautreizend.",
        "audience": "Menschen und Haustiere",
    },
    "spathiphyllum wallisii": {
        "details": "Enthält Calciumoxalat, das bei Haustieren zu starkem Speichelfluss und Magenreizung führt.",
        "audience": "Menschen und Haustiere",
    },
    "urtica dioica": {
        "details": "Brennhaare enthalten Histamin und Ameisensäure (führt zu schmerzhaftem Hautbrennen bei Berührung).",
        "audience": "Menschen und Tiere (bei Berührung)",
    },
}


# ============================================================
# SYNONYME & DEUTSCHE NAMEN ZUORDNUNG
# ============================================================

PLANT_ALIASES = {
    "monstera": "monstera deliciosa",
    "fensterblatt": "monstera deliciosa",
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
    "eiche": "quercus robur",
    "stieleiche": "quercus robur",
    "buche": "fagus sylvatica",
    "rotbuche": "fagus sylvatica",
    "birke": "betula pendula",
    "ahorn": "acer pseudoplatanus",
    "kiefer": "pinus sylvestris",
    "fichte": "picea abies",
    "loewenzahn": "taraxacum officinale",
    "löwenzahn": "taraxacum officinale",
    "gaensebluemchen": "bellis perennis",
    "gänseblümchen": "bellis perennis",
    "brennnessel": "urtica dioica",
    "rose": "rosa canina",
    "hagebutte": "rosa canina",
}


# ============================================================
# HILFSFUNKTIONEN (IMMER AUF DEUTSCH & SICHER)
# ============================================================

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
        if alias in normalized:
            canonical_name = PLANT_ALIASES[alias]
            if canonical_name in database:
                return database[canonical_name]

    for key in sorted(database, key=len, reverse=True):
        if key in normalized:
            return database[key]

    return None


def get_plant_information(label):
    result = find_database_entry(label, PLANT_INFO)
    if result:
        return result
    
    # Automatischer deutscher Fallback, damit niemals ein "komischer" Name da steht!
    clean_fallback = label.replace("_", " ").capitalize()
    return {
        "name": clean_fallback,
        "description": f"Das Modell hat diese Pflanze erkannt ({label}). In unserer Datenbank sind hierzu aktuell noch keine spezifischen Zusatzinfos hinterlegt.",
        "care": "Bitte informiere dich über den genauen Pflanzennamen für optimale Pflegebedingungen.",
    }


def get_toxicity_information(label):
    result = find_database_entry(label, TOXICITY_INFO)
    if result:
        return {"known": True, "details": result["details"], "audience": result["audience"]}
    return {
        "known": False, 
        "details": "Keine bekannten Giftstoffe in unserer Datenbank hinterlegt. Vorsicht ist dennoch geboten.", 
        "audience": "Nicht überprüft"
    }


def classify_image(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)[0]
    count = min(5, probabilities.shape[0])
    top_results = torch.topk(probabilities, k=count)
    results = []
    for score, index in zip(top_results.values, top_results.indices):
        raw_label = model.config.id2label[index.item()]
        
        # Direkt den deutschen Namen aus der DB holen falls vorhanden!
        db_entry = find_database_entry(raw_label, PLANT_INFO)
        display_name = db_entry["name"] if db_entry else raw_label.replace("_", " ").capitalize()
        
        results.append({
            "name": display_name,
            "raw": str(raw_label),
            "score": float(score.item())
        })
    return results


# ============================================================
# HEADER BEREICH
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
# UPLOAD BEREICH
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
            st.session_state.view = "care"
            st.session_state.plant_results = None
            st.session_state.analysis_error = None

        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        st.image(image, caption="Dein hochgeladenes Pflanzenfoto", width=400)

        # KI-Analyse ausführen
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
            label = best_result["raw"]
            confidence = best_result["score"] * 100

            plant = get_plant_information(label)
            toxicity = get_toxicity_information(label)

            st.divider()

            # ----------------------------------------
            # AUSWAHL-BUTTONS
            # ----------------------------------------
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("🌱 Pflanzenpflege anzeigen", use_container_width=True):
                    st.session_state.view = "care"
            with col_b2:
                if st.button("🔍 KI-Erkennung & Giftcheck", use_container_width=True):
                    st.session_state.view = "recognition"

            st.markdown("<br>", unsafe_allow_html=True)

            # ----------------------------------------
            # ANSICHT: PFLEGE
            # ----------------------------------------
            if st.session_state.view == "care":
                st.markdown('<div class="section-heading">🌱 Pflanzenpflege & Informationen</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="card-container">
                    <p class="muted">Erkannte Pflanzenart</p>
                    <div class="result-name">{html.escape(plant["name"])}</div>
                    <p style="margin-top: 10px;"><b>Beschreibung:</b> {html.escape(plant["description"])}</p>
                    <hr>
                    <h3 style="color: #174b2e; margin-bottom: 8px;">💧 Pflegehinweise</h3>
                    <p>{html.escape(plant["care"])}</p>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("⚠️ Sicherheit & Giftigkeit")
                if toxicity["known"]:
                    st.warning("Mögliche Giftigkeit: " + toxicity["details"])
                else:
                    st.info("Giftigkeit nicht überprüft: " + toxicity["details"])
                st.caption("Betroffene Gruppe laut Datenbank: " + toxicity["audience"])

            # ----------------------------------------
            # ANSICHT: ERKENNUNG & GIFTCHECK
            # ----------------------------------------
            elif st.session_state.view == "recognition":
                st.markdown('<div class="section-heading">🔍 KI-Pflanzenerkennung & Giftigkeitscheck</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="card-container">
                    <p class="muted">Wahrscheinlichste Pflanzenart</p>
                    <div class="result-name">{html.escape(plant["name"])}</div>
                    <p style="margin-top: 10px;"><b>Wissenschaftlicher Name:</b> {html.escape(label)}</p>
                    <p>{html.escape(plant["description"])}</p>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Modell-Konfidenz", f"{confidence:.2f} %")
                with col2:
                    st.metric("Weitere Ergebnisse", len(results))

                if confidence < 50:
                    st.warning("Die Erkennung ist unsicher. Bitte überprüfe die Pflanzenbestimmung.")
                else:
                    st.info("Das Modell zeigt eine vergleichsweise hohe Konfidenz. Eine sichere Bestimmung ist dadurch nicht garantiert.")

                st.subheader("⚠️ Giftigkeitscheck")
                if toxicity["known"]:
                    st.error("Für diese Pflanzenart sind mögliche Risiken bekannt.")
                else:
                    st.warning("Giftigkeit nicht überprüft – unbekannt bedeutet nicht ungefährlich.")
                
                st.write(toxicity["details"])
                st.caption("Betroffene Gruppe laut Datenbank: " + toxicity["audience"])

                st.info("Die Pflanzenbestimmung kann falsch sein. Dieser Check ist keine vollständige Giftigkeitsprüfung. Bei unbekannten Pflanzen sollten Menschen und Haustiere vorsichtig sein.")

                st.subheader("📊 Weitere KI-Ergebnisse")
                for result in results:
                    percentage = result["score"] * 100
                    st.write(f"**{result['name']}** – {percentage:.2f} %")
                    st.progress(float(result["score"]))

                st.caption("Die Prozentwerte zeigen die Modell-Konfidenz innerhalb seiner Klassen und sind keine Garantie für die tatsächliche Richtigkeit.")

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
