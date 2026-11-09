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
# DESIGN
# ============================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f3faef 0%, #e7f3e2 100%);
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    .main-title {
        color: #174b2e;
        font-size: clamp(52px, 8vw, 82px);
        font-weight: 900;
        letter-spacing: -4px;
        line-height: 1.05;
        margin-bottom: 8px;
    }

    .subtitle {
        color: #65816b;
        font-size: 19px;
        margin-bottom: 30px;
    }

    .hero {
        background: linear-gradient(120deg, #174b2e, #388653);
        padding: 38px;
        border-radius: 25px;
        color: white;
        margin: 25px 0 32px 0;
        box-shadow: 0 12px 32px rgba(23,75,46,0.16);
    }

    .hero h2 {
        color: white !important;
        font-size: 30px;
        margin-bottom: 10px;
    }

    .hero p {
        color: #e4f2e6;
        font-size: 16px;
        line-height: 1.7;
    }

    .section-heading {
        color: #174b2e;
        font-size: 27px;
        font-weight: 800;
        margin: 25px 0 15px 0;
    }

    .feature-card {
        background: white;
        padding: 27px;
        border-radius: 20px;
        border: 1px solid #dcebd8;
        min-height: 185px;
        box-shadow: 0 5px 16px rgba(30,80,40,0.06);
        transition: all 0.25s ease;
    }

    .feature-card:hover {
        transform: translateY(-7px);
        box-shadow: 0 15px 30px rgba(30,80,40,0.15);
        border-color: #68a875;
    }

    .feature-card h3 {
        color: #205b36;
        font-size: 23px;
    }

    .feature-card p {
        color: #617566;
        font-size: 15px;
        line-height: 1.6;
    }

    .info-card {
        background: white;
        padding: 27px;
        border-radius: 19px;
        border: 1px solid #dcebd8;
        margin: 15px 0;
        box-shadow: 0 5px 15px rgba(30,80,40,0.05);
    }

    .result-name {
        color: #174b2e;
        font-size: 29px;
        font-weight: 800;
    }

    .muted {
        color: #718276;
        font-size: 14px;
    }

    div[data-testid="stFileUploader"] {
        background: white;
        padding: 20px;
        border: 1px solid #dcebd8;
        border-radius: 18px;
    }

    div.stButton > button {
        background: #287344;
        color: white;
        border: none;
        border-radius: 12px;
        min-height: 48px;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background: #174b2e;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(22,75,46,0.2);
    }

    [data-testid="stDialog"] {
        border-radius: 22px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #dcebd8;
        border-radius: 15px;
        padding: 16px;
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

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID
    )

    model.eval()

    return processor, model


# ============================================================
# PFLANZEN-DATENBANK
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


# Nur ausgewählte, bekannte Beispiele.
# Nicht aufgeführte Pflanzen gelten NICHT automatisch als ungiftig.

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


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

# Alternative wissenschaftliche und deutsche Namen.
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

    # Deutsche Namen und bekannte wissenschaftliche Synonyme zuordnen.
    for alias in sorted(PLANT_ALIASES, key=len, reverse=True):
        if normalized == alias or normalized.startswith(alias + " "):
            canonical_name = PLANT_ALIASES[alias]
            if canonical_name in database:
                return database[canonical_name]

    # Zusätze hinter dem wissenschaftlichen Namen im Modell-Label erlauben.
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
        "description": (
            "Das Modell hat diese Pflanzenart erkannt. "
            "In unserer kleinen Informationsdatenbank "
            "liegen dazu noch keine genau zugeordneten Informationen vor."
        ),
        "care": (
            "Bitte informiere dich anhand des wissenschaftlichen "
            "Pflanzennamens über die individuellen Pflegeansprüche."
        ),
    }


def get_toxicity_information(label):

    result = find_database_entry(label, TOXICITY_INFO)

    if result:
        return {
            "known": True,
            "details": result["details"],
            "audience": result["audience"],
        }

    return {
        "known": False,
        "details": (
            "Diese Pflanzenart ist nicht in unserer kleinen "
            "Giftigkeitsdatenbank enthalten. Daraus lässt sich "
            "nicht schließen, dass sie ungiftig ist."
        ),
        "audience": "Nicht sicher überprüft",
    }


# ============================================================
# KI-ERKENNUNG
# ============================================================

def classify_image(image, processor, model):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.inference_mode():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    count = min(5, probabilities.shape[0])

    top_results = torch.topk(
        probabilities,
        k=count
    )

    results = []

    for score, index in zip(
        top_results.values,
        top_results.indices
    ):

        label = model.config.id2label[index.item()]

        results.append({
            "name": str(label),
            "score": float(score.item()),
        })

    return results


# ============================================================
# AUSWAHL-POPUP
# ============================================================

@st.dialog("🌿 Was möchtest du entdecken?", width="large")
def choose_mode():

    st.write(
        "Dein Pflanzenfoto ist bereit! "
        "Wähle einen Bereich aus."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="feature-card">
            <h3>🌱 Pflanzenpflege</h3>
            <p>
                Entdecke Pflegehinweise, Standort,
                Wasserbedarf und mögliche Risiken.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Pflege öffnen",
            key="care_choice",
            use_container_width=True
        ):
            st.session_state.view = "care"
            st.rerun()

    with col2:

        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Pflanzenerkennung</h3>
            <p>
                Entdecke den vermuteten Namen,
                die KI-Ergebnisse und den Giftigkeitscheck.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Erkennung öffnen",
            key="recognition_choice",
            use_container_width=True
        ):
            st.session_state.view = "recognition"
            st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌿 PlantScan</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Dein intelligenter Begleiter für gesunde Pflanzen.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="hero">
    <h2>Entdecke die Welt deiner Pflanzen.</h2>
    <p>
        Lade ein Foto hoch und entdecke mit künstlicher Intelligenz,
        welche Pflanze vor dir steht und was sie benötigt.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="section-heading">📷 Deine Pflanze entdecken</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Lade ein Foto deiner Pflanze hoch",
    type=["jpg", "jpeg", "png", "webp"],
    key="plant_upload"
)


if uploaded_file is None:

    st.markdown("""
    <div class="info-card">
        <h3>🌱 So funktioniert PlantScan</h3>
        <p>
            <b>1.</b> Lade ein Foto deiner Pflanze hoch.<br>
            <b>2.</b> Wähle zwischen Pflege und Erkennung.<br>
            <b>3.</b> Entdecke Informationen und mögliche Risiken.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:

    try:

        image_bytes = uploaded_file.getvalue()

        current_hash = hashlib.sha256(image_bytes).hexdigest()

        # Bei einem neuen Bild alte Ergebnisse zurücksetzen.
        if current_hash != st.session_state.image_hash:

            st.session_state.image_hash = current_hash
            st.session_state.view = None
            st.session_state.plant_results = None
            st.session_state.analysis_error = None

        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        st.image(
            image,
            caption="Dein hochgeladenes Pflanzenfoto",
            width=450
        )

        # Popup nach dem Upload öffnen.
        if st.session_state.view is None:
            choose_mode()

        # KI nur einmal pro Bild ausführen.
        if st.session_state.view is not None:

            if st.session_state.plant_results is None:

                try:

                    with st.spinner(
                        "🌿 Die KI untersucht deine Pflanze..."
                    ):

                        processor, model = load_model()

                        results = classify_image(
                            image,
                            processor,
                            model
                        )

                        st.session_state.plant_results = results
                        st.session_state.analysis_error = None

                except Exception as error:

                    st.session_state.analysis_error = str(error)

            if st.session_state.analysis_error:

                st.error(
                    "Die Pflanzenanalyse konnte nicht durchgeführt werden."
                )

                st.code(
                    st.session_state.analysis_error
                )

            elif st.session_state.plant_results:

                results = st.session_state.plant_results

                best_result = results[0]

                label = best_result["name"]
                confidence = best_result["score"] * 100

                plant = get_plant_information(label)
                toxicity = get_toxicity_information(label)

                st.divider()

                # ----------------------------------------
                # PFLEGE
                # ----------------------------------------

                if st.session_state.view == "care":

                    st.markdown(
                        '<div class="section-heading">🌱 Pflanzenpflege</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
                        <div class="info-card">
                            <p class="muted">Erkannte Pflanzenart</p>
                            <div class="result-name">
                                {html.escape(plant["name"])}
                            </div>
                            <p>{html.escape(plant["description"])}</p>
                            <hr>
                            <h3>💧 Pflegehinweise</h3>
                            <p>{html.escape(plant["care"])}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.subheader("⚠️ Sicherheit")

                    if toxicity["known"]:

                        st.warning(
                            "Mögliche Giftigkeit: "
                            + toxicity["details"]
                        )

                    else:

                        st.info(
                            "Giftigkeit nicht überprüft: "
                            + toxicity["details"]
                        )

                    st.caption(
                        "Betroffene Gruppe laut Datenbank: "
                        + toxicity["audience"]
                    )

                # ----------------------------------------
                # ERKENNUNG
                # ----------------------------------------

                elif st.session_state.view == "recognition":

                    st.markdown(
                        '<div class="section-heading">🔍 KI-Pflanzenerkennung</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
                        <div class="info-card">
                            <p class="muted">Wahrscheinlichste Pflanzenart</p>
                            <div class="result-name">
                                {html.escape(plant["name"])}
                            </div>
                            <p>{html.escape(label)}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Modell-Konfidenz",
                            f"{confidence:.2f} %"
                        )

                    with col2:
                        st.metric(
                            "Weitere Ergebnisse",
                            len(results)
                        )

                    if confidence < 50:

                        st.warning(
                            "Die Erkennung ist unsicher. "
                            "Bitte überprüfe die Pflanzenbestimmung."
                        )

                    else:

                        st.info(
                            "Das Modell zeigt eine vergleichsweise "
                            "hohe Konfidenz. Eine sichere Bestimmung "
                            "ist dadurch nicht garantiert."
                        )

                    st.subheader("⚠️ Giftigkeitscheck")

                    if toxicity["known"]:

                        st.error(
                            "Für diese Pflanzenart sind mögliche Risiken bekannt."
                        )

                    else:

                        st.warning(
                            "Giftigkeit nicht überprüft – "
                            "unbekannt bedeutet nicht ungefährlich."
                        )

                    st.write(toxicity["details"])

                    st.caption(
                        "Betroffene Gruppe laut Datenbank: "
                        + toxicity["audience"]
                    )

                    st.info(
                        "Die Pflanzenbestimmung kann falsch sein. "
                        "Dieser Check ist keine vollständige "
                        "Giftigkeitsprüfung. Bei unbekannten Pflanzen "
                        "sollten Menschen und Haustiere vorsichtig sein."
                    )

                    st.subheader("📊 Weitere KI-Ergebnisse")

                    for result in results:

                        percentage = result["score"] * 100

                        st.write(
                            f"**{result['name']}** – {percentage:.2f} %"
                        )

                        st.progress(
                            float(result["score"])
                        )

                    st.caption(
                        "Die Prozentwerte zeigen die Modell-Konfidenz "
                        "innerhalb seiner Klassen und sind keine "
                        "Garantie für die tatsächliche Richtigkeit."
                    )

                # ----------------------------------------
                # ZURÜCK
                # ----------------------------------------

                st.divider()

                if st.button(
                    "↩ Zurück zur Auswahl",
                    use_container_width=True
                ):

                    st.session_state.view = None
                    st.rerun()

    except Exception as error:

        st.error(
            "Das Bild konnte nicht verarbeitet werden."
        )

        st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
<div style="
    text-align:center;
    padding:25px;
    color:#55745d;
">
    <h3 style="color:#174b2e;">🌿 PlantScan</h3>
    Dein digitaler Pflanzenhelfer<br>
    <small>Ein KI-Schulprojekt</small>
</div>
""", unsafe_allow_html=True)
