
import streamlit as st
import torch
import html
import re

from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification


# ==================================================
# PLANTSCAN - KONFIGURATION
# ==================================================

MODEL_ID = "domai-tb/OpenPlants-Identification-ViT-Base-Patch16-224"

st.set_page_config(
    page_title="PlantScan | KI-Pflanzenhelfer",
    page_icon="🌿",
    layout="wide"
)


# ==================================================
# DESIGN
# ==================================================

st.markdown("""
<style>
    .stApp {
        background-color: #f5f7f2;
    }

    h1, h2, h3 {
        color: #183b2b;
    }

    .main-title {
        font-size: 43px;
        font-weight: 800;
        color: #183b2b;
        margin-bottom: 0;
    }

    .subtitle {
        color: #68796d;
        font-size: 17px;
        margin-top: 5px;
    }

    .info-card {
        background-color: white;
        padding: 23px;
        border-radius: 15px;
        border: 1px solid #e2e9df;
        margin-bottom: 15px;
    }

    .result-name {
        font-size: 25px;
        font-weight: 700;
        color: #183b2b;
    }

    .small-text {
        color: #68796d;
        font-size: 14px;
    }

    .stButton > button {
        background-color: #285c40;
        color: white;
        border-radius: 9px;
        border: none;
        padding: 10px 22px;
    }

    .stButton > button:hover {
        background-color: #397650;
        color: white;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #e2e9df;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# MODELL LADEN
# ==================================================

@st.cache_resource(show_spinner=False)
def load_model():

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID
    )

    model.eval()

    return processor, model


# ==================================================
# HILFSFUNKTIONEN
# ==================================================

def normalize_name(name):

    name = name.lower().strip()
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name)

    return name


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

    top_results = torch.topk(
        probabilities,
        k=min(5, len(probabilities))
    )

    results = []

    for score, index in zip(
        top_results.values,
        top_results.indices
    ):

        label = model.config.id2label[index.item()]

        results.append({
            "name": label,
            "score": score.item()
        })

    return results


# ==================================================
# PFLANZENINFORMATIONEN
# ==================================================

PLANT_INFO = {

    "monstera deliciosa": {
        "name": "Monstera",
        "description": "Eine beliebte tropische Zimmerpflanze mit großen, eingeschnittenen Blättern.",
        "care": "Helles, indirektes Licht. Mäßig gießen und Staunässe vermeiden."
    },

    "sansevieria trifasciata": {
        "name": "Bogenhanf",
        "description": "Eine robuste Zimmerpflanze mit langen, aufrecht wachsenden Blättern.",
        "care": "Heller bis halbschattiger Standort. Nur sparsam gießen."
    },

    "chlorophytum comosum": {
        "name": "Grünlilie",
        "description": "Eine verbreitete Zimmerpflanze mit langen, schmalen Blättern.",
        "care": "Heller Standort ohne starke direkte Mittagssonne. Regelmäßig, aber mäßig gießen."
    },

    "aloe vera": {
        "name": "Aloe vera",
        "description": "Eine sukkulente Pflanze mit dicken, fleischigen Blättern.",
        "care": "Viel Licht und sparsam gießen. Erde zwischen den Wassergaben trocknen lassen."
    },

    "zamioculcas zamiifolia": {
        "name": "Glücksfeder",
        "description": "Eine robuste Zimmerpflanze mit glänzenden, kräftigen Blättern.",
        "care": "Heller bis halbschattiger Standort. Wenig Wasser und keine Staunässe."
    },

    "epipremnum aureum": {
        "name": "Efeutute",
        "description": "Eine Kletterpflanze mit herzförmigen Blättern.",
        "care": "Heller Standort ohne starke direkte Sonne. Mäßig gießen."
    },

    "ficus elastica": {
        "name": "Gummibaum",
        "description": "Eine Zimmerpflanze mit großen, dicken und glänzenden Blättern.",
        "care": "Heller Standort und mäßige Wassergaben."
    },

    "ficus lyrata": {
        "name": "Geigenfeige",
        "description": "Eine Zimmerpflanze mit großen, geigenförmigen Blättern.",
        "care": "Heller Standort ohne starke direkte Mittagssonne."
    },

    "spathiphyllum wallisii": {
        "name": "Einblatt",
        "description": "Eine Zimmerpflanze mit dunkelgrünen Blättern und weißen Blüten.",
        "care": "Heller bis halbschattiger Standort. Erde leicht feucht halten."
    },

    "dracaena marginata": {
        "name": "Drachenbaum",
        "description": "Eine Zimmerpflanze mit schmalen, langen Blättern.",
        "care": "Heller Standort und mäßig gießen."
    },

    "crassula ovata": {
        "name": "Geldbaum",
        "description": "Eine Sukkulente mit dicken, ovalen Blättern.",
        "care": "Viel Licht und sparsam gießen."
    },

    "yucca elephantipes": {
        "name": "Yucca-Palme",
        "description": "Eine Zimmerpflanze mit holzigem Stamm und langen, spitzen Blättern.",
        "care": "Heller Standort und mäßige Wassergaben."
    }
}


# ==================================================
# GIFTIGKEITSINFORMATIONEN
# ==================================================

# Diese Liste enthält nur ausgewählte, bekannte Beispiele.
# Nicht aufgeführte Pflanzen sind NICHT automatisch ungiftig.

TOXICITY_INFO = {

    "monstera deliciosa": {
        "status": "giftig",
        "title": "Für Haustiere problematisch",
        "details": "Die Monstera enthält reizende Calciumoxalat-Kristalle. Sie kann insbesondere für Katzen und Hunde problematisch sein.",
        "audience": "Katzen und Hunde"
    },

    "sansevieria trifasciata": {
        "status": "giftig",
        "title": "Für Haustiere problematisch",
        "details": "Der Bogenhanf enthält Saponine und kann bei Aufnahme Beschwerden verursachen.",
        "audience": "Katzen und Hunde"
    },

    "aloe vera": {
        "status": "giftig",
        "title": "Für Haustiere problematisch",
        "details": "Bestandteile der Aloe vera können bei Haustieren Beschwerden verursachen.",
        "audience": "Katzen und Hunde"
    },

    "zamioculcas zamiifolia": {
        "status": "giftig",
        "title": "Reizende Pflanzenbestandteile",
        "details": "Die Glücksfeder enthält Calciumoxalat-Kristalle, die Schleimhäute reizen können.",
        "audience": "Menschen und Haustiere"
    },

    "epipremnum aureum": {
        "status": "giftig",
        "title": "Für Haustiere problematisch",
        "details": "Die Efeutute enthält Calciumoxalat-Kristalle und kann bei Aufnahme Reizungen verursachen.",
        "audience": "Menschen und Haustiere"
    },

    "ficus elastica": {
        "status": "giftig",
        "title": "Vorsicht bei Haustieren",
        "details": "Der Gummibaum enthält einen reizenden Milchsaft.",
        "audience": "Menschen und Haustiere"
    },

    "spathiphyllum wallisii": {
        "status": "giftig",
        "title": "Reizende Pflanzenbestandteile",
        "details": "Das Einblatt enthält Calciumoxalat-Kristalle, die bei Kontakt oder Aufnahme reizen können.",
        "audience": "Menschen und Haustiere"
    },

    "chlorophytum comosum": {
        "status": "ungiftig",
        "title": "Als ungiftig für Haustiere gelistet",
        "details": "Die Grünlilie wird in der ASPCA-Datenbank als ungiftig für Katzen und Hunde geführt.",
        "audience": "Katzen und Hunde"
    }
}


def get_plant_information(label):

    normalized = normalize_name(label)

    if normalized in PLANT_INFO:
        return PLANT_INFO[normalized]

    return {
        "name": label,
        "description": "Das Modell hat diese Pflanzenart erkannt. Es liegen in unserer kleinen Informationsdatenbank keine genau zugeordneten Informationen vor.",
        "care": "Bitte informiere dich anhand des wissenschaftlichen Pflanzennamens über die individuellen Pflegeansprüche."
    }


def get_toxicity_information(label):

    normalized = normalize_name(label)

    if normalized in TOXICITY_INFO:
        return TOXICITY_INFO[normalized]

    return {
        "status": "unbekannt",
        "title": "Giftigkeit nicht überprüft",
        "details": "Diese Pflanze ist nicht in unserer kleinen Giftigkeitsdatenbank enthalten. Daraus lässt sich nicht schließen, dass sie ungiftig ist.",
        "audience": "Nicht sicher bestimmt"
    }


# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<p class="main-title">🌿 PlantScan</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Dein KI-Pflanzenhelfer – Pflanzen erkennen, Pflege erfahren und Risiken entdecken.</p>',
    unsafe_allow_html=True
)

st.write("")

st.markdown("""
<div class="info-card">
    <h3>🌱 Willkommen bei PlantScan</h3>
    Lade ein Foto einer Pflanze hoch. Unsere künstliche Intelligenz
    versucht, die Pflanzenart zu bestimmen. Anschließend erhältst du
    Informationen über die Pflanze, ihre Pflege und mögliche Risiken.
</div>
""", unsafe_allow_html=True)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("🌿 PlantScan")

    st.write("Dein digitaler Pflanzenhelfer.")

    st.divider()

    st.markdown("### Funktionen")

    st.write("📷 Pflanzen erkennen")
    st.write("🌱 Pflegeinformationen")
    st.write("⚠️ Giftigkeit überprüfen")

    st.divider()

    st.caption(
        "Version B – vortrainiertes Hugging-Face-Modell"
    )


# ==================================================
# BILDUpload
# ==================================================

st.subheader("📷 Pflanze hochladen")

uploaded_file = st.file_uploader(
    "Wähle ein Pflanzenfoto aus",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file)

        image = ImageOps.exif_transpose(image).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Dein Pflanzenfoto",
                width=350
            )

        with col2:

            st.markdown("### Bereit zur Analyse?")

            st.write(
                "PlantScan analysiert das Bild und sucht "
                "nach den wahrscheinlichsten Pflanzenarten."
            )

            analyze = st.button(
                "🌿 Pflanze erkennen",
                use_container_width=True
            )

        if analyze:

            with st.spinner(
                "Das KI-Modell wird geladen und analysiert deine Pflanze..."
            ):

                processor, model = load_model()

                results = classify_image(
                    image,
                    processor,
                    model
                )

            best_result = results[0]

            label = best_result["name"]

            confidence = best_result["score"] * 100

            plant = get_plant_information(label)

            toxicity = get_toxicity_information(label)

            st.divider()

            st.subheader("🔍 Ergebnis der Erkennung")

            st.markdown(
                f"""
                <div class="info-card">
                    <p class="small-text">Wahrscheinlich erkannte Pflanzenart</p>
                    <p class="result-name">{html.escape(plant["name"])}</p>
                    <p class="small-text">
                        Modell-Konfidenz: {confidence:.2f} %
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if confidence >= 50:

                st.success(
                    "Das Modell zeigt eine vergleichsweise hohe Konfidenz. "
                    "Die Bestimmung ist trotzdem nicht garantiert."
                )

            else:

                st.warning(
                    "Die Erkennung ist unsicher. "
                    "Bitte überprüfe die Pflanzenbestimmung."
                )

            # ------------------------------------------
            # PFLANZENINFORMATIONEN
            # ------------------------------------------

            st.subheader("🌱 Pflanzeninformationen")

            st.write(plant["description"])

            st.markdown("**Allgemeine Pflegehinweise:**")

            st.write(plant["care"])

            # ------------------------------------------
            # GIFTIGKEITSCHECK
            # ------------------------------------------

            st.divider()

            st.subheader("⚠️ Giftigkeits- und Gefahrencheck")

            if toxicity["status"] == "giftig":

                st.error(
                    f"⚠️ {toxicity['title']}"
                )

            elif toxicity["status"] == "ungiftig":

                st.success(
                    f"✓ {toxicity['title']}"
                )

            else:

                st.warning(
                    f"❔ {toxicity['title']}"
                )

            st.write(toxicity["details"])

            st.caption(
                "Betroffene Gruppe laut Datenbank: "
                + toxicity["audience"]
            )

            st.info(
                "Wichtig: Dieser Check ist keine vollständige "
                "Giftigkeitsprüfung. Die Pflanzenbestimmung kann falsch sein. "
                "Bitte lasse Pflanzen nicht von Kindern oder Haustieren "
                "probieren. Bei möglicher Aufnahme einer unbekannten "
                "oder giftigen Pflanze sollte fachlicher Rat eingeholt werden."
            )

            # ------------------------------------------
            # WEITERE ERGEBNISSE
            # ------------------------------------------

            st.divider()

            st.subheader("📊 Weitere mögliche Pflanzenarten")

            for result in results:

                percentage = result["score"] * 100

                safe_label = html.escape(result["name"])

                st.write(
                    f"**{safe_label}** – {percentage:.2f} %"
                )

                st.progress(
                    float(result["score"])
                )

            st.caption(
                "Die Prozentwerte beschreiben die Modell-Konfidenz "
                "innerhalb seiner Klassen. Sie sind keine Garantie "
                "für die tatsächliche Richtigkeit."
            )

    except Exception as error:

        st.error(
            "Bei der Verarbeitung ist ein Fehler aufgetreten."
        )

        st.exception(error)


else:

    st.info(
        "Lade ein Foto hoch, um mit der Pflanzenanalyse zu beginnen."
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.markdown("""
<div style="text-align:center; color:#68796d; padding:15px;">
    🌿 PlantScan – Ein KI-Schulprojekt<br>
    <small>Powered by Hugging Face Transformers</small>
</div>
""", unsafe_allow_html=True)
