
import streamlit as st
import torch

from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification


# --------------------------------------------------
# PLANTSCAN - KONFIGURATION
# --------------------------------------------------

MODEL_ID = "domai-tb/OpenPlants-Identification-ViT-Base-Patch16-224"

st.set_page_config(
    page_title="PlantScan | KI-Pflanzenhelfer",
    page_icon="🌿",
    layout="wide"
)


# --------------------------------------------------
# DESIGN
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #f5f7f2;
    }

    h1, h2, h3 {
        color: #183b2b;
    }

    .main-title {
        font-size: 42px;
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
        padding: 22px;
        border-radius: 14px;
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

    div.stButton > button {
        background-color: #285c40;
        color: white;
        border-radius: 9px;
        border: none;
        padding: 10px 22px;
    }

    div.stButton > button:hover {
        background-color: #397650;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODELL LADEN
# --------------------------------------------------

@st.cache_resource
def load_model():

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID
    )

    model.eval()

    return processor, model


# --------------------------------------------------
# PFLANZENINFORMATIONEN
# --------------------------------------------------

def get_plant_information(label):

    name = label.lower().replace("_", " ")

    plants = {
        "monstera": {
            "description": "Die Monstera ist eine beliebte tropische Zimmerpflanze.",
            "care": "Helles, indirektes Licht und mäßiges Gießen."
        },
        "aloe": {
            "description": "Aloe ist eine sukkulente Pflanzengattung.",
            "care": "Viel Licht und eher sparsam gießen."
        },
        "sansevieria": {
            "description": "Der Bogenhanf ist eine robuste Zimmerpflanze.",
            "care": "Heller bis halbschattiger Standort und wenig Wasser."
        },
        "snake plant": {
            "description": "Der Bogenhanf besitzt aufrecht wachsende Blätter.",
            "care": "Heller bis halbschattiger Standort und wenig Wasser."
        },
        "chlorophytum": {
            "description": "Die Grünlilie ist eine verbreitete Zimmerpflanze.",
            "care": "Heller Standort ohne starke direkte Mittagssonne."
        },
        "spider plant": {
            "description": "Die Grünlilie bildet lange, schmale Blätter.",
            "care": "Heller Standort und regelmäßig, aber mäßig gießen."
        },
        "zamioculcas": {
            "description": "Die Glücksfeder besitzt glänzende, kräftige Blätter.",
            "care": "Halbschattiger Standort und sparsam gießen."
        },
        "zz plant": {
            "description": "Die Glücksfeder ist eine robuste Zimmerpflanze.",
            "care": "Halbschattiger Standort und sparsam gießen."
        },
        "epipremnum": {
            "description": "Die Efeutute ist eine beliebte Kletterpflanze.",
            "care": "Heller Standort und mäßig feucht halten."
        },
        "pothos": {
            "description": "Die Efeutute besitzt herzförmige Blätter.",
            "care": "Heller Standort und mäßig feucht halten."
        }
    }

    for plant_key, info in plants.items():
        if plant_key in name:
            return info

    return {
        "description": "Das Modell hat eine Pflanzenart erkannt.",
        "care": "Bitte prüfe die genaue Pflanzenart und ihre individuellen Pflegeansprüche."
    }


# --------------------------------------------------
# PFLANZE ERKENNEN
# --------------------------------------------------

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


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<p class="main-title">🌿 PlantScan</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Dein KI-Pflanzenhelfer – Pflanzen erkennen und mehr über sie erfahren.</p>',
    unsafe_allow_html=True
)

st.write("")

st.markdown("""
<div class="info-card">
    <h3>🌱 Willkommen bei PlantScan</h3>
    Lade ein Foto einer Pflanze hoch und lasse unsere
    künstliche Intelligenz versuchen, die Pflanzenart zu bestimmen.
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# BILDUpload
# --------------------------------------------------

st.subheader("📷 Pflanze hochladen")

uploaded_file = st.file_uploader(
    "Wähle ein Pflanzenfoto aus",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    image = ImageOps.exif_transpose(image).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:

        st.image(
            image,
            caption="Dein hochgeladenes Pflanzenfoto",
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

        try:

            with st.spinner(
                "Die KI analysiert deine Pflanze..."
            ):

                processor, model = load_model()

                results = classify_image(
                    image,
                    processor,
                    model
                )

            best_result = results[0]

            confidence = best_result["score"] * 100

            st.divider()

            st.subheader("🔍 Ergebnis der Analyse")

            if confidence >= 50:

                st.success("Die KI hat eine relativ deutliche Vermutung.")

            elif confidence >= 20:

                st.warning("Die Erkennung ist unsicher.")

            else:

                st.warning(
                    "Die KI ist sich sehr unsicher. "
                    "Das Ergebnis sollte überprüft werden."
                )

            st.markdown(
                f"""
                <div class="info-card">
                    <p class="small-text">Wahrscheinlichste Pflanzenart</p>
                    <p class="result-name">{best_result["name"]}</p>
                    <p class="small-text">
                        Modell-Konfidenz: {confidence:.2f} %
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Pflegeinformationen

            info = get_plant_information(
                best_result["name"]
            )

            st.subheader("🌱 Pflanzeninformationen")

            st.write(info["description"])

            st.markdown("**Allgemeine Pflegehinweise:**")

            st.write(info["care"])

            # Weitere Vorhersagen

            st.subheader("📊 Weitere mögliche Ergebnisse")

            for result in results:

                percentage = result["score"] * 100

                st.write(
                    f"**{result['name']}** – {percentage:.2f} %"
                )

                st.progress(
                    float(result["score"])
                )

            st.caption(
                "Die Prozentwerte sind Modell-Konfidenzen "
                "und keine Garantie für die tatsächliche Richtigkeit."
            )

            st.info(
                "Hinweis: PlantScan kann Pflanzen falsch erkennen. "
                "Bitte überprüfe die Ergebnisse bei Bedarf anhand "
                "weiterer Merkmale."
            )

        except Exception as error:

            st.error(
                f"Bei der Analyse ist ein Fehler aufgetreten: {error}"
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown("""
<div style="text-align:center; color:#68796d; padding:15px;">
    🌿 PlantScan – Ein KI-Schulprojekt<br>
    <small>Powered by Hugging Face Transformers</small>
</div>
""", unsafe_allow_html=True)
