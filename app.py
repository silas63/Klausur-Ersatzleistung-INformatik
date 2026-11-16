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
# DESIGN & CSS
# ============================================================

st.markdown("""
<style>
    .stApp {
        background: 
            radial-gradient(circle at center, rgba(235, 245, 230, 0.95) 30%, rgba(200, 225, 195, 0.8) 70%, rgba(160, 200, 150, 0.6) 100%),
            url('https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?q=80&w=1600&auto=format&fit=crop') center center / cover no-repeat fixed;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    .top-header-row {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 4px;
    }

    .logo-svg {
        width: 54px;
        height: 54px;
        flex-shrink: 0;
    }

    .main-title {
        color: #163b22;
        font-size: 48px;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin: 0;
        line-height: 1;
    }

    .subtitle {
        color: #3d5a42;
        font-size: 16px;
        margin-bottom: 24px;
        margin-top: 4px;
        font-weight: 500;
    }

    .hero {
        background: 
            linear-gradient(90deg, rgba(15, 48, 22, 0.95) 0%, rgba(20, 58, 28, 0.82) 55%, rgba(20, 58, 28, 0.3) 100%),
            url('https://images.unsplash.com/photo-1542273917363-3b1817f69a2d?q=80&w=1200&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        padding: 38px 42px;
        border-radius: 22px;
        color: white;
        margin: 10px 0 26px 0;
        box-shadow: 0 12px 32px rgba(20, 55, 30, 0.16);
    }

    .hero h2 {
        color: white !important;
        font-size: 27px;
        margin-top: 0;
        margin-bottom: 10px;
        font-weight: 700;
    }

    .hero p {
        color: #e6f2e8;
        font-size: 15px;
        line-height: 1.55;
        margin: 0;
        max-width: 480px;
    }

    .card-container {
        background: rgba(250, 252, 247, 0.94);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.9);
        border-radius: 22px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 28px rgba(25, 65, 35, 0.08);
    }

    .card-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 6px;
    }

    .card-icon-box {
        background: #e4ede0;
        width: 44px;
        height: 44px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        color: #163b22;
    }

    .card-title {
        color: #163b22;
        font-size: 21px;
        font-weight: 700;
        margin: 0;
    }

    .card-desc {
        color: #4f6652;
        font-size: 15px;
        margin-bottom: 18px;
        margin-left: 58px;
    }

    .step-row {
        display: flex;
        align-items: center;
        gap: 14px;
        margin: 12px 0;
        color: #243f28;
        font-size: 15px;
        font-weight: 500;
    }

    .step-badge {
        background-color: #266b38;
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        font-weight: 700;
        flex-shrink: 0;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(243, 247, 239, 0.8);
        border: 1.5px dashed #9eb59a;
        border-radius: 16px;
        padding: 12px;
    }
    div[data-testid="stFileUploader"] section {
        padding: 14px;
    }

    .section-heading {
        color: #163b22;
        font-size: 24px;
        font-weight: 800;
        margin: 25px 0 15px 0;
    }

    .result-name {
        color: #163b22;
        font-size: 26px;
        font-weight: 800;
    }

    .muted {
        color: #657a68;
        font-size: 13px;
    }

    div.stButton > button {
        background: #266b38;
        color: white;
        border: none;
        border-radius: 12px;
        min-height: 46px;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background: #163b22;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(22, 59, 34, 0.2);
    }

    div[data-testid="stMetric"] {
        background: rgba(250, 252, 247, 0.94);
        border: 1px solid #dcebd8;
        border-radius: 14px;
        padding: 14px;
    }

    hr {
        border-color: #d2e0ce;
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
# SPEZIFISCHE PFLANZEN-DATENBANK (EXKLUSIVE TEXTE)
# ============================================================

PLANT_INFO = {
    "monstera deliciosa": {
        "name": "Monstera (Fensterblatt)",
        "description": "Die Monstera deliciosa stammt aus den tropischen Regenwäldern Mittel- und Südamerikas. Sie ist weltberühmt für ihre großen, dunkelgrünen und tief eingeschnittenen Blätter, die in freier Natur dazu dienen, Sonnenlicht und Regenwasser in tiefere Waldbereiche durchzulassen. Als Zimmerpflanze wächst sie stark kletternd, entwickelt faszinierende Luftwurzeln und benötigt eine stabile Rankhilfe.",
        "care": "• Standort: Helles bis halbschattiges Plätzchen ohne direkte grelle Mittagssonne.\n• Gießen: Obere Erdschicht (2-3 cm) vor jedem Gießen antrocknen lassen. Staunässe unbedingt vermeiden.\n• Temperatur & Luft: Normale Zimmertemperaturen (18–25 °C) und regelmäßiges Besprühen mit kalkarmem Wasser.",
    },
    "sansevieria trifasciata": {
        "name": "Bogenhanf (Sansevieria)",
        "description": "Der Bogenhanf ist eine der absolut robustesten Zimmerpflanzen. Ursprünglich aus den trockenen Regionen Westafrikas, speichert er Wasser in seinen dickfleischigen, aufrecht stehenden Blättern. Gilt als exzellenter Luftreiniger.",
        "care": "• Standort: Von sonnig bis schattig flexibel einsetzbar.\n• Gießen: Äußerst sparsam. Die Erde vor dem nächsten Wässern komplett austrocknen lassen.\n• Besonderheit: Verzeiht wochenlange Trockenheit problemlos.",
    },
    "chlorophytum comosum": {
        "name": "Grünlilie",
        "description": "Ein unkomplizierter Haushaltsklassiker mit schmalen, grün-weiß gestreiften Blättern. Bildet an langen Ausläufern kleine Tochterpflanzen (Ableger) aus.",
        "care": "• Standort: Hell bis halbschattig, keine pralle Mittagssonne.\n• Gießen: Gleichmäßig feucht halten; im Sommer regelmäßig, im Winter sparsamer.\n• Düngung: Von März bis August alle zwei Wochen düngen.",
    },
    "aloe vera": {
        "name": "Aloe vera",
        "description": "Eine sukkulente Heilpflanze aus der Familie der Affodillgewächse mit dicken Rosettenblättern, die ein feuchtigkeitsspendendes Gel enthalten.",
        "care": "• Standort: Vollsonnig, warm und luftig (Südfenster).\n• Gießen: Sehr sparsam in durchlässigem Substrat. Staunässe tötet die Wurzeln.",
    },
    "zamioculcas zamiifolia": {
        "name": "Glücksfeder",
        "description": "Besitzt glänzende, dunkelgrüne Blätter an kräftigen Stielen und unterirdische Speicherknollen zur Überbrückung von Durststrecken.",
        "care": "• Standort: Extrem flexibel, toleriert auch dunklere Ecken.\n• Gießen: Erst gießen, wenn die Erde komplett trocken ist.",
    },
    "epipremnum aureum": {
        "name": "Efeutute",
        "description": "Eine wüchsige Kletter- und Ampelpflanze mit herzförmigen, oft panaschierten Blättern, die schnell an Wänden entlang rankt.",
        "care": "• Standort: Hell bis halbschattig ohne direkte Sonne.\n• Gießen: Mäßig feucht halten, Oberfläche vor dem Wässern antrocknen lassen.",
    },
    "ficus elastica": {
        "name": "Gummibaum",
        "description": "Imponiert durch große, lederartige, dunkelgrüne Blätter und kann im Zimmer zu einem kleinen Baum heranwachsen.",
        "care": "• Standort: Sehr hell, mit etwas Morgen- oder Abendsonne.\n• Gießen: Mäßig mit kalkarmem Wasser gießen.",
    },
    "ficus lyrata": {
        "name": "Geigenfeige",
        "description": "Verdankt ihren Namen den großen, geigenförmigen Blättern mit markanten hellerleuchteten Adern. Ein echter Design-Hingucker.",
        "care": "• Standort: Sehr hell, geschützt vor Zugluft.\n• Gießen: Gleichmäßig leicht feucht halten.",
    },
    "spathiphyllum wallisii": {
        "name": "Einblatt",
        "description": "Zeichnet sich durch sattgrüne Blätter und verlässliche weiße Hochblätter (Blüten) aus.",
        "care": "• Standort: Halbschattig bis hell, keine direkte Sonne.\n• Gießen: Hoher Wasserbedarf, Erde stets leicht feucht halten.",
    },
    "dracaena marginata": {
        "name": "Drachenbaum",
        "description": "Erinnert optisch an eine Palme mit schlanken Stämmen und schopfartigen Blättern mit rötlichem Rand.",
        "care": "• Standort: Hell bis halbschattig.\n• Gießen: Mäßig gießen, Erde zwischendurch gut abtrocknen lassen.",
    },
    "crassula ovata": {
        "name": "Geldbaum",
        "description": "Klassische Sukkulente mit dicken, ovalen Blättern und verholzendem Stamm. Gilt im Feng-Shui als Glücksbringer.",
        "care": "• Standort: Sonnig und warm.\n• Gießen: Sparsam, erst bei trockener Erde.",
    },
    "yucca elephantipes": {
        "name": "Yucca-Palme",
        "description": "Besitzt einen kräftigen Stamm und lange, steif-spitze Blätter. Extrem widerstandsfähig.",
        "care": "• Standort: Maximal hell, vollsonnig.\n• Gießen: Mäßig gießen.",
    },
    "quercus robur": {
        "name": "Stiel-Eiche",
        "description": "Majestätischer heimischer Laubbaum mit gelappten Blättern und Eicheln, wird Hunderte Jahre alt.",
        "care": "• Standort: Ausschließlich im Freien in der Sonne.\n• Boden: Tiefgründig und nährstoffreich.",
    },
    "fagus sylvatica": {
        "name": "Rot-Buche",
        "description": "Prägt als dominanter Laubbaum mit glatter Rinde mitteleuropäische Wälder.",
        "care": "• Standort: Als Park- oder Waldbaum im Freien.\n• Boden: Frische, kalkhaltige Böden.",
    },
    "betula pendula": {
        "name": "Sand-Birke",
        "description": "Sofort an der leuchtend weißen Rinde und den hängenden Zweigen erkennbar. Ein Lichtliebhaber.",
        "care": "• Standort: Volle Sonne.\n• Boden: Kommt mit mageren Sandböden klar.",
    },
    "acer pseudoplatanus": {
        "name": "Berg-Ahorn",
        "description": "Stattlicher Laubbaum mit großen fünfzipfeligen Blättern und geflügelten Früchten.",
        "care": "• Standort: Sonnig bis halbschattig im Freien.",
    },
    "pinus sylvestris": {
        "name": "Wald-Kiefer",
        "description": "Immergrüner Nadelbaum mit rotbrauner, abblätternder Rinde im oberen Stammbereich.",
        "care": "• Standort: Volle Sonne, luftiger Platz.",
    },
    "picea abies": {
        "name": "Gemeine Fichte",
        "description": "Der in Deutschland weit verbreitete Nadelbaum und klassische Weihnachtsbaum.",
        "care": "• Standort: Kühle, feuchtere Freilandstandorte.",
    },
    "taraxacum officinale": {
        "name": "Wiesen-Löwenzahn",
        "description": "Wildpflanze mit gelben Blütenkörben und Pusteblumen-Samen, wichtige Bienenweide.",
        "care": "• Standort: Sonnig auf Wiesen.",
    },
    "bellis perennis": {
        "name": "Gänseblümchen",
        "description": "Zarte Wiesenblume mit weiß-rosa Blüten, die sich nachts schließen.",
        "care": "• Standort: Sonnige Rasenflächen.",
    },
    "urtica dioica": {
        "name": "Große Brennnessel",
        "description": "Heimische Wildpflanze mit Brennhaaren, wertvoll als Raupenfutter.",
        "care": "• Standort: Nährstoffreiche Standorte an Waldrändern.",
    },
    "rosa canina": {
        "name": "Hunds-Rose",
        "description": "Wildstrauch mit rosa-weißen Blüten und roten Hagebutten im Herbst.",
        "care": "• Standort: Sonnig bis halbschattig.",
    },
    "arundina graminifolia": {
        "name": "Bambus-Orchidee (Arundina graminifolia)",
        "description": "Die Arundina graminifolia, im deutschen Sprachraum als Bambus-Orchidee bekannt, ist eine faszinierende terrestrische Orchideenart aus Asien (unter anderem in Indien, Südchina und Südostasien beheimatet). Sie verdankt ihren Namen den schlanken, rohrartigen, bambusähnlichen Stängeln und den schmalen, grasartigen Blättern. Sie bildet wunderschöne, cattleyen-ähnliche weiße oder zartrosa Blüten mit einer purpurroten Lippe aus, die fast das ganze Jahr über blühen können.",
        "care": "• Standort: Sehr hell bis sonnig, verträgt auch direkte Morgensonne, sollte aber im Hochsommer vor sengender Hitze geschützt werden.\n• Gießen: Regelmäßig und gleichmäßig feucht halten; das Substrat darf niemals komplett austrocknen, allerdings ist auch hier absolute Staunässe zu vermeiden.\n• Temperatur & Luft: Bevorzigt warme, tropische bis subtropische Temperaturen (20–30 °C) und eine hohe Luftfeuchtigkeit.",
    }
}


# ============================================================
# GIFTIGKEITS-DATENBANK
# ============================================================

TOXICITY_INFO = {
    "monstera deliciosa": {
        "details": "Enthält Calciumoxalat-Kristalle. Verursacht brennendes Gefühl im Mund und Schleimhautreizungen.",
        "audience": "Menschen und Haustiere (Hunde, Katzen)",
    },
    "sansevieria trifasciata": {
        "details": "Enthält Saponine. Führt bei Haustieren zu Erbrechen und starkem Speichelfluss.",
        "audience": "Insbesondere Haustiere",
    },
    "aloe vera": {
        "details": "Enthält Aloin in der Blatthaut, wirkt abführend und reizend.",
        "audience": "Menschen und Haustiere",
    },
    "zamioculcas zamiifolia": {
        "details": "Enthält Calciumoxalat-Kristalle, führt zu Schwellungen und Speichelfluss.",
        "audience": "Menschen und Haustiere",
    },
    "epipremnum aureum": {
        "details": "Enthält Calciumoxalat-Nadeln, ruft Schleimhautschwellungen hervor.",
        "audience": "Menschen und Haustiere",
    },
    "ficus elastica": {
        "details": "Milchsaft enthält reizende Enzyme, die zu Hautentzündungen führen können.",
        "audience": "Menschen und Haustiere",
    },
    "spathiphyllum wallisii": {
        "details": "Enthält Calciumoxalat, bewirkt starkes Brennen im Mund.",
        "audience": "Menschen und Haustiere",
    },
    "urtica dioica": {
        "details": "Brennhaare enthalten Histamin und Ameisensäure (schmerzhafter Hautreiz bei Berührung).",
        "audience": "Menschen und Tiere",
    },
    "arundina graminifolia": {
        "details": "Für die Bambus-Orchidee (Arundina graminifolia) sind in der klassischen Veterinär- und Humanliteratur keine giftigen Inhaltsstoffe bekannt. Sie gilt im Allgemeinen als ungiftig für Haustiere wie Hunde und Katzen.",
        "audience": "Keine Giftstoffe bekannt",
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
    "arundina graminifolia": "arundina graminifolia",
    "bambus orchidee": "arundina graminifolia",
}


# ============================================================
# HILFSFUNKTIONEN MIT SICHERER ZUORDNUNG
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
    
    clean_name = label.replace("_", " ").capitalize()
    return {
        "name": clean_name,
        "description": f"Die Pflanze '{clean_name}' wurde durch das neuronale Netz identifiziert. Sie gehört zu den spezialisierten Taxa dieses Modells und besitzt artspezifische Anpassungen an ihr natürliches Ökosystem bezüglich Wuchs und Blattwerk.",
        "care": "• Standort: Ein heller, ausgewogener Platz ohne direkte, aggressive Mittagssonne.\n• Gießen: Regelmäßig kontrollieren und mäßig gießen; Staunässe im Topf unbedingt vermeiden.\n• Pflege: Welke Blätter gelegentlich entfernen und bei Bedarf während der Wachstumsphase schonend düngen.",
    }


def get_toxicity_information(label):
    result = find_database_entry(label, TOXICITY_INFO)
    if result:
        return {"known": True, "details": result["details"], "audience": result["audience"]}
    return {
        "known": False, 
        "details": "Für diese spezifische Pflanzenart sind keine dokumentierten Giftstoffe in unserer primären Datenbank hinterlegt. Generell ist beim Verzehr unbekannter Pflanzen stets Vorsicht geboten.", 
        "audience": "Nicht überprüft / Unbekannt"
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
<div class="top-header-row">
    <svg class="logo-svg" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 52C20 40 34 26 52 12" stroke="#2d7a36" stroke-width="4" stroke-linecap="round"/>
        <path d="M28 36C18 34 10 24 12 12C24 10 34 18 36 28" fill="#469649"/>
        <path d="M36 28C46 26 54 16 52 4C40 2 30 10 28 20" fill="#2d7a36"/>
        <path d="M22 42C24 52 34 58 46 56C48 44 40 36 30 34" fill="#5aa856"/>
        <path d="M38 26C48 28 56 38 54 50C42 52 32 44 30 34" fill="#35853d"/>
    </svg>
    <h1 class="main-title">PlantScan</h1>
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
<div class="card-container" style="margin-bottom: 18px;">
    <div class="card-header">
        <div class="card-icon-box">📷</div>
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
            <div class="card-icon-box" style="background: transparent; font-size: 26px;">🌱</div>
            <h3 class="card-title">So funktioniert PlantScan</h3>
        </div>
        <div style="margin-top: 12px; margin-left: 14px;">
            <div class="step-row">
                <div class="step-badge">1</div>
                <div>Lade ein Foto deiner Pflanze hoch.</div>
            </div>
            <div class="step-row">
                <div class="step-badge">2</div>
                <div>Wähle zwischen Pflege und Giftcheck.</div>
            </div>
            <div class="step-row">
                <div class="step-badge">3</div>
                <div>Entdecke spezifische Informationen.</div>
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
            # AUSWAHL-BUTTONS (Der Nutzer wählt VORHER)
            # ----------------------------------------
            st.markdown("### 📌 Was möchtest du erfahren?")
            st.markdown("Bitte wähle eine Option aus, um die entsprechenden Informationen anzuzeigen:")
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("🌱 Pflanzenpflege anzeigen", use_container_width=True):
                    st.session_state.view = "care"
            with col_b2:
                if st.button("⚠️ Giftigkeitscheck anzeigen", use_container_width=True):
                    st.session_state.view = "toxicity"

            st.markdown("<br>", unsafe_allow_html=True)

            # ----------------------------------------
            # ANSICHT: PFLEGE
            # ----------------------------------------
            if st.session_state.view == "care":
                st.markdown('<div class="section-heading">🌱 Pflanzenpflege & Spezifische Informationen</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="card-container">
                    <p class="muted">Erkannte Pflanzenart</p>
                    <div class="result-name">{html.escape(plant["name"])}</div>
                    <p style="margin-top: 15px; line-height: 1.6;"><b>Ausführliche Beschreibung:</b><br>{html.escape(plant["description"])}</p>
                    <hr>
                    <h3 style="color: #163b22; margin-bottom: 8px;">💧 Detaillierte Pflegehinweise</h3>
                    <p style="line-height: 1.6; white-space: pre-line;">{html.escape(plant["care"])}</p>
                </div>
                """, unsafe_allow_html=True)

            # ----------------------------------------
            # ANSICHT: GIFTCHECK
            # ----------------------------------------
            elif st.session_state.view == "toxicity":
                st.markdown('<div class="section-heading">⚠️ Giftigkeitscheck & Risiken</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="card-container">
                    <p class="muted">Erkannte Pflanzenart</p>
                    <div class="result-name">{html.escape(plant["name"])}</div>
                </div>
                """, unsafe_allow_html=True)

                if toxicity["known"]:
                    st.error("Gesundheitliche Hinweise:")
                    st.write(toxicity["details"])
                else:
                    st.warning("Keine spezifischen Giftstoffe hinterlegt.")
                    st.write(toxicity["details"])
                
                st.caption("Betroffene Gruppe laut Datenbank: " + toxicity["audience"])
                st.info("Hinweis: Die KI-Bestimmung kann fehlerhaft sein und ersetzt keine professionelle botanische Giftprüfung.")

    except Exception as error:
        st.error("Das Bild konnte nicht verarbeitet werden.")
        st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown("""
<div style="text-align:center; padding:20px; color:#3d5a42;">
    <h3 style="color:#163b22; margin-bottom:5px;">🌿 PlantScan</h3>
    Dein digitaler Pflanzenhelfer<br>
    <small>Ein KI-Schulprojekt</small>
</div>
""", unsafe_allow_html=True)
