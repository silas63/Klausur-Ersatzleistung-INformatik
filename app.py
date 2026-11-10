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
# DESIGN & CSS (Exakt wie auf deinem Design-Bild)
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
        max-width: 950px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Top-Bar mit Titel, Icons rechts und Untertitel */
    .top-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        margin-bottom: 4px;
    }

    .title-left {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .logo-badge {
        background: #287344;
        width: 54px;
        height: 54px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        box-shadow: 0 6px 16px rgba(40,115,68,0.25);
    }

    .main-title {
        color: #174b2e;
        font-size: 46px;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin: 0;
        line-height: 1;
    }

    .top-right-icons {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .icon-pill {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid #dcebd8;
        padding: 8px 14px;
        border-radius: 30px;
        display: flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 4px 12px rgba(30,80,40,0.06);
    }

    .profile-circle {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid #dcebd8;
        width: 42px;
        height: 42px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        box-shadow: 0 4px 12px rgba(30,80,40,0.06);
    }

    .subtitle {
        color: #5c7960;
        font-size: 16px;
        margin-bottom: 22px;
        margin-top: 2px;
    }

    /* Hero Banner im exakten Entwurfs-Look */
    .hero {
        background: linear-gradient(rgba(23, 75, 46, 0.75), rgba(23, 75, 46, 0.75)), 
                    url('https://images.unsplash.com/photo-1530595467537-0b5996c41f2d?q=80&w=1200&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        padding: 36px 40px;
        border-radius: 24px;
        color: white;
        margin: 10px 0 24px 0;
        box-shadow: 0 10px 30px rgba(23,75,46,0.15);
    }

    .hero h2 {
        color: white !important;
        font-size: 26px;
        margin-top: 0;
        margin-bottom: 8px;
        font-weight: 700;
    }

    .hero p {
        color: #e4f2e6;
        font-size: 15px;
        line-height: 1.5;
        margin: 0;
        max-width: 680px;
    }

    /* Karten-Container im Bild-Stil */
    .card-container {
        background: #fbfdf9;
        border: 1px solid #dcebd8;
        border-radius: 24px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(30, 80, 40, 0.05);
    }

    .card-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 6px;
    }

    .card-icon-box {
        background: #e6f2e0;
        width: 44px;
        height: 44px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
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
        margin-bottom: 18px;
        margin-left: 58px;
    }

    /* Nummerierte Schritte für "So funktioniert PlantScan" */
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

    /* Streamlit File Uploader optisch anpassen (Gestrichelte Box wie im Bild) */
    div[data-testid="stFileUploader"] {
        background: #f5f9f2;
        border: 2px dashed #b8d4b0;
        border-radius: 16px;
        padding: 10px;
        margin-left: 58px;
        width: calc(100% - 58px);
    }
    div[data-testid="stFileUploader"] section {
        padding: 12px;
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
# PFLANZEN-DATENBANK
# ============================================================

PLANT_INFO = {
    "monstera deliciosa": {
        "name": "Monstera (Fensterblatt)",
        "description": "Die Monstera deliciosa stammt aus den tropischen Regenwäldern Mittel- und Südamerikas. Sie ist weltberühmt für ihre großen, dunkelgrünen und tief eingeschnittenen Blätter, die in freier Natur dazu dienen, Sonnenlicht und Regenwasser in tiefere Waldbereiche durchzulassen. Als Zimmerpflanze wächst sie stark kletternd und benötigt eine Rankhilfe oder viel Platz.",
        "care": "• Standort: Helles bis halbschattiges Plätzchen, aber ohne direkte grelle Mittagssonne, die zu Verbrennungen führen kann.\n• Gießen: Die obere Erdschicht (ca. 2-3 cm) vor jedem Gießen antrocknen lassen. Wöchentlich mäßig gießen und Staunässe im Übertopf unbedingt vermeiden, da dies zu Wurzelfäule führt.\n• Temperatur & Luft: Bevorzugt normale Zimmertemperaturen (18–25 °C) und dankt regelmäßiges Besprühen mit kalkarmem Wasser oder das Abwischen der Blätter mit einem feuchten Tuch mit einer wunderschönen, staubfreien Glanzoptik.",
    },
    "sansevieria trifasciata": {
        "name": "Bogenhanf (Sansevieria)",
        "description": "Der Bogenhanf ist eine der absolut robustesten und anspruchslosesten Zimmerpflanzen überhaupt. Ursprünglich in den trockenen Regionen Westafrikas beheimatet, speichert er Wasser in seinen dickfleischigen, aufrecht stehenden Blättern. Zudem gilt er als hervorragender Luftreiniger, der Schadstoffe aus der Raumluft filtern kann.",
        "care": "• Standort: Kommt mit fast jedem Standort klar – von sonnig bis schattig. Ein heller Platz fördert jedoch die intensive Blattzeichnung.\n• Gießen: Äußerst sparsam! Lieber zu wenig als zu viel gießen. Die Erde vor dem nächsten Wässern komplett austrocknen lassen.\n• Besonderheit: Perfekt für Anfänger, da sie wochenlange Trockenheit problemlos verzeiht.",
    },
    "chlorophytum comosum": {
        "name": "Grünlilie",
        "description": "Die Grünlilie ist ein echter Klassiker in deutschen Haushalten. Sie zeichnet sich durch schmale, bogenförmig überhängende Blätter aus, die meist grün-weiß gestreift sind. An langen Ausläufern bildet sie kleine Tochterpflanzen (Ableger), die sich super leicht vermehren lassen.",
        "care": "• Standort: Hell bis halbschattig, direkte pralle Sonne sollte vermieden werden.\n• Gießen: Gleichmäßig feucht halten. Im Frühling und Sommer regelmäßig gießen, im Winter etwas weniger.\n• Düngung: Von März bis August alle zwei Wochen mit einem milden Flüssigdünger versorgen.",
    },
    "aloe vera": {
        "name": "Aloe vera",
        "description": "Die Echte Aloe ist eine sukkulente Heilpflanze aus der Familie der Affodillgewächse. Sie bildet dicke, fleischige Rosettenblätter, die ein durchsichtiges, feuchtigkeitsspendendes Gel enthalten, welches in der Naturheilkunde geschätzt wird.",
        "care": "• Standort: Ein vollsonniger, warmer und luftiger Platz (gerne auf der Südfensterbank) ist ideal.\n• Gießen: Sehr sparsam gießen. Das Substrat sollte absolut durchlässig sein. Staunässe führt sofort zum Absterben der Wurzeln.",
    },
    "zamioculcas zamiifolia": {
        "name": "Glücksfeder",
        "description": "Die Glücksfeder (Zamioculcas) hat glänzende, dunkelgrüne Blätter, die wie aufgereiht an kräftigen Stielen sitzen. Sie besitzt unter der Erde dicke Speicherknollen, mit denen sie Trockenperioden spielend überbrückt.",
        "care": "• Standort: Extrem flexibel. Sie toleriert dunkle Ecken im Raum, wächst an hellen Standorten jedoch etwas schneller.\n• Gießen: Erst gießen, wenn die Erde komplett trocken ist. Lieber einmal zu wenig als zu viel Wasser geben.",
    },
    "epipremnum aureum": {
        "name": "Efeutute",
        "description": "Die Efeutute ist eine wüchsige, grüne Kletter- und Ampelpflanze mit herzförmigen Blättern, die oft goldgelb oder weiß panaschiert sind. Sie rankt Meter um Meter an Wänden und Regalen entlang.",
        "care": "• Standort: Hell bis halbschattiger Standort ohne direkte Sonne.\n• Gießen: Mäßig feucht halten. Die Oberfläche antrocknen lassen, dann durchdringend gießen.",
    },
    "ficus elastica": {
        "name": "Gummibaum",
        "description": "Der Gummibaum imponiert durch seine großen, lederartigen, dunkelgrünen Blätter, die majestätisch nach oben streben. Bei guter Pflege kann er im Zimmer im Laufe der Jahre zu einem kleinen Baum heranwachsen.",
        "care": "• Standort: Sehr hell, gerne mit etwas morgendlicher oder abendlicher Sonne.\n• Gießen: Mäßige Wassergaben mit kalkarmem Wasser bei Zimmertemperatur.",
    },
    "ficus lyrata": {
        "name": "Geigenfeige",
        "description": "Die Geigenfeige verdankt ihren Namen den großen, geigen- beziehungsweise leierförmigen Blättern mit markanten hellerleuchteten Blattadern. Ein echter Hingucker für moderne Interieurs.",
        "care": "• Standort: Sehr hell, geschützt vor kalter Zugluft und direkter Mittagssonne.\n• Gießen: Gleichmäßig leicht feucht halten. Staunässe führt zu braunen Blatträndern.",
    },
    "spathiphyllum wallisii": {
        "name": "Einblatt",
        "description": "Das Einblatt zeichnet sich durch seine sattgrünen Blätter und die markanten weißen Hochblätter aus. Es ist eine der wenigen Zimmerpflanzen, die im Haus verlässlich weiße Blüten hervorbringen.",
        "care": "• Standort: Halbschattig bis hell, keinesfalls direkte Sonne.\n• Gießen: Hoher Wasserbedarf! Die Erde sollte stets leicht feucht gehalten werden.",
    },
    "dracaena marginata": {
        "name": "Drachenbaum",
        "description": "Der Drachenbaum erinnert optisch an eine Palme, besitzt schlanke, verholzte Stämme und schopfartig sitzende Blätter mit rötlichem Rand. Extrem pflegeleicht und langlebig.",
        "care": "• Standort: Hell bis halbschattig.\n• Gießen: Mäßig gießen. Die Erde vor dem erneuten Gießen gut antrocknen lassen.",
    },
    "crassula ovata": {
        "name": "Geldbaum (Kirchbäumchen)",
        "description": "Klassische Sukkulente mit dicken, fleischigen, ovalen Blättern. Im Feng-Shui gilt er als glückbringender Hausschmuck.",
        "care": "• Standort: Ein sonniger, warmer Platz am Fenster ist optimal.\n• Gießen: Sparsam gießen. Erst wenn die Erde komplett ausgetrocknet ist.",
    },
    "yucca elephantipes": {
        "name": "Yucca-Palme (Palmlilie)",
        "description": "Besitzt einen kräftigen Stamm und schopfartig angeordnete, lange, steif-spitze Blätter. Gedeiht auch in trockener Zimmerluft hervorragend.",
        "care": "• Standort: Maximal hell, gerne vollsonnig (Südfenster).\n• Gießen: Mäßig gießen, im Winter noch sparsamer.",
    },
    "quercus robur": {
        "name": "Stiel-Eiche",
        "description": "Einer der majestätischsten heimischen Laubbäume mit gelappten Blättern und Eicheln. Wird Hunderte Jahre alt.",
        "care": "• Standort: Ausschließlich im Freien an vollsonnigen Standorten.\n• Boden: Tiefgründige, nährstoffreiche und humose Böden.",
    },
    "fagus sylvatica": {
        "name": "Rot-Buche",
        "description": "Prägt als dominanter Laubbaum weite Teile der mitteleuropäischen Wälder mit glatter Rinde.",
        "care": "• Standort: Als Wald- und Parkbaum im Freien.\n• Boden: Frische, nährstoffreiche, kalkhaltige Böden.",
    },
    "betula pendula": {
        "name": "Sand-Birke",
        "description": "Sofort an ihrer leuchtend weißen Rinde und den elegant herabhängenden Zweigen erkennbar.",
        "care": "• Standort: Volle Sonne. Ein absoluter 'Lichtliebhaber'.\n• Boden: Gedeiht selbst auf sehr mageren Sandböden.",
    },
    "acer pseudoplatanus": {
        "name": "Berg-Ahorn",
        "description": "Stattlicher, rasch wachsender Laubbaum mit großen, fünfzipfeligen Blättern und geflügelten Früchten.",
        "care": "• Standort: Sonnig bis halbschattig in großen Gärten und Parks.",
    },
    "pinus sylvestris": {
        "name": "Wald-Kiefer (Föhre)",
        "description": "Immergrüner Nadelbaum mit rotbrauner, abblätternder Rinde im oberen Stammbereich.",
        "care": "• Standort: Volle Sonne und ein luftiger Platz im Garten.",
    },
    "picea abies": {
        "name": "Gemeine Fichte (Rotfichte)",
        "description": "Der in Deutschland am weitesten verbreitete Nadelbaum und klassischer Weihnachtsbaum.",
        "care": "• Standort: Kühle, feuchtere Standorte im Freien.",
    },
    "taraxacum officinale": {
        "name": "Wiesen-Löwenzahn",
        "description": "Wildpflanze mit gelben Blütenkörben und Pusteblumen-Samen. Wichtige Bienenweide.",
        "care": "• Standort: Sonnig bis halbschattig auf Wiesen und Rasenflächen.",
    },
    "bellis perennis": {
        "name": "Gänseblümchen (Tausendschön)",
        "description": "Zarte Wiesenblume mit weiß-rosa Blütenblättern, die sich bei Nacht oder Regen schließen.",
        "care": "• Standort: Sonnige bis halbschattige Rasenflächen.",
    },
    "urtica dioica": {
        "name": "Große Brennnessel",
        "description": "Heimische Wildpflanze mit Brennhaaren, extrem wertvoll als Raupenfutter für Schmetterlinge.",
        "care": "• Standort: Nährstoffreiche Standorte an Waldrändern.",
    },
    "rosa canina": {
        "name": "Hunds-Rose (Heckenrose)",
        "description": "Heimischer Wildstrauch mit rosa-weißen Blüten und roten Hagebutten im Herbst.",
        "care": "• Standort: Sonnig bis halbschattig, ideal für Hecken.",
    },
}


# ============================================================
# GIFTIGKEITS-DATENBANK
# ============================================================

TOXICITY_INFO = {
    "monstera deliciosa": {
        "details": "Die Monstera enthält im gesamten Pflanzengewebe Calciumoxalat-Kristalle. Bei Verzehr brennendes Gefühl und Schleimhautreizungen.",
        "audience": "Menschen und Haustiere (Katzen, Hunde)",
    },
    "sansevieria trifasciata": {
        "details": "Enthält in den Blättern Saponine. Führt bei Haustieren zu Erbrechen, Durchfall und starkem Speichelfluss.",
        "audience": "Insbesondere Haustiere (Hunde und Katzen)",
    },
    "aloe vera": {
        "details": "Die Blatthaut enthält Aloin, das abführend und reizend auf den Magen-Darm-Trakt wirkt.",
        "audience": "Menschen und Haustiere",
    },
    "zamioculcas zamiifolia": {
        "details": "Enthält Calciumoxalat-Kristalle. Bei Verzehr drohen Schwellungen im Rachenraum und Speichelfluss.",
        "audience": "Menschen und Haustiere",
    },
    "epipremnum aureum": {
        "details": "Enthält Calciumoxalat-Nadeln. Führt bei Verschlucken zu Schleimhautschwellungen.",
        "audience": "Menschen und Haustiere",
    },
    "ficus elastica": {
        "details": "Der weiße Milchsaft enthält reizende Enzyme. Bei Hautkontakt oder Verzehr drohen Entzündungen.",
        "audience": "Menschen und Haustiere",
    },
    "spathiphyllum wallisii": {
        "details": "Enthält Calciumoxalat. Bewirkt heftiges Brennen im Mund und Schluckbeschwerden.",
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
# HILFSFUNKTIONEN
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
    
    clean_fallback = label.replace("_", " ").capitalize()
    return {
        "name": clean_fallback,
        "description": f"Das KI-Modell hat diese Pflanze als '{clean_fallback}' klassifiziert. In unserer erweiterten Datenbank sind zu dieser Art aktuell keine spezifischen Detailtexte hinterlegt.",
        "care": "Achten Sie auf einen ausgewogenen Standort mit ausreichend Licht, vermeiden Sie permanente Staunässe im Wurzelbereich und passen Sie die Wassergaben an.",
    }


def get_toxicity_information(label):
    result = find_database_entry(label, TOXICITY_INFO)
    if result:
        return {"known": True, "details": result["details"], "audience": result["audience"]}
    return {
        "known": False, 
        "details": "Für diese Pflanzenart liegen keine bekannten Giftstoffe in unserer Datenbank vor. Vorsicht ist dennoch geboten.", 
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
# HEADER BEREICH (EXAKT WIE AUF DEM BILD)
# ============================================================

st.markdown("""
<div class="top-header-row">
    <div class="title-left">
        <div class="logo-badge">🌿</div>
        <h1 class="main-title">PlantScan</h1>
    </div>
    <div class="top-right-icons">
        <div class="icon-pill">
            <span>☀️</span>
            <div style="width: 24px; height: 12px; background: #c8dec4; border-radius: 10px; position: relative;">
                <div style="width: 10px; height: 10px; background: white; border-radius: 50%; position: absolute; top: 1px; left: 1px;"></div>
            </div>
        </div>
        <div class="profile-circle">👤</div>
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
# UPLOAD BEREICH (IM EXAKT GLEICHEN DESIGN-KARTEN-STIL)
# ============================================================

st.markdown("""
<div class="card-container" style="margin-bottom: 15px;">
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
    # So funktioniert PlantScan Karte (wie im Bild)
    st.markdown("""
    <div class="card-container">
        <div class="card-header">
            <div class="card-icon-box">🌱</div>
            <h3 class="card-title">So funktioniert PlantScan</h3>
        </div>
        <div style="margin-top: 15px; margin-left: 58px;">
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
                    <p style="margin-top: 15px; line-height: 1.6;"><b>Ausführliche Beschreibung:</b><br>{html.escape(plant["description"])}</p>
                    <hr>
                    <h3 style="color: #174b2e; margin-bottom: 8px;">💧 Detaillierte Pflegehinweise</h3>
                    <p style="line-height: 1.6; white-space: pre-line;">{html.escape(plant["care"])}</p>
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
                    <p style="margin-top: 10px;"><b>Wissenschaftlicher Modell-Name:</b> {html.escape(label)}</p>
                    <p style="margin-top: 10px; line-height: 1.6;"><b>Über die Pflanze:</b><br>{html.escape(plant["description"])}</p>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Modell-Konfidenz", f"{confidence:.2f} %")
                with col2:
                    st.metric("Weitere Ergebnisse", len(results))

                if confidence < 50:
                    st.warning("Die Erkennung ist unsicher. Bitte überprüfe die Pflanzenbestimmung sorgfältig.")
                else:
                    st.info("Das Modell zeigt eine vergleichsweise hohe Konfidenz. Eine hundertprozentige Bestimmungssicherheit kann dennoch nicht garantiert werden.")

                st.subheader("⚠️ Giftigkeitscheck & Risiken")
                if toxicity["known"]:
                    st.error("Für diese Pflanzenart sind gesundheitliche Risiken bekannt.")
                else:
                    st.warning("Giftigkeit nicht explizit überprüft – unbekannt bedeutet keinesfalls harmlos.")
                
                st.write(toxicity["details"])
                st.caption("Betroffene Gruppe laut Datenbank: " + toxicity["audience"])

                st.info("Hinweis: Die automatisierte Pflanzenbestimmung mittels KI kann fehlerhaft sein. Dieser Check ersetzt keinesfalls eine professionelle botanische Giftigkeitsprüfung.")

                st.subheader("📊 Weitere KI-Ergebnisse")
                for result in results:
                    percentage = result["score"] * 100
                    st.write(f"**{result['name']}** – {percentage:.2f} %")
                    st.progress(float(result["score"]))

                st.caption("Die Prozentwerte spiegeln die relative Modell-Wahrscheinlichkeit wider.")

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
