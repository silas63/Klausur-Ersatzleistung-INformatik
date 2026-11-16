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
# DESIGN & CSS (Clean & Perfekter weicher Natur-Look)
# ============================================================

st.markdown("""
<style>
    /* Hintergrund mit perfektem, weichem Übergang */
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

    /* Top-Header mit Logo und Titel */
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

    /* Hero Banner */
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

    /* Clean Milchglas-Karten */
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

    /* Nummerierte Schritte */
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
# PFLANZEN-DATENBANK MIT AUSFÜHRLICHEN INFOTEXTEN
# ============================================================

PLANT_INFO = {
    "monstera deliciosa": {
        "name": "Monstera (Fensterblatt)",
        "description": "Die Monstera deliciosa stammt aus den tropischen Regenwäldern Mittel- und Südamerikas. Sie ist weltberühmt für ihre großen, dunkelgrünen und tief eingeschnittenen Blätter, die in freier Natur dazu dienen, Sonnenlicht und Regenwasser in tiefere Waldbereiche durchzulassen. Als Zimmerpflanze wächst sie stark kletternd, entwickelt faszinierende Luftwurzeln und benötigt eine stabile Rankhilfe oder ausreichend Raum zur Entfaltung.",
        "care": "• Standort: Helles bis halbschattiges Plätzchen, aber ohne direkte grelle Mittagssonne, die zu unschönen Verbrennungen auf den Blättern führen kann.\n• Gießen: Die obere Erdschicht (ca. 2-3 cm) vor jedem Gießen gut antrocknen lassen. Wöchentlich mäßig gießen und Staunässe im Übertopf unbedingt vermeiden, da dies zu Wurzelfäule führt.\n• Temperatur & Luft: Bevorzugt normale Zimmertemperaturen (18–25 °C) und dankt regelmäßiges Besprühen mit kalkarmem Wasser oder das Abwischen der Blätter mit einem feuchten Tuch mit einer wunderschönen, staubfreien Glanzoptik.",
    },
    "sansevieria trifasciata": {
        "name": "Bogenhanf (Sansevieria)",
        "description": "Der Bogenhanf ist eine der absolut robustesten und anspruchslosesten Zimmerpflanzen überhaupt. Ursprünglich in den trockenen, nährstoffarmen Regionen Westafrikas beheimatet, speichert er große Mengen Wasser in seinen dickfleischigen, aufrecht stehenden Blättern. Zudem gilt er als hervorragender natürlicher Luftreiniger, der Schadstoffe effektiv aus der Raumluft filtern kann und auch bei vernachlässigter Pflege treu durchhält.",
        "care": "• Standort: Kommt mit fast jedem Standort klar – von sonnig bis schattig. Ein heller Platz fördert jedoch die intensive, kontrastreiche Blattzeichnung.\n• Gießen: Äußerst sparsam! Lieber zu wenig als zu viel gießen. Die Erde vor dem nächsten Wässern komplett austrocknen lassen.\n• Besonderheit: Perfekt für Anfänger geeignet, da sie wochenlange Trockenheit sowie trockene Heizungsluft problemlos und ohne Schaden verzeiht.",
    },
    "chlorophytum comosum": {
        "name": "Grünlilie",
        "description": "Die Grünlilie ist ein echter, unkomplizierter Klassiker in deutschen Haushalten und Büros. Sie zeichnet sich durch schmale, bogenförmig überhängende Blätter aus, die meist markant grün-weiß gestreift sind. An langen, bogigen Ausläufern bildet sie im Laufe der Zeit kleine Tochterpflanzen aus, die sich kinderleicht abtrennen und im Wasserglas oder direkt in Erde vermehren lassen.",
        "care": "• Standort: Hell bis halbschattig, direkte pralle Sonne sollte im Sommer vermieden werden, um Blattschäden zu verhindern.\n• Gießen: Gleichmäßig leicht feucht halten. Im Frühling und Sommer regelmäßig gießen, im Winter etwas weniger Wasser verabreichen.\n• Düngung: Von März bis August alle zwei Wochen mit einem milden Flüssigdünger versorgen, um das kräftige Wachstum optimal zu unterstützen.",
    },
    "aloe vera": {
        "name": "Aloe vera",
        "description": "Die Echte Aloe ist eine faszinierende sukkulente Heilpflanze aus der Familie der Affodillgewächse, die auf eine jahrtausendealte Geschichte als Nutz- und Kosmetikpflanze zurückblickt. Sie bildet dicke, fleischige Rosettenblätter, die ein durchsichtiges, feuchtigkeitsspendendes und kühlendes Gel enthalten, welches in der Naturheilkunde und Hautpflege hoch geschätzt wird.",
        "care": "• Standort: Ein vollsonniger, warmer und luftiger Platz (gerne direkt auf der Südfensterbank) ist ideal für ein gesundes Wachstum.\n• Gießen: Sehr sparsam gießen. Das Substrat sollte absolut durchlässig und mineralisch sein. Permanente Staunässe führt sofort zum Absterben der Wurzeln.\n• Überwinterung: Im Winter etwas kühler (ca. 10-15 °C) und fast gänzlich trocken halten.",
    },
    "zamioculcas zamiifolia": {
        "name": "Glücksfeder",
        "description": "Die Glücksfeder (Zamioculcas) besticht durch ihre glänzenden, dunkelgrünen, fleischigen Blätter, die wie aufgereiht an kräftigen, aufrechten Stielen sitzen. Sie besitzt unter der Erde dicke, knollige Wasserspeicherorgane, mit denen sie auch längere Trockenperioden und dunklere Standorte im Raum spielend und ohne Wachstumsstopp überbrückt.",
        "care": "• Standort: Extrem flexibel und anpassungsfähig. Sie toleriert selbst dunklere Ecken im Raum, wächst an hellen Standorten jedoch etwas schneller.\n• Gießen: Erst gießen, wenn die Erde im Topf komplett ausgetrocknet ist. Lieber einmal zu wenig als zu viel Wasser geben, da sie extrem empfindlich auf nasse Füße reagiert.",
    },
    "epipremnum aureum": {
        "name": "Efeutute",
        "description": "Die Efeutute ist eine extrem wüchsige, grüne Kletter- und Ampelpflanze mit dekorativen, herzförmigen Blättern, die oft goldgelb oder weiß panaschiert sind. Sie rankt in kürzester Zeit Meter um Meter an Wänden, Rankhilfen oder Regalen entlang und bringt tropisches Flair in jeden Wohnraum.",
        "care": "• Standort: Hell bis halbschattiger Standort ohne direkte, sengende Mittagssonne.\n• Gießen: Mäßig feucht halten. Die obere Erdoberfläche zwischen den Wassergaben gut antrocknen lassen, dann durchdringend wässern.\n• Rückschnitt: Kann bei zu starkem Wuchs problemlos jederzeit zurückgeschnitten werden.",
    },
    "ficus elastica": {
        "name": "Gummibaum",
        "description": "Der Gummibaum imponiert durch seine großen, festen, lederartigen und dunkelgrünen Blätter, die majestätisch nach oben streben. Bei guter Pflege und ausreichend Licht kann er im Zimmer im Laufe der Jahre zu einem stattlichen, kleinen Baum heranwachsen, der jedem Raum eine elegante Note verleiht.",
        "care": "• Standort: Sehr hell, gerne mit etwas sanfter morgendlicher oder abendlicher Sonne für kräftige Farben.\n• Gießen: Mäßige Wassergaben mit kalkarmem Wasser bei Zimmertemperatur. Staunässe im Übertopf unbedingt vermeiden.\n• Pflege: Die großen Blätter regelmäßig mit einem feuchten Tuch vom Haustaub befreien, um die Photosynthese zu fördern.",
    },
    "ficus lyrata": {
        "name": "Geigenfeige",
        "description": "Die Geigenfeige verdankt ihren Namen den charakteristischen, großen, geigen- beziehungsweise leierförmigen Blättern mit markanten, hell hervortretenden Blattadern. Sie ist ein absoluter architektonischer Hingucker für moderne Interieurs und anspruchsvolle Pflanzenliebhaber.",
        "care": "• Standort: Sehr hell, aber geschützt vor kalter Zugluft und direkter greller Mittagssonne.\n• Gießen: Gleichmäßig leicht feucht halten. Unregelmäßiges Gießen oder Staunässe führt schnell zu braunen Blatträndern.\n• Luftfeuchtigkeit: Schätzt eine erhöhte Luftfeuchtigkeit und gelegentliches Besprühen.",
    },
    "spathiphyllum wallisii": {
        "name": "Einblatt",
        "description": "Das Einblatt zeichnet sich durch seine sattgrünen, glänzenden Blätter und die markanten, reinweißen Hochblätter (Spathen) aus. Es gehört zu den wenigen beliebten Zimmerpflanzen, die bei guter Pflege im Haus verlässlich und langanhaltend weiße Blüten hervorbringen.",
        "care": "• Standort: Halbschattig bis hell, keinesfalls direkter Sonneneinstrahlung aussetzen.\n• Gießen: Hoher Wasserbedarf! Die Erde sollte stets leicht feucht gehalten werden. Trocknet es zu stark aus, lässt die Pflanze sofort die Blätter hängen.",
    },
    "dracaena marginata": {
        "name": "Drachenbaum",
        "description": "Der Drachenbaum erinnert optisch an eine kleine Palme, besitzt schlanke, verholzte Stämme und schopfartig am Ende sitzende, schmale Blätter mit einem charakteristischen rötlichen Rand. Er ist extrem langlebig und passt perfekt in moderne Wohn- und Büroumgebungen.",
        "care": "• Standort: Hell bis halbschattig, ohne direkte pralle Sonne.\n• Gießen: Mäßig gießen. Die Erde vor dem erneuten Wässern gut antrocknen lassen.\n• Besonderheit: Sehr unempfindlich gegenüber trockener Raumluft.",
    },
    "crassula ovata": {
        "name": "Geldbaum (Kirchbäumchen)",
        "description": "Der Geldbaum ist eine klassische, sukkulente Pflanze mit dicken, fleischigen, ovalen Blättern und einem dicken, baumartig verholzenden Stamm. Im asiatischen Raum und nach Feng-Shui gilt er als traditioneller, glückbringender Hausschmuck, der Wohlstand anziehen soll.",
        "care": "• Standort: Ein vollsonniger, warmer und heller Platz direkt am Fenster ist optimal.\n• Gießen: Sparsam gießen. Erst dann wieder wässern, wenn die Erde komplett durchgetrocknet ist.",
    },
    "yucca elephantipes": {
        "name": "Yucca-Palme (Palmlilie)",
        "description": "Die Yucca-Palme besitzt einen krassen, dicken Stamm und schopfartig angeordnete, lange, steif-spitze Blätter. Sie ist unglaublich robust, wüchsig und gedeiht auch in trockener Zimmerluft absolut hervorragend.",
        "care": "• Standort: Maximal hell, gerne vollsonnig (idealerweise Südfenster).\n• Gießen: Mäßig gießen, im Winter während der Ruhephase noch deutlich sparsamer.",
    },
    "quercus robur": {
        "name": "Stiel-Eiche",
        "description": "Die Stiel-Eiche ist einer der majestätischsten und langlebigsten heimischen Laubbäume mit tief gelappten Blättern und typischen Eicheln. Sie prägt ganze Landschaften und kann Hunderte von Jahren alt werden.",
        "care": "• Standort: Ausschließlich im Freien an vollsonnigen, offenen Standorten.\n• Boden: Tiefgründige, nährstoffreiche und humusreiche Böden.",
    },
    "fagus sylvatica": {
        "name": "Rot-Buche",
        "description": "Die Rot-Buche prägt als dominanter heimischer Laubbaum weite Teile derteleuropäischen Wälder mit glatter, silbergrauer Rinde und markant gezähnten Blättern.",
        "care": "• Standort: Als großer Wald- und Parkbaum im Freien.\n• Boden: Frische, nährstoffreiche und kalkhaltige Böden.",
    },
    "betula pendula": {
        "name": "Sand-Birke",
        "description": "Die Sand-Birke ist sofort an ihrer leuchtend weißen, markanten Rinde und den elegant herabhängenden Zweigen erkennbar. Ein anspruchsloser Lichtliebhaber.",
        "care": "• Standort: Volle Sonne. Benötigt viel Licht.\n• Boden: Gedeiht selbst auf sehr mageren Sandböden.",
    },
    "acer pseudoplatanus": {
        "name": "Berg-Ahorn",
        "description": "Ein stattlicher, rasch wachsender Laubbaum mit großen, fünfzipfeligen Blättern und typischen geflügelten Früchten (Nasenwickler).",
        "care": "• Standort: Sonnig bis halbschattig in großen Gärten, Wäldern oder Parks.",
    },
    "pinus sylvestris": {
        "name": "Wald-Kiefer (Föhre)",
        "description": "Ein immergrüner Nadelbaum mit markanter, rotbrauner und im oberen Stammbereich abblätternder Rinde sowie graugrünen Nadeln.",
        "care": "• Standort: Volle Sonne und ein luftiger, freier Platz.",
    },
    "picea abies": {
        "name": "Gemeine Fichte (Rotfichte)",
        "description": "Der in Mitteleuropa am weitesten verbreitete Nadelbaum, der traditionell auch als klassischer Weihnachtsbaum geschätzt wird.",
        "care": "• Standort: Kühle, feuchtere Standorte im Freien.",
    },
    "taraxacum officinale": {
        "name": "Wiesen-Löwenzahn",
        "description": "Eine weitverbreitete Wildpflanze mit leuchtend gelben Blütenkörben und den bekannten Pusteblumen-Samen. Wichtige Nahrungsquelle für Bienen.",
        "care": "• Standort: Sonnig bis halbschattig auf Wiesen und Rasenflächen.",
    },
    "bellis perennis": {
        "name": "Gänseblümchen (Tausendschön)",
        "description": "Eine zarte, heimische Wiesenblume mit weiß-rosa Blütenblättern, die sich bei Nacht oder schlechtem Wetter schützen schließend zusammenfalten.",
        "care": "• Standort: Sonnige bis halbschattige Rasenflächen.",
    },
    "urtica dioica": {
        "name": "Große Brennnessel",
        "description": "Eine heimische Wildpflanze mit feinen Brennhaaren, die extrem wertvoll als Raupenfutter für zahlreiche bedrohte Schmetterlingsarten ist.",
        "care": "• Standort: Nährstoffreiche Standorte an Waldrändern und Hecken.",
    },
    "rosa canina": {
        "name": "Hunds-Rose (Heckenrose)",
        "description": "Ein robuster, heimischer Wildstrauch mit zarten rosa-weißen Blüten im Frühsommer und vitaminreichen roten Hagebutten im Herbst.",
        "care": "• Standort: Sonnig bis halbschattig, ideal als freiwachsende Hecke.",
    },
}


# ============================================================
# GIFTIGKEITS-DATENBANK
# ============================================================

TOXICITY_INFO = {
    "monstera deliciosa": {
        "details": "Die Monstera enthält im gesamten Pflanzengewebe Calciumoxalat-Kristalle. Bei Verzehr kommt es zu brennendem Gefühl im Mund, starkem Speichelfluss und Schleimhautreizungen.",
        "audience": "Menschen und Haustiere (Katzen, Hunde)",
    },
    "sansevieria trifasciata": {
        "details": "Enthält in den Blättern Saponine. Führt bei Haustieren beim Verzehr zu Erbrechen, Durchfall und starkem Speichelfluss.",
        "audience": "Insbesondere Haustiere (Hunde und Katzen)",
    },
    "aloe vera": {
        "details": "Die Blatthaut enthält Aloin, das stark abführend und reizend auf den Magen-Darm-Trakt wirkt.",
        "audience": "Menschen und Haustiere",
    },
    "zamioculcas zamiifolia": {
        "details": "Enthält Calciumoxalat-Kristalle. Bei versehentlichem Verzehr drohen Schwellungen im Rachenraum und starker Speichelfluss.",
        "audience": "Menschen und Haustiere",
    },
    "epipremnum aureum": {
        "details": "Enthält Calciumoxalat-Nadeln. Führt bei Verschlucken zu sofortigen Schleimhautschwellungen und Reizungen.",
        "audience": "Menschen und Haustiere",
    },
    "ficus elastica": {
        "details": "Der weiße Milchsaft enthält reizende Enzyme. Bei Hautkontakt oder Verzehr drohen schmerzhafte Entzündungen.",
        "audience": "Menschen und Haustiere",
    },
    "spathiphyllum wallisii": {
        "details": "Enthält Calciumoxalat. Bewirkt heftiges Brennen im Mund, Magenbeschwerden und Schluckbeschwerden.",
        "audience": "Menschen und Haustiere",
    },
    "urtica dioica": {
        "details": "Die Brennhaare enthalten Histamin und Ameisensäure, was bei Hautkontakt zu schmerzhaftem brennenden Ausschlag führt.",
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
                <div>Entdecke ausführliche Informationen.</div>
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
            # AUSWAHL-BUTTONS (Der Nutzer muss VORHER wählen)
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
            # ANSICHT: PFLEGE (Wird nur angezeigt, wenn ausgewählt)
            # ----------------------------------------
            if st.session_state.view == "care":
                st.markdown('<div class="section-heading">🌱 Pflanzenpflege & Ausführliche Informationen</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="card-container">
                    <p class="muted">Erkannte Pflanzenart (Deutscher Name)</p>
                    <div class="result-name">{html.escape(plant["name"])}</div>
                    <p style="margin-top: 15px; line-height: 1.6;"><b>Ausführliche Beschreibung:</b><br>{html.escape(plant["description"])}</p>
                    <hr>
                    <h3 style="color: #163b22; margin-bottom: 8px;">💧 Detaillierte Pflegehinweise</h3>
                    <p style="line-height: 1.6; white-space: pre-line;">{html.escape(plant["care"])}</p>
                </div>
                """, unsafe_allow_html=True)

            # ----------------------------------------
            # ANSICHT: GIFTCHECK (Wird nur angezeigt, wenn ausgewählt)
            # ----------------------------------------
            elif st.session_state.view == "toxicity":
                st.markdown('<div class="section-heading">⚠️ Giftigkeitscheck & Risiken</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="card-container">
                    <p class="muted">Erkannte Pflanzenart (Deutscher Name)</p>
                    <div class="result-name">{html.escape(plant["name"])}</div>
                </div>
                """, unsafe_allow_html=True)

                if toxicity["known"]:
                    st.error("Für diese Pflanzenart sind gesundheitliche Risiken bekannt:")
                    st.write(toxicity["details"])
                else:
                    st.warning("Für diese Pflanzenart sind aktuell keine Giftstoffe in unserer Datenbank hinterlegt. Vorsicht ist dennoch geboten.")
                    st.write(toxicity["details"])
                
                st.caption("Betroffene Gruppe laut Datenbank: " + toxicity["audience"])
                st.info("Hinweis: Die automatisierte Pflanzenbestimmung mittels KI kann fehlerhaft sein. Dieser Check ersetzt keinesfalls eine professionelle botanische Giftigkeitsprüfung.")

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
