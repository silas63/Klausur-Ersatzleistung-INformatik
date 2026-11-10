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
# AUSFÜHRLICHE PFLANZEN-DATENBANK (Extrem lange & detaillierte Infotexte)
# ============================================================

PLANT_INFO = {
    "monstera deliciosa": {
        "name": "Monstera (Fensterblatt)",
        "description": "Die Monstera deliciosa stammt aus den tropischen Regenwäldern Mittel- und Südamerikas. Sie ist weltberühmt für ihre großen, dunkelgrünen und tief eingeschnittenen Blätter, die in freier Natur dazu dienen, Sonnenlicht und Regenwasser in tiefere Waldbereiche durchzulassen. Als Zimmerpflanze wächst sie stark kletternd und benötigt eine Rankhilfe oder viel Platz.",
        "care": "• Standort: Helles bis halbschattiges Plätzchen, aber ohne direkte grelle Mittagssonne, die zu Verbrennungen führen kann.\n• Gießen: Die obere Erdschicht (ca. 2-3 cm) vor jedem Gießen antrocknen lassen. Wöchentlich mäßig gießen und Staunässe im Übertopf unbedingt vermeiden, da dies zu Wurzelfäule führt.\n• Temperatur & Luft: Bevorzugt normale Zimmertemperaturen (18–25 °C) und dankt regelmäßiges Besprühen mit kalkarmem Wasser oder das Abwischen der Blätter mit einem feuchten Tuch mit einer wunderschönen, staubfreien Glanzoptik.",
    },
    "sansevieria trifasciata": {
        "name": "Bogenhanf (Sansevieria)",
        "description": "Der Bogenhanf ist eine der absolut robustesten und anspruchslosesten Zimmer überhaupt. Ursprünglich in den trockenen Regionen Westafrikas beheimatet, speichert er Wasser in seinen dickfleischigen, aufrecht stehenden Blättern. Zudem gilt er als hervorragender Luftreiniger, der Schadstoffe aus der Raumluft filtern kann und auch nachts Sauerstoff abgibt.",
        "care": "• Standort: Kommt mit fast jedem Standort klar – von sonnig bis schattig. Ein heller Platz fördert jedoch die intensive Blattzeichnung.\n• Gießen: Äußerst sparsam! Lieber zu wenig als zu viel gießen. Die Erde vor dem nächsten Wässern komplett austrocknen lassen.\n• Besonderheit: Perfekt für Anfänger oder Menschen mit einem 'weniger grünen Daumen', da sie wochenlange Trockenheit problemlos verzeiht.",
    },
    "chlorophytum comosum": {
        "name": "Grünlilie",
        "description": "Die Grünlilie ist ein echter Klassiker in deutschen Haushalten. Sie zeichnet sich durch schmale, bogenförmig überhängende Blätter aus, die meist grün-weiß gestreift sind. An langen Ausläufern bildet sie kleine Tochterpflanzen (Ableger), die sich super leicht vermehren lassen. Sie ist ungiftig und absolut unkompliziert.",
        "care": "• Standort: Hell bis halbschattig, direkte pralle Sonne sollte jedoch vermieden werden.\n• Gießen: Gleichmäßig feucht halten. Im Frühling und Sommer regelmäßig gießen, im Winter etwas weniger.\n• Düngung: Von März bis August alle zwei Wochen mit einem milden Flüssigdünger versorgen, um das Wachstum und die Blattfarbe zu unterstützen.",
    },
    "aloe vera": {
        "name": "Aloe vera",
        "description": "Die Echte Aloe ist eine sukkulente Heilpflanze aus der Familie der Affodillgewächse. Sie bildet dicke, fleischige Rosettenblätter, die ein durchsichtiges, feuchtigkeitsspendendes Gel enthalten. Dieses Gel wird seit Jahrtausenden in der Naturheilkunde und Kosmetik zur Hautpflege bei Sonnenbrand oder leichten Verbrennungen eingesetzt.",
        "care": "• Standort: Ein vollsonniger, warmer und luftiger Platz (gerne auf der Südfensterbank) ist ideal.\n• Gießen: Sehr sparsam gießen. Das Substrat sollte absolut durchlässig (Kakteenerde) sein. Staunässe führt sofort zum Absterben der Wurzeln.\n• Winterruhe: Im Winter kühler (ca. 10–15 °C) stellen und extrem wenig gießen.",
    },
    "zamioculcas zamiifolia": {
        "name": "Glücksfeder",
        "description": "Die Glücksfeder (Zamioculcas) hat glänzende, dunkelgrüne Blätter, die wie aufgereiht an kräftigen Stielen sitzen. Sie besitzt unter der Erde dicke Speicherknollen, mit denen sie Trockenperioden spielend überbrückt. Sie wächst langsam, bleibt kompakt und sieht durch ihren glänzenden Wuchs fast künstlich und extrem edel aus.",
        "care": "• Standort: Extrem flexibel. Sie toleriert dunkle Ecken im Raum, wächst an hellen Standorten jedoch etwas schneller.\n• Gießen: Erst gießen, wenn die Erde komplett trocken ist. Lieber einmal zu wenig als zu viel Wasser geben.\n• Pflege-Tipp: Gelegentlich die Blätter mit einem feuchten Tuch entstauben, damit sie ihren edlen Glanz behalten.",
    },
    "epipremnum aureum": {
        "name": "Efeutute",
        "description": "Die Efeutute ist eine wüchsige, grüne Kletter- und Ampelpflanze mit herzförmigen Blättern, die oft goldgelb oder weiß panaschiert sind. Sie rankt Meter um Meter an Wänden, Regalen oder Rankhilfen entlang und gehört zu den besten Luftverbesserern unter den Zimmerpflanzen.",
        "care": "• Standort: Hell bis halbschattig. Je heller der Standort (ohne direkte Sonne), desto ausgeprägter ist die helle Blattmusterung.\n• Gießen: Mäßig feucht halten. Die Oberfläche antrocknen lassen, dann durchdringend gießen.\n• Vermehrung: Lässt sich extrem leicht über Stecklinge vermehren, die man einfach in ein Glas Wasser stellt, bis sie Wurzeln bilden.",
    },
    "ficus elastica": {
        "name": "Gummibaum",
        "description": "Der Gummibaum imponiert durch seine großen, derartigen, dunkelgrünen Blätter, die majestätisch nach oben streben. Früher ein absoluter Pflichtbestandteil in jedem deutschen Wohnzimmer, erlebt er aktuell ein großes Comeback. Er kann bei guter Pflege im Zimmer im Laufe der Jahre zu einem echten kleinen Baum heranwachsen.",
        "care": "• Standort: Sehr hell, gerne mit etwas morgendlicher oder abendlicher Sonne, aber ohne brennende Hitze.\n• Gießen: Mäßige Wassergaben mit kalkarmem Wasser bei Zimmertemperatur. Keine Staunässe.\n• Besonderheit: Bei Verletzungen tritt ein milchiger Saft aus, der bei empfindlichen Personen leichte Hautreizungen auslösen kann.",
    },
    "ficus lyrata": {
        "name": "Geigenfeige",
        "description": "Die Geigenfeige verdankt ihren Namen den großen, geigen- beziehungsweise leierförmigen Blättern mit markanten hellerleuchteten Blattadern. Sie ist ein echter Hingucker für moderne Interieurs, verlangt allerdings einen standorttreuen Platz, da sie auf Standortwechsel oft mit leichtem Blattabwurf reagieren kann.",
        "care": "• Standort: Sehr hell, geschützt vor kalter Zugluft und direkter Mittagssonne.\n• Gießen: Gleichmäßig leicht feucht halten. Staunässe führt hier extrem schnell zu braunen Blatträndern.\n• Luftfeuchtigkeit: Da sie tropisch veranlagt ist, liebt sie es, ab und zu mit kalkfreiem Wasser besprüht zu werden.",
    },
    "spathiphyllum wallisii": {
        "name": "Einblatt",
        "description": "Das Einblatt zeichnet sich durch seine sattgrünen, lanzettlichen Blätter und die markanten weißen Hochblätter (Spathen) aus, die wie elegante Segel über der Pflanze thronen. Es ist eine der wenigen Zimmerpflanzen, die bei guter Pflege im Haus verlässlich und über Wochen hinweg wunderschöne weiße Blüten hervorbringen.",
        "care": "• Standort: Halbschattig bis hell, aber keinesfalls direkte Sonne. Auch für schattigere Bäder oder Flure geeignet.\n• Gießen: Hoher Wasserbedarf! Die Erde sollte stets leicht feucht gehalten werden. Trocknet sie komplett aus, lässt das Einblatt sofort die Blätter hängen (erholt sich nach dem Gießen aber meist schnell wieder).\n• Luftfeuchtigkeit: Hohe Luftfeuchtigkeit dankt sie mit prächtigem Wachstum.",
    },
    "dracaena marginata": {
        "name": "Drachenbaum",
        "description": "Der Drachenbaum erinnert optisch an eine Palme, gehört botanisch jedoch zu den Spargelgewächsen (Asparagaceae). Er besitzt schlanke, verholzte Stämme und schopfartig am Ende sitzende, schmal-lanzettliche Blätter, die oft einen wunderschönen rötlichen Rand aufweisen. Er ist extrem pflegeleicht und langlebig.",
        "care": "• Standort: Hell bis halbschattig. Je heller der Standort, desto intensiver leuchtet die rote Färbung der Blätter.\n• Gießen: Mäßig gießen. Die Erde vor dem erneuten Gießen gut antrocknen lassen. Im Winter den Wasserbedarf noch weiter reduzieren.\n• Schnitt: Zu lang gewordene Stämme können problemlos abgeschnitten werden; sie treiben kurz darauf an der Schnittstelle neu aus.",
    },
    "crassula ovata": {
        "name": "Geldbaum (Kirchbäumchen)",
        "description": "Der Geldbaum ist eine klassische Sukkulente mit dicken, fleischigen, ovalen Blättern, die Wasser speichern. Mit der Zeit bildet er einen dicken, holzigen Stamm und eine baumartige Krone aus. In der chinesischen Feng-Shui-Lehre gilt er als glückbringender und reichtumsfördernder Hausschmuck.",
        "care": "• Standort: Ein sonniger, warmer Platz am Fenster ist optimal. Im Sommer steht er auch gerne geschützt im Freien.\n• Gießen: Sparsam gießen. Erst wenn die Erde komplett ausgetrocknet ist, wird durchdringend gewässert.\n• Erde: Spezielle Kakteen- oder Sukkulentenerde mit hohem Sand- und Mineralanteil verwenden.",
    },
    "yucca elephantipes": {
        "name": "Yucca-Palme (Palmlilie)",
        "description": "Die Yucca-Palme besitzt einen kräftigen, palmenähnlichen Stamm (der an Elefantenfüße erinnert) und schopfartig angeordnete, lange, steif-spitze Blätter. Sie ist eine sehr robuste, lichthungrige Pflanze, die auch in trockener Zimmerluft hervorragend gedeiht.",
        "care": "• Standort: Maximal hell, gerne vollsonnig. Ein Südfenster oder ein heller Wintergarten sind perfekt.\n• Gießen: Während der Wachstumsphase im Sommer mäßig gießen, im Winter deutlich sparsamer halten.\n• Rückschnitt: Zu große Exemplare können im Frühjahr problemlos gekürzt werden, da sie aus dem Stamm willig wieder austreiben.",
    },
    "quercus robur": {
        "name": "Stiel-Eiche",
        "description": "Die Stiel-Eiche ist einer der bedeutendsten und langlebigsten heimischen Laubbäume in Mitteleuropa. Sie kann Hunderte von Jahren alt werden, entwickelt eine mächtige, knorrige Krone und tief eingekerbte Blätter. Ökologisch bietet sie Hunderten von Insekten- und Vogelarten einen unverzichtbaren Lebensraum.",
        "care": "• Standort: Ausschließlich im Freien an vollsonnigen Standorten in großen Gärten, Parks oder in freier Natur.\n• Boden: Tiefgründige, nährstoffreiche und humose Böden mit guter Wasserversorgung.\n• Pflege: Ausgewachsene Exemplare benötigen im Garten keine Pflege und versorgen sich komplett selbst.",
    },
    "fagus sylvatica": {
        "name": "Rot-Buche",
        "description": "Die Rot-Buche prägt als dominanter Laubbaum weite Teile der mitteleuropäischen Wälder. Sie hat eine glatte, silbergraue Rinde und leicht gewellte Blätter, die im Herbst wunderschön goldbraun leuchten und oft den ganzen Winter über am Baum haften bleiben.",
        "care": "• Standort: Als stattlicher Wald- und Parkbaum im Freien, bevorzugt in kühleren, feuchteren Regionen.\n• Boden: Frische, nährstoffreiche, kalkhaltige bis leicht saure Ton- und Lehmböden.\n• Verwendung: Häufig auch als formgeschnittene Hecke in großen Grundstücken im Einsatz.",
    },
    "betula pendula": {
        "name": "Sand-Birke",
        "description": "Die Sand-Birke (auch Hänge-Birke genannt) ist sofort an ihrer leuchtend weißen Rinde (Papierrinde) und den elegant herabhängenden Zweigen erkennbar. Sie ist ein sogenannter Pionierbaum, der rasch auf kahlen Flächen wächst und viel Licht benötigt.",
        "care": "• Standort: Volle Sonne. Die Birke ist ein extremer 'Lichtliebhaber' und verträgt keinerlei Schatten.\n• Boden: Gedeiht selbst auf sehr mageren, sandigen oder trockenen Böden problemlos.\n• Besonderheit: Sie hat einen extrem hohen Wasserbedarf, was bei Pflanzung in Hausnähe im Sommer beachtet werden muss.",
    },
    "acer pseudoplatanus": {
        "name": "Berg-Ahorn",
        "description": "Der Berg-Ahorn ist ein stattlicher, rasch wachsender Laubbaum mit großen, fünfzipfeligen Blättern. Im Herbst verfärben sich die Blätter leuchtend gelb. Im Frühjahr produziert er in großer Zahl geflügelte Spaltfrüchte (die sogenannten 'Helikopter'), die vom Wind weit fortgetragen werden.",
        "care": "• Standort: Sonnig bis halbschattig, ideal für große Gärten, Alleen und Parkanlagen.\n• Boden: Nährstoffreiche, tiefgründige und frische Böden.\n• Pflege: Äußerst windfest und robust gegen raues Klima in höheren Berglagen.",
    },
    "pinus sylvestris": {
        "name": "Wald-Kiefer (Föhre)",
        "description": "Die Wald-Kiefer ist ein immergrüner Nadelbaum, der im oberen Stammbereich eine charakteristisch rotbraune, abblätternde Rinde und blaugrüne, büschelig stehende Nadeln aufweist. Sie ist extrem anpassungsfähig an karge Standorte.",
        "care": "• Standort: Volle Sonne. Benötigt einen sehr hellen und luftigen Standort im Garten.\n• Boden: Kommt mit trockenen, sandigen und saurem Böden bestens zurecht, wo andere Bäume versagen.\n• Gießen: Nur in der Anwachsphase nach dem Pflanzen; danach absolut pflegeleicht.",
    },
    "picea abies": {
        "name": "Gemeine Fichte (Rotfichte)",
        "description": "Die Rotfichte ist der in Deutschland am weitesten verbreitete Nadelbaum und wird traditionell als klassischer Weihnachtsbaum geschätzt. Sie hat dunkelgrüne Nadeln, die vierkantig geformt sind, und hängt herabhängende Zapfen ab.",
        "care": "• Standort: Kühle, feuchtere Standorte im Freien.\n• Boden: Vorzugsweise saure, humose Waldböden mit gleichmäßiger Feuchtigkeit.\n• Pflege: In heißen Sommernsummen anfällig für Trockenstress und Borkenkäferbefall.",
    },
    "taraxacum officinale": {
        "name": "Wiesen-Löwenzahn",
        "description": "Der Wiesen-Löwenzahn ist eine unverkennbare Wildpflanze mit leuchtend gelben Blütenköpfen und stark gezahnten Rosettenblättern. Aus den verblühten Köpfchen entstehen die bekannten, kugelförmigen Pusteblumen, deren Samen vom Wind davongetragen werden. Er gilt oft als Wildkraut, ist aber eine wichtige Bienenweide.",
        "care": "• Standort: Sonnig bis halbschattig auf Wiesen, Rasenflächen und in Gärten.\n• Pflege: Wächst wild und benötigt keinerlei aktive Pflege.",
    },
    "bellis perennis": {
        "name": "Gänseblümchen (Tausendschön)",
        "description": "Das Gänseblümchen ist eine winzige, zarte Wiesenblume mit weiß-rosa Blütenblättern und einer gelben Scheibe in der Mitte. Es blüht bei milder Witterung fast das ganze Jahr über und schließt seine Blüten bei Regen oder Nacht.",
        "care": "• Standort: Sonnige bis halbschattige Rasenflächen und Wiesen.\n• Pflege: Völlig unempfindlich gegenüber regelmäßigem Rasenmähen.",
    },
    "urtica dioica": {
        "name": "Große Brennnessel",
        "description": "Die Große Brennnessel ist eine heimische Wildpflanze, die mit feinen Brennhaaren besetzt ist, die bei Berührung Ameisensäure abgeben. Trotz ihrer Wehrhaftigkeit ist sie ökologisch extrem wertvoll, da sie als Raupenfutterpflanze für über 50 heimische Schmetterlingsarten dient.",
        "care": "• Standort: Stickstoffreiche, nährstoffreiche Böden in Hecken, Waldrändern oder auf Brachflächen.\n• Pflege: Wächst wild und stark wuchernd.",
    },
    "rosa canina": {
        "name": "Hunds-Rose (Heckenrose)",
        "description": "Die Hunds-Rose ist ein heimischer Wildstrauch mit zarten rosa-weißen Blüten im Frühsommer und leuchtend roten, vitamin-C-reichen Hagebutten im Herbst. Sie bietet Vögeln Nahrung und Schutz.",
        "care": "• Standort: Sonnig bis halbschattig, ideal für naturnahe Hecken und Waldränder.\n• Pflege: Sehr robust, schnittverträglich und absolut pflegeleicht.",
    },
}


# ============================================================
# GIFTIGKEITS-DATENBANK
# ============================================================

TOXICITY_INFO = {
    "monstera deliciosa": {
        "details": "Die Monstera enthält im gesamten Pflanzengewebe nadelförmige Calciumoxalat-Kristalle. Bei Aufnahme oder Kauen durch Mensch oder Haustier kommt es sofort zu starkem brennenden Gefühl im Mund, Speichelfluss und Schleimhautreizungen.",
        "audience": "Menschen und Haustiere (Katzen, Hunde)",
    },
    "sansevieria trifasciata": {
        "details": "Der Bogenhanf enthält in den Blättern Saponine (Seifenstoffe). Der Verzehr von Blättern führt bei Haustieren zu heftigem Erbrechen, Durchfall, Magenkrämpfen und starkem Speichelfluss.",
        "audience": "Insbesondere Haustiere (Hunde und Katzen)",
    },
    "aloe vera": {
        "details": "In der Blatthaut befindet sich der Stoff Aloin, der stark abführend und reizend auf den Magen-Darm-Trakt wirkt. Bei innerlicher Aufnahme drohen Krämpfe und Durchfälle.",
        "audience": "Menschen und Haustiere",
    },
    "zamioculcas zamiifolia": {
        "details": "Sämtliche Pflanzenteile der Glücksfeder enthalten Calciumoxalat-Kristalle und Reizstoffe. Bei Verzehr von Blättern oder Stielen treten Schwellungen im Rachenraum und starker Speichelfluss auf.",
        "audience": "Menschen und Haustiere",
    },
    "epipremnum aureum": {
        "details": "Die Efeutute enthält giftige Calciumoxalat-Nadeln. Der Saft führt bei Hautkontakt zu Reizungen und bei Verschlucken zu schmerzhaften Schleimhautschwellungen im Mundbereich.",
        "audience": "Menschen und Haustiere",
    },
    "ficus elastica": {
        "details": "Der weiße, klebrige Milchsaft des Gummibaums enthält Kautschuk und reizende Enzyme. Bei Hautkontakt oder Schleimhautreizungen (z. B. nach Anknabbern durch Haustiere) drohen Entzündungen.",
        "audience": "Menschen und Haustiere",
    },
    "spathiphyllum wallisii": {
        "details": "Das Einblatt enthält wie viele Aronstabgewächse Calciumoxalat-Nadeln. Der Verzehr bewirkt heftiges Brennen im Mund, Schluckbeschwerden und Erbrechen bei Haustieren.",
        "audience": "Menschen und Haustiere",
    },
    "urtica dioica": {
        "details": "Die Brennhaare enthalten Histamin, Acetylcholin und Ameisensäure. Bei Hautkontakt brennt und juckt die betroffene Hautpartie stark und schwillt leicht an.",
        "audience": "Menschen und Tiere (bei direkter Berührung)",
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
        "description": f"Das KI-Modell hat diese Pflanze als '{clean_fallback}' klassifiziert. Es handelt sich hierbei um eine faszinierende Pflanzenart aus dem biologischen Klassifikationsmodell. In unserer erweiterten Datenbank sind zu dieser spezifischen Art im Moment noch keine detaillierten Textbeschreibungen hinterlegt, weshalb wir diesen allgemeinen Standardtext anzeigen.",
        "care": "Grundsätzlich gilt für unbestimmte Pflanzen: Achten Sie auf einen ausgewogenen Standort mit ausreichend (aber nicht zu grellem) Licht, vermeiden Sie permanente Staunässe im Wurzelbereich und passen Sie die Wassergaben den individuellen Umgebungsbedingungen an.",
    }


def get_toxicity_information(label):
    result = find_database_entry(label, TOXICITY_INFO)
    if result:
        return {"known": True, "details": result["details"], "audience": result["audience"]}
    return {
        "known": False, 
        "details": "Für diese spezifische Pflanzenart liegen uns in der Giftstoffdatenbank keine Einträge vor. Das bedeutet jedoch ausdrücklich nicht, dass sie garantiert ungiftig ist. Bei unbekannten Wild- und Zimmerpflanzen sollte stets vorsichtiger Umgang gewahrt werden, besonders in Haushalten mit Kleinkindern und Haustieren.", 
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

                st.info("Hinweis: Die automatisierte Pflanzenbestimmung mittels KI kann fehlerhaft sein. Dieser Check ersetzt keinesfalls eine professionelle botanische Giftigkeitsprüfung oder medizinische Beratung bei Vergiftungsverdacht.")

                st.subheader("📊 Weitere KI-Ergebnisse")
                for result in results:
                    percentage = result["score"] * 100
                    st.write(f"**{result['name']}** – {percentage:.2f} %")
                    st.progress(float(result["score"]))

                st.caption("Die Prozentwerte spiegeln die relative Modell-Wahrscheinlichkeit innerhalb der trainierten Klassen wider.")

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
