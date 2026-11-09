
# ==================================================
# NEUES PLANTSCAN DESIGN & BENUTZEROBERFLÄCHE
# ==================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f1f8ed, #e5f2df);
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1150px;
    }

    .main-title {
        font-size: clamp(48px, 7vw, 76px);
        font-weight: 900;
        letter-spacing: -3px;
        color: #164b2e;
        line-height: 1.05;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 19px;
        color: #55745d;
        margin-bottom: 30px;
    }

    .hero {
        background: linear-gradient(120deg, #164b2e, #39834d);
        padding: 35px;
        border-radius: 24px;
        color: white;
        margin: 20px 0 30px 0;
        box-shadow: 0 12px 30px rgba(22,75,46,0.15);
    }

    .hero h2 {
        color: white;
        font-size: 28px;
    }

    .hero p {
        color: #e5f4e7;
        font-size: 16px;
    }

    .feature-card {
        background: white;
        padding: 28px;
        border-radius: 20px;
        border: 1px solid #dcebd8;
        min-height: 190px;
        box-shadow: 0 5px 15px rgba(30,80,40,0.06);
        transition: all 0.25s ease;
    }

    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 30px rgba(30,80,40,0.16);
        border-color: #68a875;
    }

    .feature-card h3 {
        color: #205b36;
        font-size: 23px;
    }

    .feature-card p {
        color: #617566;
        font-size: 15px;
    }

    .section-card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #dcebd8;
        margin: 15px 0;
    }

    div[data-testid="stFileUploader"] {
        background: white;
        padding: 20px;
        border-radius: 18px;
        border: 1px solid #dcebd8;
    }

    div.stButton > button {
        background: #287344;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 13px;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background: #164b2e;
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(22,75,46,0.2);
        color: white;
    }

    [data-testid="stDialog"] {
        border-radius: 24px;
    }

    h1, h2, h3 {
        color: #164b2e;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# AUSWAHL-POPUP
# ==================================================

def reset_selection():
    st.session_state["view"] = None
    st.session_state["plant_results"] = None


@st.dialog("🌿 Was möchtest du über deine Pflanze erfahren?", width="large")
def choose_mode():

    st.write(
        "Dein Pflanzenfoto wurde hochgeladen! "
        "Wähle aus, welchen Bereich du öffnen möchtest."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="feature-card">
            <h3>🌱 Pflege</h3>
            <p>
                Erfahre mehr über deine Pflanze,
                ihren Standort, Wasserbedarf
                und ihre Pflege.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Pflege öffnen", use_container_width=True):
            st.session_state["view"] = "care"
            st.rerun()

    with col2:

        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Erkennung</h3>
            <p>
                Sieh dir die KI-Erkennung,
                die wahrscheinlichste Pflanzenart
                und mögliche Risiken an.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Erkennung öffnen", use_container_width=True):
            st.session_state["view"] = "recognition"
            st.rerun()


# ==================================================
# HEADER
# ==================================================

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


# ==================================================
# UPLOAD
# ==================================================

st.subheader("📷 Deine Pflanze entdecken")

uploaded_file = st.file_uploader(
    "Lade ein Foto deiner Pflanze hoch",
    type=["jpg", "jpeg", "png", "webp"],
    key="plant_upload",
    on_change=reset_selection
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")

    st.image(
        image,
        caption="Deine hochgeladene Pflanze",
        width=450
    )

    # Popup automatisch nach dem Upload öffnen
    if st.session_state.get("view") is None:
        choose_mode()

    # ----------------------------------------------
    # KI-ERGEBNISSE EINMALIG BERECHNEN
    # ----------------------------------------------

    if st.session_state.get("view") is not None:

        if st.session_state.get("plant_results") is None:

            try:

                with st.spinner("🌿 Die KI untersucht deine Pflanze..."):

                    processor, model = load_model()

                    results = classify_image(
                        image,
                        processor,
                        model
                    )

                    st.session_state["plant_results"] = results

            except Exception as error:

                st.error("Die Pflanzenanalyse konnte nicht durchgeführt werden.")
                st.exception(error)
                st.stop()

        results = st.session_state["plant_results"]

        best_result = results[0]
        label = best_result["name"]
        confidence = best_result["score"] * 100

        plant = get_plant_information(label)
        toxicity = get_toxicity_information(label)

        # ------------------------------------------
        # PFLEGE-BEREICH
        # ------------------------------------------

        if st.session_state["view"] == "care":

            st.divider()
            st.header("🌱 Pflanzenpflege")

            st.markdown(
                f"""
                <div class="section-card">
                    <h2>{html.escape(plant["name"])}</h2>
                    <p>{html.escape(plant["description"])}</p>
                    <h3>🌿 Pflegehinweise</h3>
                    <p>{html.escape(plant["care"])}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader("⚠️ Sicherheitshinweis")

            if toxicity["status"] == "giftig":
                st.warning(
                    "Für diese Pflanzenart sind mögliche Risiken bekannt: "
                    + toxicity["details"]
                )

            elif toxicity["status"] == "ungiftig":
                st.success(toxicity["title"])
                st.caption(
                    "Diese Angabe gilt nur für die in der Datenbank "
                    "genannte Tiergruppe und ist keine Garantie."
                )

            else:
                st.info(
                    "Die Giftigkeit ist in unserer Datenbank nicht überprüft. "
                    "Unbekannt bedeutet nicht ungefährlich."
                )

        # ------------------------------------------
        # ERKENNUNGS-BEREICH
        # ------------------------------------------

        elif st.session_state["view"] == "recognition":

            st.divider()
            st.header("🔍 KI-Pflanzenerkennung")

            st.markdown(
                f"""
                <div class="section-card">
                    <p style="color:#66806c;">Wahrscheinlichste Pflanzenart</p>
                    <h2>{html.escape(plant["name"])}</h2>
                    <p>{confidence:.2f} % Modell-Konfidenz</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader("⚠️ Giftigkeitsprüfung")

            if toxicity["status"] == "giftig":
                st.error(toxicity["title"])

            elif toxicity["status"] == "ungiftig":
                st.success(toxicity["title"])

            else:
                st.warning(toxicity["title"])

            st.write(toxicity["details"])

            st.caption(
                "Betroffene Gruppe laut Datenbank: "
                + toxicity["audience"]
            )

            st.info(
                "Die KI kann sich bei der Pflanzenbestimmung irren. "
                "Der Giftigkeitscheck ist keine vollständige Prüfung. "
                "Bitte behandle unbekannte Pflanzen vorsichtshalber "
                "als potenziell gefährlich."
            )

            st.subheader("📊 Weitere KI-Ergebnisse")

            for result in results:

                percentage = result["score"] * 100

                st.write(
                    f"**{result['name']}** – {percentage:.2f} %"
                )

                st.progress(float(result["score"]))

            st.caption(
                "Modell-Konfidenzwerte sind keine Garantie "
                "für die Richtigkeit der Bestimmung."
            )

        # ------------------------------------------
        # ZURÜCK ZUR AUSWAHL
        # ------------------------------------------

        if st.button("↩ Zurück zur Auswahl"):
            st.session_state["view"] = None
            st.rerun()

else:

    st.markdown("""
    <div class="section-card">
        <h3>🌿 Wie funktioniert PlantScan?</h3>
        <p>
            1. Lade ein Foto deiner Pflanze hoch.<br>
            2. Wähle zwischen Pflege und Erkennung.<br>
            3. Entdecke Informationen und mögliche Risiken.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.markdown("""
<div style="text-align:center; padding:25px; color:#55745d;">
    <h3>🌿 PlantScan</h3>
    Dein digitaler Pflanzenhelfer<br>
    <small>Ein KI-Schulprojekt</small>
</div>
""", unsafe_allow_html=True)
