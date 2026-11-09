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

