"""Results view for independent PlateVision recognition engines."""

import streamlit as st

from .detection import draw_candidate
from .engines import character_diagnostics, reference_check
from .ocr import prepare_ocr_image
from .segmentation import draw_boxes


def _bounded_image(image, caption: str, alt: str, max_width: int = 420,
                   scale: int = 2) -> None:
    width = min(max_width, int(image.shape[1]) * scale)
    with st.container(horizontal_alignment="center"):
        st.image(image, caption=caption, alt=alt, width=width)


def _reference_result(name: str, text: str, reference: str) -> None:
    check = reference_check(text, reference)
    if check is None:
        st.caption("STATUT · NON VÉRIFIÉE · Aucune référence fiable fournie.")
    elif check["exact"]:
        st.success(f"{name} : lecture exacte selon la référence fournie ({reference}). Concordance vérifiée.")
        st.caption(f"{check['correct_positions']}/{check['reference_length']} caractères à la bonne position.")
    else:
        st.warning(f"{name} : lecture non conforme à la référence {reference}.")
        st.caption(f"{check['correct_positions']}/{check['reference_length']} caractères à la bonne position.")


def _cnn_details(name: str, predictions: list[dict]) -> None:
    with st.expander(f"Détail des prédictions {name}"):
        st.dataframe([
            {"Position": i + 1, "Caractère": item["character"],
             "Score brut softmax": round(item["score"], 4),
             "Trois sorties principales": ", ".join(
                 f"{char} {score:.3f}" for char, score in item["top3"])}
            for i, item in enumerate(predictions)
        ], hide_index=True, alt=f"Scores bruts du {name} par caractère")


def show_analysis(result: dict, reference: str = "") -> None:
    analysis, image = result["analysis"], result["source"]
    if analysis.status != "ready":
        st.warning(analysis.message, icon="⚠️")
        if analysis.plate is not None:
            with st.expander("Examiner la région sélectionnée"):
                _bounded_image(analysis.plate, "Région examinée",
                               "Région sélectionnée qui n'a pas pu être lue")
                if analysis.segmentation is not None:
                    _bounded_image(draw_boxes(analysis.plate, analysis.segmentation.boxes),
                                   "Composantes retenues avant validation",
                                   "Composantes extraites avant échec de validation")
        return

    engines = result["engines"]
    primary_name = result["primary"]
    primary = engines.get(primary_name, {})
    st.info(analysis.message)
    glyph_count = len(analysis.segmentation.characters)
    if not 5 <= glyph_count <= 8:
        st.warning(f"{glyph_count} caractères extraits : vérifiez le recadrage et la segmentation avant d'interpréter la lecture.")
    if result["view_mode"] == "Simple":
        with st.container(border=True):
            st.caption(f"RÉSULTAT PRINCIPAL · {primary_name.upper()}")
            if primary.get("error"):
                st.header("Lecture indisponible")
                st.warning(f"{primary_name} : {primary['error']}")
                if primary_name == "EasyOCR":
                    st.caption("Installez requirements-ocr.txt et les poids vérifiés avant l'analyse.")
            else:
                text = primary.get("text") or ""
                st.header(text or "Aucun texte")
                st.caption("Lecture proposée · scores bruts non calibrés.")
                _reference_result(primary_name, text, reference)
                if reference and text != reference:
                    differences = [
                        {"Position": i + 1,
                         "Référence": reference[i] if i < len(reference) else "—",
                         primary_name: text[i] if i < len(text) else "—"}
                        for i in range(max(len(reference), len(text)))
                        if i >= len(reference) or i >= len(text) or reference[i] != text[i]
                    ]
                    st.dataframe(differences, hide_index=True,
                                 alt="Positions différentes entre lecture et référence")

    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True):
            st.subheader("Plaque extraite")
            if analysis.candidate is not None:
                _bounded_image(analysis.candidate.crop, "Détection brute",
                               "Région brute trouvée par le détecteur", max_width=300)
            _bounded_image(analysis.plate, "Recadrage analysé",
                           "Région de plaque analysée", max_width=360)
            if analysis.candidate is not None:
                with st.expander("Localisation dans la photographie"):
                    _bounded_image(draw_candidate(image, analysis.candidate),
                                   "Région candidate", "Quadrilatère de la plaque détectée",
                                   max_width=520, scale=1)
    with right:
        with st.container(border=True):
            st.subheader("Caractères segmentés")
            seg = analysis.segmentation
            _bounded_image(draw_boxes(analysis.plate, seg.boxes),
                           f"{len(seg.boxes)} composantes · polarité {seg.polarity}",
                           "Boîtes des caractères extraits", max_width=360)
            st.image(seg.characters, width=44, clamp=True,
                     alt=[f"Caractère extrait {i+1}" for i in range(len(seg.characters))])

    for name in ("CNN V2", "CNN V3.1"):
        item = engines.get(name)
        if item and item.get("predictions"):
            _cnn_details(name, item["predictions"])

    if len(engines) > 1 or result["view_mode"] == "Comparaison":
        st.subheader("Comparaison des méthodes de reconnaissance")
        st.caption("Même plaque extraite, préparations propres à chaque moteur.")
        with st.expander("Voir les images utilisées pour la comparaison"):
            raw = analysis.raw_plate
            if raw is not None:
                raw_col, prepared_col = st.columns(2, gap="medium")
                with raw_col:
                    _bounded_image(raw, "Région brute de plaque", "Recadrage brut détecté",
                                   max_width=260)
                with prepared_col:
                    _bounded_image(prepare_ocr_image(raw),
                                   "Préparation EasyOCR · Otsu ×4",
                                   "Plaque agrandie et binarisée pour EasyOCR", max_width=260,
                                   scale=1)
        columns = st.columns(len(engines), gap="medium")
        for column, (name, item) in zip(columns, engines.items()):
            with column, st.container(border=True, height="stretch"):
                if item.get("error"):
                    st.metric(name, "Indisponible")
                    if name == "EasyOCR":
                        st.caption("Installer requirements-ocr.txt et les poids vérifiés.")
                    else:
                        st.warning(item["error"])
                else:
                    text = item.get("text") or ""
                    st.metric(name, text or "Aucun texte")
                    if item.get("elapsed_ms") is not None:
                        st.caption(f"Exécution : {item['elapsed_ms']:.2f} ms")
                    if name == "EasyOCR":
                        ocr = item["ocr"]
                        if ocr["score"] is not None:
                            st.metric("Score brut EasyOCR", f"{ocr['score']:.4f}")
                        elif ocr["scores"]:
                            st.caption("Scores bruts par fragment : " + ", ".join(
                                f"{score:.4f}" for score in ocr["scores"]))
                    _reference_result(name, text, reference)
        texts = [(name, item.get("text")) for name, item in engines.items()
                 if not item.get("error") and item.get("text")]
        if len(texts) >= 2:
            if len({text for _, text in texts}) == 1:
                st.info("Les moteurs exécutés concordent ; sans référence, cela ne confirme pas l'exactitude.")
            else:
                st.info("Les moteurs divergent ; aucune lecture n'est choisie automatiquement.")
        st.caption("Scores CNN et EasyOCR bruts, sans probabilité calibrée de réussite.")

    v2, v31 = engines.get("CNN V2", {}), engines.get("CNN V3.1", {})
    if v2.get("predictions") and v31.get("predictions"):
        rows = character_diagnostics(v2["predictions"], v31["predictions"], reference)
        with st.container(border=True):
            st.subheader("Diagnostic des caractères")
            st.caption("Chaque ligne correspond à un caractère segmenté, de gauche à droite. Les alertes signalent un désaccord entre CNN ou avec la référence parmi les paires sensibles ; aucune correction n'est appliquée.")
            st.caption("Paires surveillées : 0/O, 1/I, J/U et 8/B.")
            st.dataframe(rows, hide_index=True,
                         alt="Prédictions par caractère et désaccords entre les deux CNN")
            alerts = [row["Confusion à examiner"] for row in rows
                      if row["Confusion à examiner"]]
            if alerts:
                st.warning("Confusions à examiner : " + ", ".join(alerts))
