"""PlateVision AI Pro V3.2 — independent CNN and OCR comparison."""

import hashlib
import json
from time import perf_counter
import zipfile
from pathlib import Path

import streamlit as st

from src.cnn_inference import load_classes, load_model, predict_characters
from src.detection import draw_candidate
from src.engines import load_v31_model, predict_v31, prediction_text, v31_availability
from src.exports import annotated_png, result_csv, result_json
from src.ocr import availability, create_reader, read_plate
from src.pipeline import analyze_image
from src.preprocessing import decode_upload, enhance_plate
from src.segmentation import draw_boxes
from src.ui_v32 import show_analysis

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "platevision_cnn_v2.keras"
V31_PATH = ROOT / "models" / "platevision_cnn_v31.keras"
V31_MANIFEST_PATH = ROOT / "data" / "manifest_v31.json"
V31_METRICS_PATH = ROOT / "data" / "validation_metrics_v31.json"
V31_CHART_PATH = ROOT / "assets" / "training_curves_v31.png"
CLASSES_PATH = ROOT / "data" / "classes.json"
RESULTS_PATH = ROOT / "data" / "results.json"
CHART_PATH = ROOT / "assets" / "cnn_v2_training.png"

st.set_page_config(page_title="PlateVision AI Pro", page_icon="🔎", layout="wide",
                   initial_sidebar_state="auto")


@st.cache_resource(show_spinner="Chargement du CNN…")
def cached_model():
    return load_model(MODEL_PATH)


@st.cache_resource(show_spinner="Chargement du CNN V3.1…", max_entries=1)
def cached_v31_model():
    return load_v31_model(V31_PATH, V31_MANIFEST_PATH, load_classes(CLASSES_PATH))


@st.cache_resource(show_spinner="Initialisation d'EasyOCR sur CPU…", max_entries=1)
def cached_reader():
    return create_reader()


@st.cache_data
def experimental_results():
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def model_architecture():
    with zipfile.ZipFile(MODEL_PATH) as archive:
        config = json.loads(archive.read("config.json"))
    return [(layer["class_name"], layer["config"].get("filters") or
             layer["config"].get("units") or "") for layer in config["config"]["layers"]]


def heading(kicker: str, title: str, description: str):
    st.caption(kicker.upper())
    st.title(title)
    st.write(description)


def recognition_page(comparison: bool = False):
    heading("Comparaison IA" if comparison else "Analyse de plaque",
            "Comparer les trois moteurs" if comparison else "Lire une plaque",
            "Examinez séparément la détection, les caractères et les lectures proposées."
            if comparison else "Importez une photo, vérifiez la région détectée et explorez la lecture proposée.")
    with st.container(border=True):
        st.subheader("01 · Image source")
        upload = st.file_uploader("Déposer une image JPG ou PNG", type=["jpg", "jpeg", "png"],
                                  help="15 Mo et 20 millions de pixels maximum.")
        if upload is None:
            st.info("Choisissez une image pour commencer.")
            steps = st.columns(3, gap="medium")
            for col, title, description in zip(
                steps,
                ("Photo", "Région", "Lecture"),
                ("Importez un véhicule ou une plaque.", "Détectez ou recadrez la plaque.", "Examinez les caractères et le CNN."),
            ):
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{title}**")
                        st.caption(description)
            return
        try:
            content = upload.getvalue()
            image = decode_upload(content)
        except ValueError as exc:
            st.error(str(exc), icon="🚫")
            return
        st.image(image, caption=f"Aperçu de l'image importée · {image.shape[1]} × {image.shape[0]} px",
                 alt="Aperçu de l'image source", width="stretch")
    with st.container(border=True):
        st.subheader("02 · Réglages")
        image_type = st.radio("Type d'image", ["Photo de véhicule", "Plaque déjà recadrée"],
                              horizontal=True, help="Une image de courbes ou un document n'est pas une plaque.")
        if image_type == "Photo de véhicule":
            method = st.radio("Méthode de sélection", ["Détection automatique", "Recadrage manuel"],
                              horizontal=True)
            mode = "photo" if method == "Détection automatique" else "manual"
        else:
            mode = "cropped"
        crop_coords = None
        if mode == "manual":
            height, width = image.shape[:2]
            st.caption("Positionnez la région de plaque en pixels pour ajuster les petites images.")
            left, right = st.slider("Limites horizontales (px)", 0, width, (0, width))
            top, bottom = st.slider("Limites verticales (px)", 0, height, (0, height))
            crop_coords = (left, top, right, bottom)
            if left < right and top < bottom:
                st.image(image[top:bottom, left:right], caption="Aperçu du recadrage manuel",
                         alt="Aperçu de la région sélectionnée", width="stretch")
        polarity = st.selectbox("Polarité des caractères",
                                ["auto", "dark-on-light", "light-on-dark"],
                                format_func=lambda value: {"auto": "Automatique",
                                                           "dark-on-light": "Foncés sur fond clair",
                                                           "light-on-dark": "Clairs sur fond sombre"}[value])
        view_mode = "Comparaison" if comparison else "Simple"
        engine_names = ["CNN V2", "CNN V3.1", "EasyOCR"]
        if view_mode == "Simple":
            primary = st.selectbox("Moteur de reconnaissance", engine_names)
        else:
            selected_comparison = st.multiselect("Moteurs à comparer", engine_names,
                                                  default=engine_names)
            primary = selected_comparison[0] if selected_comparison else ""
        v31_state = v31_availability(V31_PATH, V31_MANIFEST_PATH,
                                      load_classes(CLASSES_PATH))
        if v31_state != "ready":
            st.caption(f"CNN V3.1 indisponible ({v31_state}) ; les autres moteurs restent utilisables.")
        ocr_state = availability()
        use_ocr = (st.checkbox("Comparer avec EasyOCR", value=False,
                               disabled=ocr_state != "ready",
                               help="La comparaison nécessite les modèles CRAFT et English G2.")
                   if view_mode == "Simple" and primary != "EasyOCR" else False)
        if ocr_state == "package_missing":
            st.caption("EasyOCR n'est pas installé ; la lecture CNN reste disponible.")
        elif ocr_state == "weights_missing":
            st.caption("Poids EasyOCR absents. Exécutez python -m scripts.download_easyocr, puis rechargez l'application.")
        elif ocr_state == "weights_invalid":
            st.caption("Poids EasyOCR incomplets ou invalides. Relancez python -m scripts.download_easyocr.")
        reference = "".join(ch for ch in st.text_input(
            "Référence connue (facultative)",
            help="Exemple synthétique : 456DEF. Sert uniquement à comparer la lecture obtenue.",
        ).upper() if ch.isalnum())
        if reference:
            st.caption(f"Référence normalisée : {reference}")
        selected = ([primary] + (["EasyOCR"] if use_ocr and primary != "EasyOCR" else [])
                    if view_mode == "Simple" else selected_comparison)
        signature = hashlib.sha256(content + repr((mode, crop_coords, polarity,
                                                    view_mode, tuple(selected))).encode()).hexdigest()
        if st.button("Lancer l'analyse", type="primary", icon="🔎", width="stretch"):
            st.session_state.pop("analysis", None)
            if not selected:
                st.warning("Sélectionnez au moins un moteur de reconnaissance.")
                return
            try:
                with st.status("Analyse en cours…", expanded=True) as status:
                    st.write("Validation de la région et segmentation")
                    classes = load_classes(CLASSES_PATH)
                    started = perf_counter()
                    analysis = analyze_image(image, mode, polarity, classes,
                                             crop_coords=crop_coords, run_cnn=False)
                    pipeline_elapsed_ms = round((perf_counter() - started) * 1000, 2)
                    engine_results = {}
                    if analysis.status == "ready":
                        for name in selected:
                            st.write(f"Reconnaissance {name}")
                            engine_started = perf_counter()
                            try:
                                if name == "CNN V2":
                                    predictions = predict_characters(
                                        cached_model(), classes, analysis.segmentation.characters)
                                    analysis.predictions = predictions  # Legacy V2 diagnostics.
                                    engine_results[name] = {"text": prediction_text(predictions),
                                                            "predictions": predictions}
                                elif name == "CNN V3.1":
                                    predictions = predict_v31(
                                        cached_v31_model(), classes,
                                        analysis.segmentation.characters)
                                    engine_results[name] = {"text": prediction_text(predictions),
                                                            "predictions": predictions}
                                else:
                                    ocr = read_plate(cached_reader(), analysis.raw_plate)
                                    engine_results[name] = {"text": ocr["text"], "ocr": ocr}
                            except Exception as exc:
                                engine_results[name] = {"error": f"{type(exc).__name__} : {exc}"}
                            engine_results[name]["elapsed_ms"] = round((perf_counter() - engine_started) * 1000, 2)
                    status.update(label="Analyse terminée", state="complete", expanded=False)
                st.session_state.analysis = {"signature": signature, "analysis": analysis,
                                             "engines": engine_results, "primary": primary,
                                             "view_mode": view_mode, "source": image,
                                             "pipeline_elapsed_ms": pipeline_elapsed_ms}
            except (ValueError, RuntimeError, OSError, ImportError) as exc:
                st.error(f"Analyse impossible : {exc}", icon="🚫")
    result = st.session_state.get("analysis")
    if result and result["signature"] == signature:
        st.subheader("03 · Résultat")
        show_analysis(result, reference)
        st.caption(f"Détection et segmentation : {result['pipeline_elapsed_ms']:.2f} ms · "
                   "Première lecture = chargement éventuel du moteur inclus.")
        download_cols = st.columns(3)
        with download_cols[0]:
            st.download_button("Télécharger JSON", result_json(result, reference),
                               file_name="platevision_result.json", mime="application/json", width="stretch")
        with download_cols[1]:
            st.download_button("Télécharger CSV", result_csv(result, reference),
                               file_name="platevision_result.csv", mime="text/csv", width="stretch")
        annotated = annotated_png(result)
        if annotated is not None:
            with download_cols[2]:
                st.download_button("Télécharger image annotée", annotated,
                                   file_name="platevision_detection.png", mime="image/png", width="stretch")


def model_page():
    heading("Évaluation fournie", "Analyse du modèle",
            "Résultats sur des caractères synthétiques isolés. Ils ne mesurent pas la lecture de plaques réelles.")
    try:
        data = experimental_results()
        v1, v2 = data["accuracy_v1"], data["accuracy_v2"]
        cols = st.columns(4)
        for col, (label, value) in zip(cols, [
            ("CNN V1", f"{v1:.2%}"), ("CNN V2", f"{v2:.2%}"),
            ("Amélioration", f"+{(round(v2*100, 2)-round(v1*100, 2)):.2f} points"),
            ("Classes", str(data["num_classes"]))]):
            with col:
                with st.container(border=True):
                    st.metric(label, value)
        st.caption(f"{data['test_samples']:,} images de test · {data['image_size'][0]} × {data['image_size'][1]} pixels · perte de test V2 {data['test_loss_v2']:.4f}")
        if CHART_PATH.exists():
            st.image(str(CHART_PATH), caption="Courbes d'entraînement fournies",
                     alt="Accuracy et perte du CNN V2 en entraînement et validation", width="stretch")
        with st.expander("Architecture réelle du CNN V2"):
            st.dataframe([{"Couche": name, "Filtres ou unités": size} for name, size in model_architecture()],
                         hide_index=True, alt="Couches lues dans le fichier Keras fourni")
        if V31_METRICS_PATH.is_file():
            v31 = json.loads(V31_METRICS_PATH.read_text(encoding="utf-8"))
            st.subheader("CNN V3.1 · validation synthétique Lato")
            cols = st.columns(4)
            values = [("Accuracy V3.1", f"{v31['validation_accuracy']:.2%}"),
                      ("Macro F1 V3.1", f"{v31['macro_f1']:.2%}"),
                      ("Rappel J", f"{v31['focus_recall']['J']:.0%}"),
                      ("Rappel 0", f"{v31['classification_report']['0']['recall']:.0%}")]
            for col, (label, value) in zip(cols, values):
                with col:
                    st.metric(label, value)
            st.caption("Ces chiffres proviennent de la validation synthétique Lato de V3.1. Le test final DejaVu est réservé. Ils ne sont pas directement comparables au test historique V2.")
            if V31_CHART_PATH.is_file():
                st.image(str(V31_CHART_PATH), caption="Courbes d'entraînement V3.1 fournies",
                         alt="Accuracy et perte V3.1 en entraînement et validation", width="stretch")
        st.info("Aucune accuracy de détection, de segmentation ou de lecture de plaques réelles n'est disponible.")
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        st.error(f"Artefacts d'analyse indisponibles : {exc}")


def pipeline_page():
    heading("Vision par ordinateur", "Comprendre le pipeline",
            "Explorez les six étapes. Les images intermédiaires proviennent uniquement de la dernière analyse disponible.")
    tabs = st.tabs(["1 · Image", "2 · Détection", "3 · Recadrage", "4 · Prétraitement", "5 · Segmentation", "6 · CNN"])
    result = st.session_state.get("analysis")
    analysis = result["analysis"] if result else None
    details = [
        "Image JPG ou PNG décodée, redressée selon EXIF et limitée en taille.",
        "Contours OpenCV et quadrilatère horizontal plausible pour une photo de véhicule.",
        "La région candidate est redressée par transformation de perspective, ou fournie déjà recadrée.",
        "Niveaux de gris, contraste CLAHE et seuillage d'Otsu selon la polarité.",
        "Composantes filtrées par taille et alignement, triées de gauche à droite.",
        "Les glyphes sont préparés séparément pour V2 et V3.1 ; EasyOCR lit le recadrage brut.",
    ]
    for i, tab in enumerate(tabs):
        with tab:
            st.write(details[i])
            if result is None:
                st.caption("Lancez une analyse dans l'espace Reconnaissance pour afficher une image à cette étape.")
            elif i == 0:
                st.image(result["source"], alt="Image initiale", width="stretch")
            elif i == 1 and analysis.candidate is not None:
                st.image(draw_candidate(result["source"], analysis.candidate), alt="Région candidate", width="stretch")
            elif i == 2 and analysis.plate is not None:
                st.image(analysis.plate, alt="Région recadrée", width="stretch")
            elif i == 3 and analysis.plate is not None:
                st.image(enhance_plate(analysis.plate), alt="Contraste amélioré", width="stretch", clamp=True)
            elif i == 4 and analysis.segmentation is not None:
                st.image(analysis.segmentation.binary, alt="Masque binaire", width="stretch", clamp=True)
                st.image(draw_boxes(analysis.plate, analysis.segmentation.boxes), alt="Composantes retenues", width="stretch")
            elif i == 5 and analysis.status == "ready":
                for name, item in result.get("engines", {}).items():
                    if item.get("text"):
                        st.write(f"{name} : {item['text']}")
                    elif item.get("error"):
                        st.warning(f"{name} : {item['error']}")
            else:
                st.caption("Étape non disponible pour cette analyse.")


def about_page():
    heading("Le projet", "À propos de PlateVision",
            "Une démonstration de reconnaissance de caractères de plaques pour un portfolio de Deep Learning.")
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Objectif et architecture")
            st.write("Une photo ou un recadrage traverse la détection OpenCV puis la segmentation. Les CNN V2 et V3.1 classent les caractères avec des préparations distinctes ; EasyOCR lit la région brute sur CPU.")
            st.write("Le modèle et les résultats expérimentaux proviennent des artefacts fournis. Aucun réentraînement n'est effectué par cette application.")
    with right:
        with st.container(border=True):
            st.subheader("Résultats et limites")
            st.write("CNN V1 : 57,36 % ; CNN V2 : 80,65 % sur le test synthétique historique. V3.1 : 98,17 % en validation synthétique Lato. Ces protocoles ne sont pas directement comparables.")
            st.write("La détection est heuristique, le filtrage suppose une ligne horizontale et la segmentation peut échouer avec flou, reflets, bordures ou caractères accolés.")
    with st.expander("Pistes de recherche"):
        st.write("Évaluer un jeu de plaques réelles annotées, mesurer séparément détection et lecture, puis étudier un détecteur entraîné et une calibration des sorties CNN.")


with st.sidebar:
    st.title("PlateVision")
    st.caption("AI PRO · RELEASE 1.0")
    st.divider()
    space = st.radio("Navigation", ["Analyse de plaque", "Comparaison IA", "À propos du projet"],
                     key="space")
    st.divider()
    st.caption("Traitement sur CPU · V2, V3.1 et EasyOCR")
    st.caption("Une lecture n'est jamais une vérification d'identité.")

if space == "Analyse de plaque":
    recognition_page()
elif space == "Comparaison IA":
    recognition_page(comparison=True)
else:
    about_page()
    with st.expander("Résultats des modèles"):
        model_page()
    with st.expander("Étapes du pipeline"):
        pipeline_page()
