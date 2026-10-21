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


def image_preview(image, caption: str, alt: str, max_width: int = 640,
                  scale: int = 1, clamp: bool = False):
    """Keep previews centered and bounded without changing their aspect ratio."""
    width = min(max_width, int(image.shape[1]) * scale)
    with st.container(horizontal_alignment="center"):
        st.image(image, caption=caption, alt=alt, width=width, clamp=clamp)


def recognition_page(comparison: bool = False):
    heading("Comparaison IA" if comparison else "Analyse de plaque",
            "Comparer les trois moteurs" if comparison else "Lire une plaque",
            "Examinez séparément la détection, les caractères et les lectures proposées."
            if comparison else "Importez une photo, vérifiez la région détectée et explorez la lecture proposée.")
    with st.container(border=True):
        st.subheader("Image à analyser")
        upload = st.file_uploader("Déposer une image JPG ou PNG", type=["jpg", "jpeg", "png"],
                                  help="15 Mo et 20 millions de pixels maximum.")
        if upload is None:
            st.info("Choisissez une image pour commencer.")
            return
        try:
            content = upload.getvalue()
            image = decode_upload(content)
        except ValueError as exc:
            st.error(str(exc), icon="🚫")
            return
        image_preview(image, f"Image importée · {image.shape[1]} × {image.shape[0]} px",
                      "Photographie ou plaque importée", max_width=620)
    with st.container(border=True):
        st.subheader("Réglages de l’analyse")
        type_col, method_col = st.columns(2, gap="medium")
        with type_col:
            image_type = st.radio("Type d'image", ["Photo de véhicule", "Plaque déjà recadrée"],
                                  help="Une image de courbes ou un document n'est pas une plaque.")
        with method_col:
            if image_type == "Photo de véhicule":
                method = st.radio("Méthode de sélection", ["Détection automatique", "Recadrage manuel"])
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
                image_preview(image[top:bottom, left:right], "Aperçu du recadrage manuel",
                              "Région sélectionnée manuellement", max_width=420)
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
        with st.expander("Options complémentaires · polarité et référence"):
            polarity = st.selectbox("Polarité des caractères",
                                    ["auto", "dark-on-light", "light-on-dark"],
                                    format_func=lambda value: {"auto": "Automatique",
                                                               "dark-on-light": "Foncés sur fond clair",
                                                               "light-on-dark": "Clairs sur fond sombre"}[value])
            use_ocr = (st.checkbox("Comparer avec EasyOCR", value=False,
                                   disabled=ocr_state != "ready",
                                   help="La comparaison nécessite les modèles CRAFT et English G2.")
                       if view_mode == "Simple" and primary != "EasyOCR" else False)
            reference = "".join(ch for ch in st.text_input(
                "Référence connue (facultative)",
                help="Exemple : J389NLT. Sert uniquement à comparer la lecture obtenue.",
            ).upper() if ch.isalnum())
            if reference:
                st.caption(f"Référence normalisée : {reference}")
        selected = ([primary] + (["EasyOCR"] if use_ocr and primary != "EasyOCR" else [])
                    if view_mode == "Simple" else selected_comparison)
        if ocr_state != "ready" and "EasyOCR" in selected:
            st.info("EasyOCR · Indisponible. Installez requirements-ocr.txt puis lancez "
                    "python -m scripts.download_easyocr avant de recharger la page.")
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
                            if name == "EasyOCR" and ocr_state != "ready":
                                engine_results[name] = {"error": "Indisponible",
                                                        "availability": ocr_state,
                                                        "elapsed_ms": None}
                                continue
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
        st.subheader("Résultat de l’analyse")
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
                     alt="Accuracy et perte du CNN V2 en entraînement et validation", width=760)
        with st.expander("Architecture réelle du CNN V2"):
            st.dataframe([{"Couche": name, "Filtres ou unités": str(size)}
                          for name, size in model_architecture()],
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
                         alt="Accuracy et perte V3.1 en entraînement et validation", width=760)
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
                image_preview(result["source"], "Image initiale", "Image importée", max_width=640)
            elif i == 1 and analysis.candidate is not None:
                image_preview(draw_candidate(result["source"], analysis.candidate),
                              "Région candidate", "Plaque localisée dans l'image", max_width=640)
            elif i == 2 and analysis.plate is not None:
                image_preview(analysis.plate, "Région recadrée", "Plaque extraite",
                              max_width=420, scale=2)
            elif i == 3 and analysis.plate is not None:
                image_preview(enhance_plate(analysis.plate), "Contraste amélioré",
                              "Plaque après amélioration du contraste", max_width=420, scale=2,
                              clamp=True)
            elif i == 4 and analysis.segmentation is not None:
                image_preview(analysis.segmentation.binary, "Masque binaire",
                              "Masque des caractères", max_width=420, scale=2, clamp=True)
                image_preview(draw_boxes(analysis.plate, analysis.segmentation.boxes),
                              "Composantes retenues", "Boîtes des caractères", max_width=420,
                              scale=2)
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
            "Mini-projet académique de lecture expérimentale de plaques, exécuté localement.")
    with st.container(border=True):
        st.subheader("Parcours d'une image")
        st.write("Image → détection OpenCV ou recadrage manuel → segmentation → CNN V2 / CNN V3.1")
        st.caption("EasyOCR lit indépendamment la plaque extraite. Aucun moteur n'utilise la référence pour prédire.")
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True, height="stretch"):
        st.subheader("Technologies")
        st.write("Python · Streamlit · OpenCV · TensorFlow/Keras · EasyOCR")
        st.caption("Deux CNN de caractères isolés ; EasyOCR facultatif sur CPU.")
    with right, st.container(border=True, height="stretch"):
        st.subheader("Mesures et limites")
        st.write("V2 : 80,65 % sur le test synthétique historique. V3.1 : 98,17 % en validation synthétique Lato.")
        st.caption("Protocoles différents. La lecture de plaques réelles n'a pas de taux de réussite établi.")
    st.caption("Flou, reflets, faible résolution et segmentation imparfaite peuvent dégrader la lecture.")


with st.sidebar:
    st.subheader("PlateVision")
    st.caption("AI PRO  ·  PROJET CNN")
    space = st.segmented_control(
        "Navigation", ["Analyse de plaque", "Comparaison IA", "À propos du projet"],
        default="Analyse de plaque", required=True, key="space", width="stretch",
        format_func=lambda value: {"Analyse de plaque": "Lire",
                                   "Comparaison IA": "Comparer",
                                   "À propos du projet": "Projet"}[value],
        label_visibility="collapsed")
    st.caption("Analyse locale · V2 · V3.1 · EasyOCR")

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
