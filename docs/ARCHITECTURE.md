# Architecture technique V2.1

## Flux

`decode_upload` ouvre les octets avec Pillow, respecte la rotation EXIF, convertit en RGB et limite la taille à 15 Mo / 20 millions de pixels. Le mode « photo » cherche d'abord un quadrilatère OpenCV et le redresse par perspective. Si aucun contour n'est retenu, un repli black-hat groupe de petites composantes sombres alignées, valide la segmentation et retourne un recadrage rectangulaire. Les modes « plaque recadrée » et « manuel » fournissent directement une région ; le manuel se règle au pixel.

`segment_characters` applique un contraste CLAHE, un flou faible, Otsu dans les deux polarités et supprime les composantes blanches qui touchent le bord. Le pipeline contrôle les proportions avant segmentation, réduit les grandes régions à 1 600 pixels et agrandit modérément celles de moins de 40 pixels de haut. Les contours retenus sont filtrés par taille et forme, puis triés de gauche à droite. `region_is_plausible` exige une région horizontale et 3 à 10 composantes de hauteur et d'alignement cohérents. Ce contrôle est une **heuristique géométrique**, pas une classification sémantique d'image.

Si la détection, la segmentation ou la plausibilité échoue, `Analysis.status` vaut `no_candidate` ou `unusable`. Le modèle n'est alors ni chargé ni appelé. Pour une région exploitable, chaque glyphe est centré sur un canevas 32 × 32, normalisé en `[0,1]`, puis passé au CNN V2. L'ordre des 36 sorties vient de `data/classes.json`. Le chargement Keras est différé et mis en cache par `st.cache_resource`.

EasyOCR est facultatif et indépendant du CNN. La case de comparaison n'est active que si le paquet et les deux fichiers de poids attendus sont présents. Le téléchargement se fait par `python -m scripts.download_easyocr`, hors de l'interface ; la création du lecteur dans l'application interdit un téléchargement implicite. Le lecteur est mis en cache et initialisé uniquement après une analyse exploitable. Les échecs d'initialisation sont signalés sans masquer la lecture CNN.

## États et interface

La page Reconnaissance distingue entrée, réglages et résultat. Une signature SHA-256 des octets et réglages empêche d'afficher le résultat d'une configuration précédente. Une référence saisie est comparée à la sortie affichée, sans intervenir dans la détection ou l'inférence. Le dernier objet `Analysis` alimente les six onglets pédagogiques du Pipeline CV ; une étape indisponible est explicitement signalée. Les métriques et l'architecture du modèle sont lues dans les artefacts fournis, jamais recalculées depuis les quatre exemples synthétiques.

Streamlit fournit le mode clair/sombre via son menu de thème ; la configuration locale ne force que la couleur primaire. Les colonnes et les blocs natifs s'adaptent à la largeur du navigateur.

## Validation et risques

`pytest` couvre chargement, formes, normalisation, polarité, segmentation, ordre, entrée invalide, détection, recadrage, régions multiples, image non pertinente et navigation AppTest. `python -m scripts.validate_v21` enregistre les sorties réelles sur huit cas synthétiques et les écarts positionnels. Ces cas vérifient le fonctionnement du pipeline, pas sa performance statistique.

Le détecteur n'énumère pas toutes les plaques d'une image ; il choisit la meilleure région candidate. La suppression des bords peut effacer des glyphes qui touchent le cadre. Les plaques à deux lignes ou très inclinées sont en dehors de la validation actuelle. Aucun modèle de détection entraîné adapté n'a été fourni.
