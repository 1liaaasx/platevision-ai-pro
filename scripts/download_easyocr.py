"""Download official EasyOCR weights with resume and checksum validation.

Run once after installing requirements-ocr.txt. The Streamlit app never downloads.
"""

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import requests
from easyocr.config import detection_models, recognition_models


def model_directory() -> Path:
    base = os.environ.get("EASYOCR_MODULE_PATH") or os.environ.get("MODULE_PATH")
    return (Path(base).expanduser() if base else Path.home() / ".EasyOCR") / "model"


def checksums(path: Path) -> tuple[str, str]:
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def download_model(config: dict, folder: Path) -> None:
    name = config["filename"]
    target = folder / name
    expected_md5 = config["md5sum"].lower()
    if target.is_file():
        existing_md5, existing_sha256 = checksums(target)
        if existing_md5 == expected_md5:
            print(f"{name}: déjà présent et vérifié ; SHA-256 {existing_sha256}")
            return

    partial_zip = folder / f"{name}.zip.part"
    for attempt in range(1, 6):
        try:
            head = requests.head(config["url"], allow_redirects=True, timeout=(20, 30))
            head.raise_for_status()
            total = int(head.headers["Content-Length"])
            if total <= 0 or total > 500_000_000:
                raise RuntimeError("Taille d'archive EasyOCR inattendue.")
            already = partial_zip.stat().st_size if partial_zip.exists() else 0
            if already > total:
                raise RuntimeError("Archive partielle plus grande que la source officielle.")

            def fetch_range(bounds: tuple[int, int]) -> bytes:
                start, end = bounds
                for retry in range(3):
                    try:
                        with requests.get(config["url"], headers={"Range": f"bytes={start}-{end}"},
                                          timeout=(15, 25)) as response:
                            response.raise_for_status()
                            if response.status_code != 206:
                                raise RuntimeError("Le serveur n'a pas respecté la requête Range.")
                            content_range = response.headers.get("Content-Range", "")
                            if not content_range.startswith(f"bytes {start}-{end}/{total}"):
                                raise RuntimeError("Réponse Range incohérente.")
                            data = response.content
                            if len(data) != end - start + 1:
                                raise RuntimeError("Segment incomplet.")
                            return data
                    except (requests.RequestException, RuntimeError):
                        if retry == 2:
                            raise
                        time.sleep(retry + 1)

            chunk_size = 256 * 1024
            bounds = [(start, min(start + chunk_size - 1, total - 1))
                      for start in range(already, total, chunk_size)]
            with ThreadPoolExecutor(max_workers=8) as pool, partial_zip.open("ab") as output:
                for (start, end), data in zip(bounds, pool.map(fetch_range, bounds)):
                    output.write(data)
                    output.flush()
                    if end == total - 1 or (end + 1) // (4 * 1024 * 1024) != start // (4 * 1024 * 1024):
                        print(f"{name}: {end + 1}/{total} octets", flush=True)
            with ZipFile(partial_zip) as archive:
                members = [item for item in archive.infolist()
                           if Path(item.filename).name == name and not item.is_dir()]
                if len(members) != 1 or members[0].file_size > 500_000_000:
                    raise RuntimeError(f"Archive {name} inattendue.")
                staged = folder / f"{name}.part"
                md5, sha256 = hashlib.md5(), hashlib.sha256()
                with archive.open(members[0]) as source, staged.open("wb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        md5.update(chunk)
                        sha256.update(chunk)
                        output.write(chunk)
            if md5.hexdigest() != expected_md5:
                staged.unlink(missing_ok=True)
                partial_zip.unlink(missing_ok=True)
                raise RuntimeError(f"Somme de contrôle EasyOCR invalide pour {name}.")
            os.replace(staged, target)
            partial_zip.unlink(missing_ok=True)
            print(f"{name}: {target.stat().st_size} octets, MD5 validé, SHA-256 {sha256.hexdigest()}")
            return
        except (requests.RequestException, BadZipFile, EOFError, OSError, RuntimeError) as exc:
            print(f"{name}: tentative {attempt}/5 échouée ({type(exc).__name__}: {exc}) ; "
                  f"{partial_zip.stat().st_size if partial_zip.exists() else 0} octets conservés.",
                  flush=True)
            if attempt == 5:
                raise SystemExit(f"Téléchargement incomplet de {name} ; relancer ce script pour reprendre.") from exc
            time.sleep(min(2 * attempt, 10))


def main() -> None:
    folder = model_directory()
    folder.mkdir(parents=True, exist_ok=True)
    for config in (detection_models["craft"], recognition_models["gen2"]["english_g2"]):
        download_model(config, folder)
    print("Poids EasyOCR officiels et vérifiés. Le lecteur peut être initialisé sans téléchargement.")


if __name__ == "__main__":
    main()
