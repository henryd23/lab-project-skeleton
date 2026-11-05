"""Download Tiny ImageNet if needed, train the model, then evaluate it."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PARENT = PROJECT_ROOT / "tiny-imagenet"
DATASET_DIR = DATA_PARENT / "tiny-imagenet-200"
DATASET_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
DATASET_MD5 = "90528d7ca1a48142e341f4ef8d21d0de"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pth"


def dataset_is_ready() -> bool:
    return (
        (DATASET_DIR / "train").is_dir()
        and (DATASET_DIR / "val" / "images").is_dir()
        and (DATASET_DIR / "val" / "val_annotations.txt").is_file()
    )


def download_dataset() -> None:
    if dataset_is_ready():
        print(f"[Dataset] Tiny ImageNet già presente: {DATASET_DIR}", flush=True)
        return

    if DATASET_DIR.exists():
        raise RuntimeError(
            f"La cartella {DATASET_DIR} esiste ma il dataset è incompleto. "
            "Correggila o rinominala, poi rilancia questo script."
        )

    DATA_PARENT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tiny-imagenet-") as temporary_dir:
        temporary_path = Path(temporary_dir)
        archive_path = temporary_path / "tiny-imagenet-200.zip"
        extracted_path = temporary_path / "tiny-imagenet-200"

        print("[Dataset 1/3] Download Tiny ImageNet (circa 250 MB)...", flush=True)

        last_reported_percent = -1

        def report_download_progress(block_count, block_size, total_size):
            nonlocal last_reported_percent
            if total_size <= 0:
                return
            downloaded = min(block_count * block_size, total_size)
            percent = int(downloaded * 100 / total_size)
            if percent >= last_reported_percent + 10 or percent == 100:
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(
                    f"  Download: {percent:3d}% "
                    f"({downloaded_mb:.0f}/{total_mb:.0f} MB)",
                    flush=True,
                )
                last_reported_percent = percent

        urllib.request.urlretrieve(
            DATASET_URL, archive_path, reporthook=report_download_progress
        )

        print("[Dataset 2/3] Verifico l'integrità dell'archivio...", flush=True)
        digest = hashlib.md5(archive_path.read_bytes()).hexdigest()
        if digest != DATASET_MD5:
            raise RuntimeError(
                "Il controllo dell'archivio scaricato non è riuscito. "
                "Elimina il download incompleto e riprova."
            )

        print(
            "[Dataset 3/3] Archivio verificato. Estraggo i file "
            "(può richiedere qualche minuto)...",
            flush=True,
        )
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(temporary_path)

        if not (
            (extracted_path / "train").is_dir()
            and (extracted_path / "val" / "images").is_dir()
            and (extracted_path / "val" / "val_annotations.txt").is_file()
        ):
            raise RuntimeError("L'archivio non contiene la struttura Tiny ImageNet attesa.")

        shutil.move(str(extracted_path), str(DATASET_DIR))
        print(f"[Dataset] Estrazione completata: {DATASET_DIR}", flush=True)


def run_script(script_name: str) -> None:
    script_path = PROJECT_ROOT / script_name
    print(f"\n===== Avvio di {script_name} =====", flush=True)
    subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=True)
    print(f"===== {script_name} terminato =====", flush=True)


def main() -> None:
    import torch
    import torchvision

    print("[Avvio] Controllo dell'ambiente Python...", flush=True)
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  torchvision: {torchvision.__version__}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("  GPU CUDA non disponibile; userò la CPU.", flush=True)

    download_dataset()
    print("\n[Flusso 1/2] Addestramento del modello", flush=True)
    run_script("train.py")

    if not CHECKPOINT.is_file():
        raise RuntimeError(
            f"Addestramento terminato ma checkpoint non trovato: {CHECKPOINT}"
        )

    print("\n[Flusso 2/2] Valutazione del checkpoint migliore", flush=True)
    run_script("eval.py")
    print("\n[Completato] Addestramento e valutazione terminati.", flush=True)


if __name__ == "__main__":
    main()
