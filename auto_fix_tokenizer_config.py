import logging
import os
import sys
import time

import requests

TOKENIZER_PATH = r"saved_models/sentiment/models--ProsusAI--finbert/snapshots/4556d13015211d73dccd3fdd39d39232506f3e43/tokenizer_config.json"
HUGGINGFACE_URL = (
    "https://huggingface.co/ProsusAI/finbert/resolve/main/tokenizer_config.json"
)


def try_remove_file(path: str, retries: int = 3, delay: int = 2) -> bool:
    """Attempt to remove a file with retries and logging."""
    logger = logging.getLogger(__name__)
    for i in range(retries):
        try:
            os.remove(path)
            logger.info(f"Usunięto plik: {path}")
            return True
        except PermissionError as e:
            logger.warning(
                f"Permission denied when removing {path}: {e}. Retry {i+1}/{retries}"
            )
            time.sleep(delay)
        except FileNotFoundError:
            logger.info(f"Plik nie istnieje: {path}")
            return False
        except Exception as e:
            logger.error(f"Błąd podczas usuwania pliku {path}: {e}")
            time.sleep(delay)
    return False


def download_tokenizer_config(dest_path):
    logger = logging.getLogger(__name__)
    logger.info("[INFO] Pobieranie tokenizer_config.json z HuggingFace...")
    r = requests.get(HUGGINGFACE_URL)
    if r.status_code == 200:
        with open(dest_path, "wb") as f:
            f.write(r.content)
        logger.info(f"[INFO] Plik pobrany i zapisany: {dest_path}")
        return True
    else:
        logger.error(
            f"[ERROR] Nie udało się pobrać pliku z HuggingFace. Kod: {r.status_code}"
        )
        return False


def main():
    logger = logging.getLogger(__name__)
    logger.info(f"[INFO] Próba usunięcia pliku: {TOKENIZER_PATH}")
    removed = try_remove_file(TOKENIZER_PATH)
    if not removed:
        logger.critical(
            "[FATAL] Nie można usunąć pliku tokenizer_config.json. Zamknij wszystkie programy mogące korzystać z tego pliku (np. Python, Jupyter, AV), uruchom ponownie komputer lub spróbuj ręcznie usunąć plik."
        )
        sys.exit(1)
    # Upewnij się, że katalog istnieje
    os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
    # Pobierz plik na nowo
    if download_tokenizer_config(TOKENIZER_PATH):
        logger.info("[SUCCESS] Plik tokenizer_config.json został odtworzony.")
        sys.exit(0)
    else:
        logger.critical(
            "[FATAL] Nie udało się pobrać pliku tokenizer_config.json z repozytorium HuggingFace."
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
