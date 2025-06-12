#!/bin/bash
# Automatyczna instalacja środowiska ZoL0 w WSL2 (Ubuntu)
# Uruchom ten skrypt w katalogu projektu w WSL2: bash ./setup_wsl2.sh

set -e

# 1. Instalacja zależności systemowych
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip build-essential git curl

# 2. Instalacja Poetry (jeśli nie jest zainstalowane)
if ! command -v poetry &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3.10 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# 3. Ustawienie domyślnego Pythona dla Poetry
poetry env use python3.10

# 4. Instalacja zależności projektu
poetry install --no-root

# 5. Uruchomienie testów
poetry run pytest

echo "\nŚrodowisko ZoL0 w WSL2 skonfigurowane i przetestowane!"
