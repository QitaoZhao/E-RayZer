#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "python3.10 was not found. Install Python 3.10 before running this script." >&2
    exit 1
fi

if [[ ! -f third_party/gsplat/setup.py && ! -f third_party/gsplat/pyproject.toml ]]; then
    cat >&2 <<EOF
third_party/gsplat is missing. Clone E-RayZer with --recursive or run:
  git submodule update --init --recursive
EOF
    exit 1
fi

if [[ -f "${VENV_DIR}/bin/python" ]]; then
    VENV_PYTHON_VERSION="$("${VENV_DIR}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "${VENV_PYTHON_VERSION}" != "3.10" ]]; then
        cat >&2 <<EOF
${VENV_DIR} uses Python ${VENV_PYTHON_VERSION}, but E-RayZer training expects Python 3.10.
Remove ${VENV_DIR} or set VENV_DIR to a fresh path before rerunning this script.
EOF
        exit 1
    fi
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
    torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt -r requirements-train.txt

export MAX_JOBS="${MAX_JOBS:-4}"
if python - <<'PY'
import gsplat
import gsplat.cuda._wrapper
PY
then
    echo "Using existing gsplat installation."
else
    python -m pip install --no-build-isolation -e third_party/gsplat
fi

python - <<'PY'
import gsplat
import torch
import torchvision

print(f"Using torch {torch.__version__}")
print(f"Using torchvision {torchvision.__version__}")
print(f"Using gsplat from {gsplat.__file__}")
PY

echo "Environment ready."
echo "Activate it with: source ${VENV_DIR}/bin/activate"
