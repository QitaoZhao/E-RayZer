#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-8}"
CONFIG="${CONFIG:-config/erayzer_eval_dl3dv.yaml}"
DATASET_NAME="${DATASET_NAME:-dl3dv}"
MANIFEST_PATH="${MANIFEST_PATH:-${DATASET_PATH:-data/dl3dv_release/manifests/test_full.txt}}"
CKPT_PATH="${CKPT_PATH:-checkpoints/erayzer_dl3dv.pt}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-experiments/evaluation/erayzer-dl3dv}"
FRAME_DIST="${FRAME_DIST:-96}"

export PYTHONPATH="$(pwd)/third_party/gsplat:${PYTHONPATH:-}"
if command -v python >/dev/null 2>&1; then
  TORCH_LIB_PATH="$(python - <<'PY' || true
import glob
import os

try:
    import torch
except Exception:
    raise SystemExit(0)

paths = [os.path.join(os.path.dirname(torch.__file__), "lib")]
paths.extend(glob.glob(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "*", "lib")))
print(":".join(paths))
PY
)"
  if [[ -n "${TORCH_LIB_PATH}" ]]; then
    export LD_LIBRARY_PATH="${TORCH_LIB_PATH}:${LD_LIBRARY_PATH:-}"
  fi
fi

if command -v torchrun >/dev/null 2>&1; then
  LAUNCHER=(torchrun)
elif python -c "import torch.distributed.run" >/dev/null 2>&1; then
  LAUNCHER=(python -m torch.distributed.run)
else
  echo "torchrun not found. Activate the E-RayZer environment first." >&2
  exit 127
fi

if [[ "${CHECK_MANIFEST:-true}" == "true" ]]; then
  python scripts/check_dl3dv_manifest.py \
    --dataset-name "${DATASET_NAME}" \
    --manifest-path "${MANIFEST_PATH}" \
    --max-scenes "${CHECK_MANIFEST_SCENES:-8}"
fi

"${LAUNCHER[@]}" --standalone --nproc_per_node "${NUM_GPUS}" evaluate.py \
  --config "${CONFIG}" \
  -s dataset.name "${DATASET_NAME}" \
  -s dataset.eval_manifest_path "${MANIFEST_PATH}" \
  -s training.dataset "${DATASET_NAME}" \
  -s evaluation.checkpoint_path "${CKPT_PATH}" \
  -s evaluation.out_dir "${EVAL_OUT_DIR}" \
  -s evaluation.max_batches "${MAX_BATCHES:-0}" \
  -s evaluation.save_examples "${SAVE_EXAMPLES:-8}" \
  -s training.batch_size_per_gpu "${BATCH_SIZE_PER_GPU:-1}" \
  -s training.num_workers "${NUM_WORKERS:-2}" \
  -s training.view_selector.type two_frame \
  -s training.view_selector.min_frame_dist "${FRAME_DIST}" \
  -s training.view_selector.max_frame_dist "${FRAME_DIST}"
