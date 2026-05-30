#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-8}"
CONFIG="${CONFIG:-config/erayzer_train_dl3dv.yaml}"
DATASET_NAME="${DATASET_NAME:-dl3dv}"
MANIFEST_PATH="${MANIFEST_PATH:-${DATASET_PATH:-data/dl3dv_release/manifests/mini_train_100.txt}}"
COVISIBILITY_ROOT="${COVISIBILITY_ROOT:-data/dl3dv_release/covisibility/dl3dv}"
FEATURE_SIMILARITY_ROOT="${FEATURE_SIMILARITY_ROOT:-data/dl3dv_release/dino_similarity/dl3dv}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-experiments/checkpoints/erayzer-dl3dv}"
RESUME_CKPT="${RESUME_CKPT:-none}"
RESET_TRAINING_STATE="${RESET_TRAINING_STATE:-false}"
PERCEPTUAL_LOSS_WEIGHT="${PERCEPTUAL_LOSS_WEIGHT:-0.2}"
PERCEPTUAL_LOSS_WEIGHT_PATH="${PERCEPTUAL_LOSS_WEIGHT_PATH:-${ERAYZER_VGG19_WEIGHT_PATH:-weights/imagenet-vgg-verydeep-19.mat}}"
VIEW_SELECTOR_TYPE="${VIEW_SELECTOR_TYPE:-covisibility}"
VIEW_SELECTOR_KEY_NAME="${VIEW_SELECTOR_KEY_NAME:-}"
if [[ -z "${VIEW_SELECTOR_KEY_NAME}" ]]; then
  if [[ "${VIEW_SELECTOR_TYPE}" == "feat_sim" ]]; then
    VIEW_SELECTOR_KEY_NAME="cos_sim"
  else
    VIEW_SELECTOR_KEY_NAME="avg_per_view"
  fi
fi
USE_CURRICULUM="${USE_CURRICULUM:-true}"
CURRICULUM_ITER="${CURRICULUM_ITER:-86000}"
if [[ "${VIEW_SELECTOR_TYPE}" == "feat_sim" ]]; then
  MIN_FRAME_DIST="${MIN_FRAME_DIST:-1.0}"
  MAX_FRAME_DIST="${MAX_FRAME_DIST:-0.75}"
  CURRICULUM_START_MIN_FRAME_DIST="${CURRICULUM_START_MIN_FRAME_DIST:-1.0}"
  CURRICULUM_START_MAX_FRAME_DIST="${CURRICULUM_START_MAX_FRAME_DIST:-1.0}"
elif [[ "${VIEW_SELECTOR_TYPE}" == "two_frame" ]]; then
  MIN_FRAME_DIST="${MIN_FRAME_DIST:-10}"
  MAX_FRAME_DIST="${MAX_FRAME_DIST:-10}"
  CURRICULUM_START_MIN_FRAME_DIST="${CURRICULUM_START_MIN_FRAME_DIST:-10}"
  CURRICULUM_START_MAX_FRAME_DIST="${CURRICULUM_START_MAX_FRAME_DIST:-10}"
else
  MIN_FRAME_DIST="${MIN_FRAME_DIST:-1.0}"
  MAX_FRAME_DIST="${MAX_FRAME_DIST:-0.5}"
  CURRICULUM_START_MIN_FRAME_DIST="${CURRICULUM_START_MIN_FRAME_DIST:-1.0}"
  CURRICULUM_START_MAX_FRAME_DIST="${CURRICULUM_START_MAX_FRAME_DIST:-1.0}"
fi

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
    --covisibility-root "${COVISIBILITY_ROOT}" \
    --max-scenes "${CHECK_MANIFEST_SCENES:-8}"
fi

echo "View selector: type=${VIEW_SELECTOR_TYPE} key=${VIEW_SELECTOR_KEY_NAME} min=${MIN_FRAME_DIST} max=${MAX_FRAME_DIST} start_min=${CURRICULUM_START_MIN_FRAME_DIST} start_max=${CURRICULUM_START_MAX_FRAME_DIST} curriculum=${USE_CURRICULUM} iter=${CURRICULUM_ITER}"

"${LAUNCHER[@]}" --standalone --nproc_per_node "${NUM_GPUS}" train.py \
  --config "${CONFIG}" \
  -s dataset.name "${DATASET_NAME}" \
  -s dataset.train_manifest_path "${MANIFEST_PATH}" \
  -s dataset.covisibility_root "${COVISIBILITY_ROOT}" \
  -s dataset.feature_similarity_root "${FEATURE_SIMILARITY_ROOT}" \
  -s training.dataset "${DATASET_NAME}" \
  -s training.view_selector.type "${VIEW_SELECTOR_TYPE}" \
  -s training.view_selector.key_name "${VIEW_SELECTOR_KEY_NAME}" \
  -s training.view_selector.min_frame_dist "${MIN_FRAME_DIST}" \
  -s training.view_selector.max_frame_dist "${MAX_FRAME_DIST}" \
  -s training.view_selector.use_curriculum "${USE_CURRICULUM}" \
  -s training.view_selector.curriculum_iter "${CURRICULUM_ITER}" \
  -s training.view_selector.curriculum_start_min_frame_dist "${CURRICULUM_START_MIN_FRAME_DIST}" \
  -s training.view_selector.curriculum_start_max_frame_dist "${CURRICULUM_START_MAX_FRAME_DIST}" \
  -s training.checkpoint_dir "${CHECKPOINT_DIR}" \
  -s training.resume_ckpt "${RESUME_CKPT}" \
  -s training.reset_training_state "${RESET_TRAINING_STATE}" \
  -s training.max_fwdbwd_passes "${MAX_STEPS:-152000}" \
  -s training.batch_size_per_gpu "${BATCH_SIZE_PER_GPU:-24}" \
  -s training.num_workers "${NUM_WORKERS:-4}" \
  -s training.print_every "${PRINT_EVERY:-20}" \
  -s training.checkpoint_every "${CHECKPOINT_EVERY:-2000}" \
  -s training.save_last "${SAVE_LAST:-true}" \
  -s training.perceptual_loss_weight "${PERCEPTUAL_LOSS_WEIGHT}" \
  -s training.perceptual_loss_weight_path "${PERCEPTUAL_LOSS_WEIGHT_PATH}"
