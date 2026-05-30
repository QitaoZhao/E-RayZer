# Training

This document describes the minimal DL3DV training release. See
[data.md](data.md) for the required local data format.

## Environment

The easiest setup path is:

```bash
git submodule update --init --recursive
bash scripts/setup_train_env.sh
source .venv/bin/activate
```

The setup script assumes `python3.10` is available, creates `.venv`, and
installs the inference, training, and editable `third_party/gsplat`
dependencies.

If Python 3.10 is available at a specific path, use
`PYTHON_BIN=/path/to/python3.10 bash scripts/setup_train_env.sh`.

The first install compiles the CUDA `gsplat` extension and can take several
minutes. Reusing the same environment avoids rebuilding it.

Manual setup:

```bash
conda create -n erayzer python=3.10 -y
conda activate erayzer
pip install -r requirements.txt
pip install -r requirements-train.txt
git submodule update --init --recursive
pip install -e third_party/gsplat/
```

Download the MatConvNet VGG19 weights used by the perceptual loss:

```bash
mkdir -p weights
curl -L https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat \
  -o weights/imagenet-vgg-verydeep-19.mat
md5sum weights/imagenet-vgg-verydeep-19.mat
```

The expected MD5 is `106118b7cf60435e6d8e04f6a6dc3657`.

## Default DL3DV Setup

The default config `config/erayzer_train_dl3dv.yaml` matches the released DL3DV
training setup:

- 8 GPUs.
- 10 views per sample: 5 input views and 5 target views.
- Batch size 24 per GPU, global batch size 192.
- 152k forward/backward passes.
- AdamW with cosine schedule and 3k warmup steps.
- Focal prediction detached for the first 200 steps.
- Loss: `mse + 0.2 * MatConvNet VGG19 perceptual_loss`.
- Default curriculum: covisibility sampling with `avg_per_view`, which anneals from `[1.0, 1.0]` to `[0.5, 1.0]`.

## Command

Before training, follow [data.md](data.md) to prepare the 100-sequence mini
train split:

1. Use the included `data_splits/dl3dv/mini_train_100.txt` list to download the
   selected raw DL3DV sequences from Hugging Face with
   `scripts/download_dl3dv_hf_sequences.py`.
2. Convert the downloaded raw data with `scripts/convert_dl3dv_opencv_list.py`.
3. Extract the mini-train statistics artifact from Google Drive into
   `data/dl3dv_release`.

After those steps, these files should exist:

```text
data/dl3dv_release/manifests/mini_train_100.txt
data/dl3dv_release/covisibility/dl3dv/<sequence_id>/covisibility.json
data/dl3dv_release/dino_similarity/dl3dv/<sequence_id>/covisibility.json
```

```bash
NUM_GPUS=8 \
DATASET_NAME=dl3dv \
MANIFEST_PATH=data/dl3dv_release/manifests/mini_train_100.txt \
COVISIBILITY_ROOT=data/dl3dv_release/covisibility/dl3dv \
FEATURE_SIMILARITY_ROOT=data/dl3dv_release/dino_similarity/dl3dv \
PERCEPTUAL_LOSS_WEIGHT_PATH=weights/imagenet-vgg-verydeep-19.mat \
CHECKPOINT_DIR=experiments/checkpoints/erayzer-dl3dv \
bash scripts/train_dl3dv.sh
```

The script runs a manifest check before launching `torchrun` unless
`CHECK_MANIFEST=false`.

## Common Overrides

```bash
# Short smoke test
MAX_STEPS=10 NUM_GPUS=8 bash scripts/train_dl3dv.sh

# Change per-GPU batch size or workers
BATCH_SIZE_PER_GPU=12 NUM_WORKERS=8 bash scripts/train_dl3dv.sh

# Resume weights but restart optimizer/scheduler state
RESUME_CKPT=/path/to/ckpt.pt RESET_TRAINING_STATE=true bash scripts/train_dl3dv.sh

# Resume the full training state
RESUME_CKPT=/path/to/ckpt.pt RESET_TRAINING_STATE=false bash scripts/train_dl3dv.sh

# Use DINO feature-similarity curriculum
VIEW_SELECTOR_TYPE=feat_sim \
FEATURE_SIMILARITY_ROOT=data/dl3dv_release/dino_similarity/dl3dv \
bash scripts/train_dl3dv.sh

# Override curriculum bounds explicitly
MIN_FRAME_DIST=1.0 MAX_FRAME_DIST=0.75 \
CURRICULUM_START_MIN_FRAME_DIST=1.0 CURRICULUM_START_MAX_FRAME_DIST=1.0 \
bash scripts/train_dl3dv.sh
```

Training checkpoints are written to `CHECKPOINT_DIR`. The latest checkpoint is
also saved as `last.pt` unless `SAVE_LAST=false`.

## Loss

The total loss is:

```text
loss = l2_loss_weight * mse
     + l1_loss_weight * l1
     + perceptual_loss_weight * perceptual_loss
```

DL3DV defaults:

```text
l2_loss_weight = 1.0
l1_loss_weight = 0.0
perceptual_loss_weight = 0.2
```

`perceptual_loss` uses the MatConvNet VGG19 loss used by the training setup.

## Curriculum

The default `VIEW_SELECTOR_TYPE=covisibility` samples views from local
`covisibility/dl3dv/<scene_name>/covisibility.json` files using `avg_per_view`.
It matches the old DL3DV launcher:

```text
start range: [1.0, 1.0]
final range: [0.5, 1.0]
curriculum_iter: 86000
```

`VIEW_SELECTOR_TYPE=feat_sim` uses the same curriculum logic over local
`dino_similarity/dl3dv/<scene_name>/covisibility.json` files using `cos_sim`.
It matches the old DINO feature-similarity launcher:

```text
start range: [1.0, 1.0]
final range: [0.75, 1.0]
curriculum_iter: 86000
```
