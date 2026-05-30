# DL3DV Data Preparation

This page describes the release data flow for DL3DV:

1. Download the selected DL3DV sequences from Hugging Face.
2. Convert them with the E-RayZer converter.
3. Download the mini-train curriculum statistics.
4. Validate the generated manifest before training or evaluation.

E-RayZer does not redistribute DL3DV images or camera metadata. Users should
request access to the DL3DV Hugging Face datasets and download the raw data
directly.

## Splits

The repository includes fixed sequence-id lists:

```text
data_splits/dl3dv/mini_train_100.txt
data_splits/dl3dv/train_full.txt
data_splits/dl3dv/test_full.txt
```

- `mini_train_100`: 100 training sequences for the release training example.
- `train_full`: the full filtered DL3DV training list.
- `test_full`: the 140-scene DL3DV benchmark split used for evaluation.

## 1. Log In to Hugging Face

```bash
pip install huggingface-hub
hf auth login
```

You can also set `HF_TOKEN` instead of logging in interactively.

## 2. Download Raw DL3DV

Download the 100-sequence mini-train split:

```bash
python scripts/download_dl3dv_hf_list.py \
  --scene-id-list data_splits/dl3dv/mini_train_100.txt \
  --output-dir data/dl3dv_raw \
  --source all \
  --resolution 960P \
  --clean-cache
```

Download the full DL3DV benchmark split:

```bash
python scripts/download_dl3dv_hf_list.py \
  --scene-id-list data_splits/dl3dv/test_full.txt \
  --output-dir data/dl3dv_raw \
  --source benchmark \
  --resolution 960P \
  --clean-cache
```

Optional full-training download:

```bash
python scripts/download_dl3dv_hf_list.py \
  --scene-id-list data_splits/dl3dv/train_full.txt \
  --output-dir data/dl3dv_raw \
  --source all \
  --resolution 960P \
  --clean-cache
```

For the benchmark split, the script follows the official
`DL3DV/DL3DV-Benchmark` release and downloads only the files E-RayZer needs:
`nerfstudio/transforms.json` and `nerfstudio/images_4/*`.

## 3. Convert

Convert the mini-train split:

```bash
python scripts/convert_dl3dv_opencv_list.py \
  --raw-root data/dl3dv_raw \
  --output-dir data/dl3dv_release \
  --split-name mini_train_100 \
  --scene-id-list data_splits/dl3dv/mini_train_100.txt \
  --image-mode none
```

Convert the benchmark split:

```bash
python scripts/convert_dl3dv_opencv_list.py \
  --raw-root data/dl3dv_raw \
  --output-dir data/dl3dv_release \
  --split-name test_full \
  --scene-id-list data_splits/dl3dv/test_full.txt \
  --image-mode none
```

Optional full-training conversion:

```bash
python scripts/convert_dl3dv_opencv_list.py \
  --raw-root data/dl3dv_raw \
  --output-dir data/dl3dv_release \
  --split-name train_full \
  --scene-id-list data_splits/dl3dv/train_full.txt \
  --image-mode none
```

The converter reads Hugging Face `transforms.json`, undistorts `images_4`, and
writes E-RayZer manifests under `data/dl3dv_release/manifests/`.

## 4. Download Mini-Train Statistics

The release training command uses the fixed `mini_train_100` split with
precomputed curriculum statistics. Download and extract the artifact into the
converted data directory:

```bash
curl -L "https://drive.google.com/uc?export=download&id=1J0Og8fw2sg38LUxHLRTPgpvJSOIfOstS" \
  -o dl3dv_mini_train_stats.tar.gz
tar -xzf dl3dv_mini_train_stats.tar.gz -C data/dl3dv_release
```

The same file is available from
[Google Drive](https://drive.google.com/file/d/1J0Og8fw2sg38LUxHLRTPgpvJSOIfOstS/view?usp=sharing).

This artifact contains only JSON statistics for the 100 mini-train sequences.
It does not contain DL3DV images.

## 5. Validate

Validate the mini-train manifest:

```bash
python scripts/check_dl3dv_manifest.py \
  --dataset-name dl3dv \
  --manifest-path data/dl3dv_release/manifests/mini_train_100.txt \
  --covisibility-root data/dl3dv_release/covisibility/dl3dv
```

Validate the benchmark manifest:

```bash
python scripts/check_dl3dv_manifest.py \
  --dataset-name dl3dv \
  --manifest-path data/dl3dv_release/manifests/test_full.txt
```

After this, use:

- [Training](train.md) for `scripts/train_dl3dv.sh`
- [Evaluation](eval.md) for `scripts/evaluate_dl3dv.sh`
