# Evaluation

This document describes the minimal DL3DV evaluation release. See
[data.md](data.md) for the required local data format.

## Protocol

Evaluation uses the deterministic 10-view protocol from the released DL3DV
setup. The paper DL3DV benchmark uses `FRAME_DIST=96`, which selects these
frame indices from each sequence:

```text
selected: [0, 10, 21, 32, 42, 53, 64, 74, 85, 96]
input:    [0, 21, 42, 64, 96]
target:   [10, 32, 53, 74, 85]
```

The config field `inference_view_selector_type: even_I_B` controls this split.
For quick local debugging, override `FRAME_DIST=10`.

## Command

```bash
NUM_GPUS=8 \
DATASET_NAME=dl3dv \
MANIFEST_PATH=data/dl3dv_release/manifests/test_full.txt \
CKPT_PATH=checkpoints/erayzer_dl3dv.pt \
EVAL_OUT_DIR=experiments/evaluation/erayzer-dl3dv \
FRAME_DIST=96 \
bash scripts/evaluate_dl3dv.sh
```

For a quick local check:

```bash
NUM_GPUS=1 \
MAX_BATCHES=1 \
SAVE_EXAMPLES=1 \
MANIFEST_PATH=data/dl3dv_release/manifests/test_full.txt \
CKPT_PATH=checkpoints/erayzer_dl3dv.pt \
bash scripts/evaluate_dl3dv.sh
```

The script runs a manifest check before launching `torchrun` unless
`CHECK_MANIFEST=false`.

## Expected Results

On the released 140-scene DL3DV benchmark split, using the 10-view protocol
above with `FRAME_DIST=96`, the expected aggregate results are:

| Checkpoint | PSNR | RPA@5 | RPA@15 | RPA@30 |
| --- | ---: | ---: | ---: | ---: |
| `erayzer_dl3dv.pt` | 20.3273 | 0.720952 | 0.884444 | 0.935238 |
| `erayzer_multi.pt` | 19.7037 | 0.598571 | 0.828571 | 0.901746 |

The DL3DV-only checkpoint is the one expected to match the paper DL3DV
benchmark row.

## Outputs

Evaluation writes:

- `summary.csv`: per-scene metrics, a blank line, then an `average` row.
- `metrics.txt`: aggregate metrics.
- `sample_*/`: optional predicted and target render examples.

The CSV columns are:

```text
basename,mse,psnr,RPA@5,RPA@15,RPA@30
```

`RPA` is the relative pose accuracy metric at the listed degree thresholds.
