<div align="center">
	<h1>E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training</h1>
	<a href="https://arxiv.org/abs/2512.10950"><img src="https://img.shields.io/badge/arXiv-2512.10950-b31b1b" alt="arXiv"></a>
	<a href="https://qitaozhao.github.io/E-RayZer"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
	<a href="https://huggingface.co/spaces/qitaoz/E-RayZer"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
</div>

![overview](assets/overview.gif)

![teaser](https://raw.githubusercontent.com/QitaoZhao/QitaoZhao.github.io/main/research/E-RayZer/images/erayzer_teaser.png)

## News

- 2026.05: Added minimal DL3DV training and evaluation code.
- 2025.12: Initial release.

## Quick Start

```bash
git clone --recursive https://github.com/QitaoZhao/E-RayZer.git
cd E-RayZer

bash scripts/setup_train_env.sh
source .venv/bin/activate
```

If Python 3.10 is available under another name, pass it explicitly:

```bash
PYTHON_BIN=/path/to/python3.10 bash scripts/setup_train_env.sh
```

If you cloned without `--recursive`, initialize the submodule first:

```bash
git submodule update --init --recursive
```

## Checkpoints

The Gradio demo automatically downloads missing weights on first launch. You can
also download them manually from
[Hugging Face](https://huggingface.co/qitaoz/E-RayZer/tree/main/checkpoints) and
place them under `checkpoints/`:

- `checkpoints/erayzer_multi.pt`: multi-dataset model, used by default.
- `checkpoints/erayzer_dl3dv.pt`: DL3DV-only model.

Pass `--ckpt /path/to/checkpoint.pt` if you keep weights elsewhere.

## Demo

```bash
python gradio_app.py \
	--config config/erayzer_inference.yaml \
	--device cuda:0 \
	--share
```

Upload multi-view RGB images, or use one of the bundled examples. The demo
writes predicted camera poses, target renders, `point_cloud.glb`,
`render_video.mp4`, and a zip archive under `experiments/inference/erayzer`.

## Training and Evaluation

We provide a minimal DL3DV training and evaluation release:

- [Data preparation](docs/data.md)
- [Training](docs/train.md)
- [Evaluation](docs/eval.md)

The code here is a re-implementation and differs from the original version
developed at Adobe.

## Repository Map

- `gradio_app.py`: End-to-end Gradio demo UI.
- `app_core/engine.py`: Inference wrapper for configs, checkpoints, renders,
  Gaussian point clouds, and videos.
- `erayzer_core/`: E-RayZer model, transformer blocks, losses, camera utilities,
  and Gaussian renderer.
- `config/erayzer_inference.yaml`: Default inference configuration.
- `config/erayzer_train_dl3dv.yaml`: Minimal DL3DV training configuration.
- `config/erayzer_eval_dl3dv.yaml`: Minimal DL3DV evaluation configuration.
- `training/`: Local DL3DV dataset loader and dataset factory.
- `scripts/`: Environment setup, DL3DV data conversion, manifest checking,
  training, and evaluation scripts.
- `examples/`: Curated multi-view examples for quick inference checks.
- `third_party/gsplat/`: Differentiable Gaussian splatting ops with intrinsics
  gradient support.

## Citation

If you use E-RayZer in academic or industrial research, please cite:

```bibtex
@inproceedings{zhao2026erayzer,
  title     = {E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training},
  author    = {Qitao Zhao and Hao Tan and Qianqian Wang and Sai Bi and Kai Zhang and Kalyan Sunkavalli and Shubham Tulsiani and Hanwen Jiang},
  booktitle = {CVPR},
  year      = {2026}
}
```

## Related Project

This project is inspired by and builds upon
[RayZer](https://hwjiang1510.github.io/RayZer/). We strongly encourage readers
to check it out if you have not already.

## License

- **Code**: MIT License, see `LICENSE`.
- **Model weights**: Adobe Research License, see `LICENSE-WEIGHTS`. The model
  weights are not covered by the MIT License.

## Acknowledgements

This work was partially done at Adobe Research, where Qitao Zhao worked as a
Research Scientist Intern. We thank
[Zhengqi Li](https://zhengqili.github.io/) for insightful advice. We also thank
[Frederic Fortier-Chouinard](https://lefreud.github.io/),
[Jiashun Wang](https://jiashunwang.github.io/),
[Yanbo Xu](https://www.yanboxu.com/), [Zihan Wang](https://z1hanw.github.io/),
and members of the [Physical Perception Lab](https://shubhtuls.github.io/) for
helpful discussions.
