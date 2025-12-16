<div align="center">
	<h1>E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training</h1>
	<a href="https://arxiv.org/abs/2512.10950"><img src="https://img.shields.io/badge/arXiv-2512.10950-b31b1b" alt="arXiv"></a>
	<a href="https://qitaozhao.github.io/E-RayZer"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
	<a href="https://huggingface.co/spaces/qitaoz/E-RayZer"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
</div>

![overview](assets/overview.gif)
	
![teaser](https://raw.githubusercontent.com/QitaoZhao/QitaoZhao.github.io/main/research/E-RayZer/images/erayzer_teaser.png)

## Repository Map
- `gradio_app.py`: End-to-end demo UI backed by the inference engine.
- `app_core/engine.py`: Thin wrapper that loads configs, checkpoints, and exports renders, Gaussian point clouds, and turntable videos.
- `erayzer_core/`: Model, transformer blocks, and Gaussian renderer used at inference time.
- `config/erayzer.yaml`: Default inference configuration (image sizes, number of views, view selector, transformer depth, etc.).
- `examples/`: Five curated multi-view sets for quick validation.
- `third_party/gsplat/`: Differentiable Gaussian splatting ops (with our intrinsics-gradient support).

## Quick Start

### 1. Create the environment
```bash
conda create -n erayzer python=3.10 -y
conda activate erayzer

pip install -r requirements.txt
pip install -e third_party/gsplat/  # This takes time
```
### 2. Download or place checkpoints
- `checkpoints/erayzer_multi.pt`: Multi-dataset model (default for the demo).
- `checkpoints/erayzer_dl3dv.pt`: Model trained on DL3DV only.

The Gradio demo automatically downloads missing weights on first launch. You can also fetch them manually from [Hugging Face](https://huggingface.co/qitaoz/E-RayZer/tree/main/checkpoints) and drop the files into `checkpoints/`. Update `--ckpt` if you store them elsewhere.

### 3. Launch the Gradio app
```bash
python gradio_app.py \
	--config config/erayzer.yaml \
	--ckpt checkpoints/erayzer_multi.pt \
	--device cuda:0 \
	--output-dir outputs \
	--share
```
- Upload 10-ish multi-view RGB images (the engine pads/repeats if you provide fewer). You can also click one of the bundled examples to preload sample views.
- Press **Run Inference**. The demo writes predicted camera poses, target renders, a Gaussian point cloud (`point_cloud.glb`), a sweep video (`render_video.mp4`), and a zipped archive under `outputs/<timestamp>/`.
- `--share` exposes the interface through Gradio tunnels; drop the flag to keep it local.

## Citation
If you use E-RayZer in academic or industrial research, please cite:

```bibtex
@misc{zhao2025erayzer,
	title  = {E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training},
	author = {Qitao Zhao and Hao Tan and Qianqian Wang and Sai Bi and Kai Zhang and Kalyan Sunkavalli and Shubham Tulsiani and Hanwen Jiang},
	note   = {arXiv preprint arXiv:2512.10950},
    year   = {2025}
}
```

## Related Project
This project is inspired by and builds upon [RayZer](https://hwjiang1510.github.io/RayZer/). We strongly encourage readers to check it out if you haven’t already.

## Acknowledgements & License
- Model weights (and related assets) are Copyright 2025 Adobe Inc. and are distributed under the Adobe Research License.
- This work was partially done at Adobe Research, where Qitao Zhao worked as a Research Scientist Intern.
- We thank [Zhengqi Li](https://zhengqili.github.io/) for insightful advice. We also thank [Frédéric Fortier-Chouinard](https://lefreud.github.io/), [Jiashun Wang](https://jiashunwang.github.io/), [Yanbo Xu](https://www.yanboxu.com/), [Zihan Wang](https://z1hanw.github.io/), and members of the [Physical Perception Lab](https://shubhtuls.github.io/) for helpful discussions.

