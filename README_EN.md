# DreamShot (梦镜)

A video-director workbench for ComfyUI / MiniMax H3: multi-shot scripts, reference assets, sampling and upscaling passes, a per-project video library, and final merge — all in one visual workflow.

## Features

- **Unified Director Studio** (`Ref2VAUnifiedDirectorStudio` / `Ref2VAUnifiedDirectorRunner`): per-shot prompts and reference images/videos/audio, first pass, second/multi-pass sampling, H3 Latent upscale, RTX / TE FlashVSR video upscaling, a four-group video library with recycle bin, shot merging and export, tail-frame handoff between shots.
- **Ref2VA reference gating nodes**: image/video/audio reference gates, region switches, global master gates.
- **Storyboard console**: 3-segment storyboard prompts, storyboard video archiving, automatic final-cut archive, previous-shot upload with auto tail-frame extraction.
- **Sampling control panel family**: first pass, second/N pass (legacy), same-resolution refine template, H3 Latent upscale template, second-pass mode selector, unified multi-pass runner, initial/final video output control board.
- **Model loader panel** (`Ref2VAModelLoader`): UNet / CLIP / video VAE / audio VAE / Turbo LoRA.
- Decode-and-create video, live video preview, RTX video upscale and save.

## Screenshots

> TODO: add screenshots under `docs/images/`.

![DreamShot](docs/images/dreamshot.png)

## Prerequisites

1. ComfyUI with MiniMax H3 support (a version that ships `comfy_extras/nodes_minimax_h3.py`).
2. **Companion plugin `ComfyUI_MiniMaxH3_Director`** (required — this plugin imports its `director.h3_latent_upscale` and `director.refine_pack` modules).
3. MiniMax H3 model files; optionally the H3 Latent upscaler (`minimax_h3_latent_upscaler_3d_bf16.safetensors`).
4. Optional: TE FlashVSR nodes and models; NVIDIA RTX upscaling support.

## Installation

- **Manager / Git URL**: once this repository is public, install it by Git URL from ComfyUI Manager (the repo root is the plugin directory).
- **Git clone**:
  ```bash
  cd ComfyUI/custom_nodes
  git clone https://github.com/TODO/ComfyUI-DreamShot.git
  ```
- **Release ZIP**: download `DreamShot-v1.10.1.zip` and place the `ComfyUI-DreamShot` folder into `ComfyUI/custom_nodes/`.

No extra pip dependencies beyond what ComfyUI already ships (`requirements.txt` is comment-only by default). After installing or upgrading, restart ComfyUI and press `Ctrl + F5` in the browser.

## Usage

Drag `workflows/DreamShot_v1.10.1.json` onto the canvas (single-node workflow for `Ref2VAUnifiedDirectorRunner`; requires this plugin plus `ComfyUI_MiniMaxH3_Director`), then follow the studio: configure shots and assets → first pass → optional refine/upscale passes → merge and export.

## Notes

- Node class names / IDs are stable, so existing workflows keep working. The plugin folder was renamed from `ComfyUI-Ref2VA-Gates` to `ComfyUI-DreamShot`; this does not affect workflows.
- The companion plugin `ComfyUI_MiniMaxH3_Director` is a hard dependency.
- Back up your old plugin folder and workflow JSONs before upgrading; the plugin never deletes project history, video-library content, or outputs automatically.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

See [LICENSE](LICENSE) (TODO: license not yet selected by the project owner).
