<div align="center">

# SearchWorld

**Spatial Value-Grounded Imagination for UAV Object Search via World Models**

[![Paper](https://img.shields.io/badge/Paper-ICLR%202027%20Submission-blue.svg)](#citation)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

[English](README.md) &nbsp;|&nbsp; [简体中文](README.zh-CN.md)

</div>

---

## Overview

**SearchWorld** is a recurrent state-space world model for autonomous UAV object search
under partial observability. Search areas are large, egocentric views are narrow, dense
3-D geometry restricts safe motion, and open-world instructions require picking one
specific target out of distractors.

Existing memory-augmented agents mitigate partial observability by reasoning over *past*
observations — but they remain largely reactive. World models unlock *prospective*
reasoning through imagined rollouts, yet image-generating world models are too slow to
run in the loop, and planning inside a purely latent world model is not spatially
grounded.

SearchWorld closes that gap by coupling **explicit spatial memory** with **value-guided
imagination**:

- It maintains a bird's-eye-view (BEV) memory with three channels — **exploration**,
  **obstacle**, and a learned **spatial value** layer — and decodes a task-aware value
  layer that says *where in space is worth going next*.
- A **cognition–action network** consumes that learned spatial value prior and improves
  the policy through imagined rollouts — **without training a separate scalar critic**.
- Training follows a three-stage curriculum: world-model pretraining → expert imitation
  → imagination-based exploration refinement.

By grounding imagination in explicit spatial representations rather than pixels or raw
latents, the agent learns to *plan prospectively* instead of reacting.

## Repository Structure

```
SearchWorld/
├── train.py                    # Stage 1 & 2 entry point
├── train_stage3.py             # Stage 3 (value-guided imagination) entry point
├── evaluate_uav.py             # offline evaluation on parquet splits
├── arg_parser.py               # shared CLI
├── configs/                    # gin configs, one per stage
│   ├── base_train_config.gin
│   ├── gwm_pretrain_parquet_config.gin     # Stage 1
│   ├── stage2_expert_parquet_config.gin    # Stage 2
│   └── stage3_imagination_config.gin       # Stage 3
├── model/
│   ├── searchworld/            # core network
│   │   ├── searchworld.py      #   top-level module
│   │   ├── encoders.py         #   frozen DINOv2 / SigLIP2 / pose encoders
│   │   ├── rssm.py             #   recurrent state-space model
│   │   ├── bev_memory.py       #   geometric BEV memory construction
│   │   ├── bev_encoder.py      #   spatial memory encoder
│   │   ├── bev_decoder.py      #   3-channel spatial memory decoder
│   │   ├── action_policy.py    #   discrete action head π_θ
│   │   └── decoders.py         #   RGB / segmentation decoders
│   ├── loss/
│   │   ├── bev_grid.py         #   canonical grid + world↔cell affine map
│   │   ├── value_target.py     #   V* supervision target
│   │   └── losses.py           #   reconstruction + KL + action losses
│   ├── rl/                     # Stage 3 planning
│   │   ├── imagination.py      #   ValueGuidedImaginationEngine
│   │   ├── footprint.py        #   closed-form action motion footprints
│   │   ├── actor_critic.py     #   actor with π_BC warm start
│   │   └── trainer.py          #   PyTorch-Lightning module for stage 3
│   ├── dataset/                # UAV / parquet data modules
│   └── eval/                   # SR / OSR / SPL metrics
├── scripts/
│   ├── data_pipeline/          # raw UAV-ON episodes → parquet + BEV
│   └── *.py                    # BEV visualization & checkpoint utilities
├── deploy/                     # latency profiling, deploy bundle, open-loop tests
├── ros2_deployment/            # ROS2 + Isaac Sim demo
└── tmp/                        # developer analysis utilities
```

### Where the paper lives in the code

| Paper element | Implementation |
|---|---|
| BEV grid & affine map (eq. grid) | `model/loss/bev_grid.py` — **single source of truth** |
| Exploration layer (eq. expl) | `model/searchworld/bev_memory.py`, `scripts/data_pipeline/convert_uav_to_parquet.py` |
| Obstacle layer (pinhole back-projection) | same as above |
| Value target $V^{\ast}$ (eq. valuetarget) | `model/loss/value_target.py` |
| Value-guided update $\pi^{+}$ (eq. piplus) | `model/rl/imagination.py` |
| Action loss (eq. actionloss) | `model/rl/imagination.py`, `model/rl/trainer.py` |
| Action footprints $K^a$ | `model/rl/footprint.py` |
| Three-stage curriculum | `configs/*.gin` + `train.py` / `train_stage3.py` |

## Installation

The reference environment is the NVIDIA PyTorch container; a `Dockerfile` mirrors it.

```bash
docker build --network=host -t searchworld:local .
docker run --gpus all --shm-size=512g --ipc=host -it searchworld:local bash
```

Or install into an existing Python 3.10+ environment:

```bash
pip install -r requirements.txt
```

Frozen backbone weights are pulled from the Hugging Face Hub on first use and cached:

- `facebook/dinov2-small`
- `google/siglip2-base-patch16-224`

For offline machines, pre-download them into the Hugging Face cache and set
`HF_HUB_OFFLINE=1`.

## Data Preparation

Everything that turns raw UAV-ON AirSim episodes into training artifacts lives in
`scripts/data_pipeline/`. See [`scripts/data_pipeline/README.md`](scripts/data_pipeline/README.md)
for the full flow.

> **Dataset & checkpoint release.** The preprocessed parquet splits and the
> stage 1 / 2 / 3 checkpoints used in the paper are **not bundled with this repository
> yet** — both will be uploaded (planned: Hugging Face Hub) after the review period.
> Until then, build the data with the pipeline below and train the three stages
> yourself; the instructions never depend on the download.

<!-- TODO(release): link the parquet splits and the stage 1/2/3 checkpoints once uploaded. -->

```bash
# 1) Random-action rollouts  -> Stage 1 (world-model pretraining)
python scripts/data_pipeline/convert_uav_to_parquet.py \
    --input  <raw_uav_dir> \
    --output <out_parquet_dir> \
    --samples-per-file 100 \
    --task-description "Search for the target object"

# 2) Expert demonstrations   -> Stage 2 / Stage 3
python scripts/data_pipeline/convert_expert_to_parquet.py \
    --input  <expert_episodes_dir> \
    --output <expert_parquet_dir>

# 3) SigLIP2 text-feature cache for the instruction embedding
python scripts/data_pipeline/generate_text_feat_cache.py -d <expert_parquet_dir>
```

All converters share **one** BEV definition (`BEVGenerator` in
`scripts/data_pipeline/convert_uav_to_parquet.py`), whose geometry is aligned with
`model/loss/bev_grid.py`. The value channel stored in a parquet is optional metadata —
the training loss rebuilds $V^{\ast}$ from channels 0/1 plus `target_rel`.

To eyeball the BEV channels of a single episode:

```bash
python scripts/data_pipeline/generate_bev_expert_episode.py -e <episode_dir> --gif
```

## Training

The action-head class order is `forward, left, right, ascend, descend, rotl,
rotr, stop`; `stop` is an independent terminal class. Checkpoints trained with
the former `start/stop` alias are incompatible and must be retrained (or
explicitly remapped before loading).

> All stages use AdamW, weight decay 0.01, mixed precision, learning rate 1e-5.
> A complete three-stage run takes ≈20 h on 8× H20 (≈160 GPU-hours).
>
> **No checkpoints yet.** The stage 1 / 2 / 3 checkpoints are not available for download
> yet either (see [Data Preparation](#data-preparation)), so start from Stage 1 and chain
> the stages as shown below. The `-p` argument in Stage 2 is simply the previous stage's
> output — the same slot a downloaded checkpoint would occupy.

**Stage 1 — world-model pretraining** (random-action rollouts, action head inactive):

```bash
python train.py \
    -c configs/gwm_pretrain_parquet_config.gin \
    -d <random_rollout_parquet_dir> \
    -o <output_dir>/stage1_gwm \
    -n <wandb_project> -r stage1
```

**Stage 2 — expert imitation** (action head activated, action loss weighted ×10):

```bash
python train.py \
    -c configs/stage2_expert_parquet_config.gin \
    -d <expert_parquet_dir> \
    -o <output_dir>/stage2_expert \
    -n <wandb_project> -r stage2 \
    -p <output_dir>/stage1_gwm/checkpoints/last.ckpt
```

**Stage 3 — value-guided imagination** (RSSM + value decoder frozen, action head only):

```bash
python train_stage3.py \
    -c configs/stage3_imagination_config.gin \
    -d <expert_parquet_dir> \
    -o <output_dir>/stage3_imagination \
    -n <wandb_project> -r stage3
```

Set `ImaginationRLModule.checkpoint_path` in `configs/stage3_imagination_config.gin` to
the stage-2 checkpoint (or override it with an extra gin file appended to `-c`).

> **gin loading order matters.** All DataModules must be imported — registering their
> gin configurables — **before** `gin.parse_config_file(...)` is called, otherwise
> bindings such as `RSSM.*` are silently skipped. `train.py` / `train_stage3.py` already
> follow the correct order; keep it if you add a new entry point.

## Evaluation

The official SR/OSR/SPL numbers must be obtained with the **UAV-ON project
evaluator**, because it owns the AirSim/Unreal episode reset, collision handling,
target visibility and terminal `stop` semantics. The offline command below only
checks model outputs on parquet data; it is not a substitute for the benchmark
evaluator.

Use the UAV-ON project's documented Python 3.8/AirSim setup and
`scripts/start_server.sh`, then connect a SearchWorld policy adapter to the
official UAV-ON evaluation entry point. The adapter must send seven movement
actions or the independent `stop` action back to UAV-ON. Record the UAV-ON
commit, scene split and per-episode log with every reported result.

```bash
python evaluate_uav.py \
    -c configs/stage2_expert_parquet_config.gin \
    -d <parquet_split_dir> \
    -p <stage2_checkpoint> \
    -o ./eval_results \
    --use-parquet
```

This entry point needs no Weights & Biases account: it builds the data module
inline, loads the checkpoint with `strict=False` and runs a single `Trainer.test`
pass. Benchmark protocol:

- **Success** — issuing `stop` within τ_d = 20 m of the target.
- **Oracle success (OSR)** — credits episodes where the agent passed within τ_d at any
  point.
- **SPL** — reported alongside.

## Deployment

ONNX / TensorRT export and the ROS2 + Isaac Sim closed-loop demo are inherited from the
deployment toolchain and still work with the SearchWorld checkpoints:

```bash
python onnx_conversion.py -p <checkpoint> -o model.onnx
python trt_conversion.py  -o model.onnx -t model.trt
```

See [`deploy/README.md`](deploy/README.md) and
[`ros2_deployment/README.md`](ros2_deployment/README.md) for details. TensorRT engines are
specific to the TensorRT version and GPU they were built on — rebuild on the target
platform. For stage-3 policy export see `scripts/export_stage3_policy.py`.

## Citation

The paper is currently under review. Citation information will be added once the review
period ends.

<!-- TODO(anon): replace this block with the camera-ready BibTeX entry. -->

```bibtex
@inproceedings{searchworld2027,
  title     = {SearchWorld: Spatial Value-Grounded Imagination for UAV Object Search via World Models},
  author    = {Anonymous Authors},
  booktitle = {Submitted to the International Conference on Learning Representations (ICLR)},
  year      = {2027},
  note      = {Under review}
}
```

## Acknowledgements

This codebase builds on the released [X-Mobility](https://github.com/NVlabs/X-MOBILITY)
implementation (Apache-2.0) and reuses several of its components — the recurrent
state-space world model, the multi-head decoder stack and the deployment toolchain.
We thank its authors. Parts of the upstream code are in turn derived from
[MILE](https://github.com/wayveai/mile),
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy),
[Diffusers](https://github.com/huggingface/diffusers) and
[DINOv2](https://github.com/facebookresearch/dinov2).

We also thank the authors of the **UAV-ON** benchmark for the simulator and data.

## License

Released under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for details.
