<div align="center">

# SearchWorld

**基于世界模型的空间价值引导想象式无人机目标搜索**

[![Paper](https://img.shields.io/badge/Paper-ICLR%202027%20Submission-blue.svg)](#引用)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

[English](README.md) &nbsp;|&nbsp; [简体中文](README.zh-CN.md)

</div>

---

## 概述

自主无人机目标搜索需要在**部分可观测**条件下完成"感知—决策—行动"的闭环。城市环境带来三重困难：搜索区域大而第一视角视野窄，导致覆盖效率低；三维几何密集，限制安全运动；开放词汇指令要求从干扰物中识别**特定**目标。

现有方法大多通过显式地图或记忆表征缓解部分可观测性，但本质上仍是**反应式**的——它们只对**过去**的观测做推理，并不显式预测未来状态。世界模型可以通过想象推演实现**前瞻式**推理，然而生成图像的世界模型推理延迟过高，而纯潜空间世界模型又难以做**空间落地**的规划。

SearchWorld 通过把**显式空间记忆**与**价值引导想象**耦合起来填补这一空白：

- 维护一个三通道的鸟瞰（BEV）记忆——**探索层**、**障碍层**与一个可学习的**空间价值层**——并解码出任务感知的空间价值图，回答"空间里哪里值得去"。
- **认知—动作网络**利用这一学到的空间价值先验，通过想象推演改进策略，**无需训练额外的标量 critic**。
- 训练采用三阶段课程：世界模型预训练 → 专家模仿 → 基于想象的探索精炼。

把想象建立在显式空间表征（而非像素或原始隐变量）之上，使智能体学会**前瞻规划**而不是被动反应。

## 代码结构

```
SearchWorld/
├── train.py                    # 阶段 1 & 2 入口
├── train_stage3.py             # 阶段 3（价值引导想象）入口
├── evaluate_uav.py             # parquet 切分上的离线评估
├── arg_parser.py               # 共用命令行参数
├── configs/                    # gin 配置，每阶段一个
│   ├── base_train_config.gin
│   ├── gwm_pretrain_parquet_config.gin     # 阶段 1
│   ├── stage2_expert_parquet_config.gin    # 阶段 2
│   └── stage3_imagination_config.gin       # 阶段 3
├── model/
│   ├── searchworld/            # 核心网络
│   │   ├── searchworld.py      #   顶层模块
│   │   ├── encoders.py         #   冻结的 DINOv2 / SigLIP2 / 位姿编码器
│   │   ├── rssm.py             #   循环状态空间模型
│   │   ├── bev_memory.py       #   几何构造的 BEV 记忆
│   │   ├── bev_encoder.py      #   空间记忆编码器
│   │   ├── bev_decoder.py      #   三通道空间记忆解码器
│   │   ├── action_policy.py    #   离散动作头 π_θ
│   │   └── decoders.py         #   RGB / 语义解码头
│   ├── loss/
│   │   ├── bev_grid.py         #   规范栅格 + 世界↔栅格仿射映射
│   │   ├── value_target.py     #   V* 监督目标
│   │   └── losses.py           #   重建 + KL + 动作损失
│   ├── rl/                     # 阶段 3 规划
│   │   ├── imagination.py      #   ValueGuidedImaginationEngine
│   │   ├── footprint.py        #   闭式动作运动足迹
│   │   ├── actor_critic.py     #   π_BC 热启动的 actor
│   │   └── trainer.py          #   阶段 3 的 PyTorch-Lightning 模块
│   ├── dataset/                # UAV / parquet 数据模块
│   └── eval/                   # SR / OSR / SPL 指标
├── scripts/
│   ├── data_pipeline/          # 原始 UAV-ON episode → parquet + BEV
│   └── *.py                    # BEV 可视化与 checkpoint 工具
├── deploy/                     # 延迟剖析、部署包、开环测试
├── ros2_deployment/            # ROS2 + Isaac Sim 演示
└── tmp/                        # 开发分析工具
```

### 论文各要素在代码中的位置

| 论文要素 | 实现位置 |
|---|---|
| BEV 栅格与仿射映射（eq. grid） | `model/loss/bev_grid.py` —— **唯一真源** |
| 探索层（eq. expl） | `model/searchworld/bev_memory.py`、`scripts/data_pipeline/convert_uav_to_parquet.py` |
| 障碍层（针孔反投影） | 同上 |
| 价值目标 $V^{\ast}$（eq. valuetarget） | `model/loss/value_target.py` |
| 价值引导更新 $\pi^{+}$（eq. piplus） | `model/rl/imagination.py` |
| 动作损失（eq. actionloss） | `model/rl/imagination.py`、`model/rl/trainer.py` |
| 动作足迹 $K^a$ | `model/rl/footprint.py` |
| 三阶段课程 | `configs/*.gin` + `train.py` / `train_stage3.py` |

## 安装

参考环境为 NVIDIA PyTorch 容器，仓库内 `Dockerfile` 与之对应。

```bash
docker build --network=host -t searchworld:local .
docker run --gpus all --shm-size=512g --ipc=host -it searchworld:local bash
```

或在已有的 Python 3.10+ 环境中安装：

```bash
pip install -r requirements.txt
```

冻结的骨干权重首次使用时从 Hugging Face Hub 下载并缓存：

- `facebook/dinov2-small`
- `google/siglip2-base-patch16-224`

离线机器请预先下载到 Hugging Face 缓存目录，并设置 `HF_HUB_OFFLINE=1`。

## 数据准备

把原始 UAV-ON AirSim episode 转成训练产物的一切逻辑都在 `scripts/data_pipeline/`。
完整流程见 [`scripts/data_pipeline/README.md`](scripts/data_pipeline/README.md)。

> **数据集与权重发布。** 论文所用的预处理 parquet 数据切分以及阶段 1 / 2 / 3 的 checkpoint
> **尚未随仓库提供**，两者都将在评审期结束后上传（计划托管在 Hugging Face Hub）。在此之前，
> 请按下述流程自行构建数据、依次训练三个阶段——下面的步骤不依赖下载。

<!-- TODO(release): 上传后补充 parquet 数据切分与阶段 1/2/3 checkpoint 的链接。 -->

```bash
# 1) 随机动作轨迹  -> 阶段 1（世界模型预训练）
python scripts/data_pipeline/convert_uav_to_parquet.py \
    --input  <raw_uav_dir> \
    --output <out_parquet_dir> \
    --samples-per-file 100 \
    --task-description "Search for the target object"

# 2) 专家演示      -> 阶段 2 / 阶段 3
python scripts/data_pipeline/convert_expert_to_parquet.py \
    --input  <expert_episodes_dir> \
    --output <expert_parquet_dir>

# 3) 指令嵌入所用的 SigLIP2 文本特征缓存
python scripts/data_pipeline/generate_text_feat_cache.py -d <expert_parquet_dir>
```

所有转换脚本共享**同一套** BEV 定义（`scripts/data_pipeline/convert_uav_to_parquet.py` 中的 `BEVGenerator`），
其几何与 `model/loss/bev_grid.py` 严格对齐。parquet 中存储的价值通道只是可选的元数据——
训练损失会用通道 0/1 加 `target_rel` 重新构建 $V^{\ast}$。

想直观查看单个 episode 的 BEV 三通道：

```bash
python scripts/data_pipeline/generate_bev_expert_episode.py -e <episode_dir> --gif
```

## 训练

> 三个阶段均使用 AdamW、权重衰减 0.01、混合精度、学习率 1e-5。
> 一次完整三阶段训练在 8× H20 上约需 20 小时（≈160 GPU-hours）。
>
> **暂未提供预训练权重。** 阶段 1 / 2 / 3 的 checkpoint 同样尚未提供下载，将在评审期结束后与
> 数据集一并上传（见[数据准备](#数据准备)）。因此请从阶段 1 开始、按下述顺序链接三个阶段；
> 阶段 2 的 `-p` 参数填的就是上一阶段的输出，与下载到的 checkpoint 占用同一个位置。

**阶段 1 —— 世界模型预训练**（随机动作轨迹，动作头不参与）：

```bash
python train.py \
    -c configs/gwm_pretrain_parquet_config.gin \
    -d <random_rollout_parquet_dir> \
    -o <output_dir>/stage1_gwm \
    -n <wandb_project> -r stage1
```

**阶段 2 —— 专家模仿**（动作头开启，动作分类损失加权 ×10）：

```bash
python train.py \
    -c configs/stage2_expert_parquet_config.gin \
    -d <expert_parquet_dir> \
    -o <output_dir>/stage2_expert \
    -n <wandb_project> -r stage2 \
    -p <output_dir>/stage1_gwm/checkpoints/last.ckpt
```

**阶段 3 —— 价值引导想象**（RSSM 与价值解码头冻结，只训动作头）：

```bash
python train_stage3.py \
    -c configs/stage3_imagination_config.gin \
    -d <expert_parquet_dir> \
    -o <output_dir>/stage3_imagination \
    -n <wandb_project> -r stage3
```

请把 `configs/stage3_imagination_config.gin` 中的 `ImaginationRLModule.checkpoint_path`
指向阶段 2 的 checkpoint（也可以在 `-c` 后追加一个额外的 gin 文件来覆盖）。

> **gin 的加载顺序很关键。** 必须先 import 所有 DataModule（从而注册它们的 gin
> configurables），**再**调用 `gin.parse_config_file(...)`；否则像 `RSSM.*` 这类绑定会被
> 静默跳过。`train.py` / `train_stage3.py` 已经遵循正确顺序，新增入口脚本时请保持一致。

## 评估

论文中的 SR/OSR/SPL 必须使用 **UAV-ON 项目自带的闭环评估器**计算。UAV-ON
负责 AirSim/Unreal 场景重置、碰撞处理、目标距离和 `stop` 终止语义；本仓库
的 `evaluate_uav.py` 只用于 parquet 上的离线模型检查，不能替代 benchmark
评估。请按照 UAV-ON 文档安装 Python 3.8/AirSim 环境并启动
`scripts/start_server.sh`，再把 SearchWorld policy adapter 接入 UAV-ON 官方
评估入口。每次结果都应保存 UAV-ON commit、场景切分和逐 episode 日志。

```bash
python evaluate_uav.py \
    -c configs/stage2_expert_parquet_config.gin \
    -d <parquet_split_dir> \
    -p <stage2_checkpoint> \
    -o ./eval_results \
    --use-parquet
```

该入口不需要 Weights & Biases 账号：它内联构建 data module、以 `strict=False` 加载
checkpoint，然后跑一次 `Trainer.test`。评测协议：

- **成功（SR）**——在距目标 τ_d = 20 m 内发出 `stop`。
- **Oracle 成功（OSR）**——统计智能体在任意时刻曾进入 τ_d 范围的 episode。
- **SPL**——同时报告。

## 部署

ONNX / TensorRT 导出与 ROS2 + Isaac Sim 闭环演示沿用原部署工具链，对 SearchWorld 的
checkpoint 依然适用：

```bash
python onnx_conversion.py -p <checkpoint> -o model.onnx
python trt_conversion.py  -o model.onnx -t model.trt
```

细节见 [`deploy/README.md`](deploy/README.md) 与
[`ros2_deployment/README.md`](ros2_deployment/README.md)。TensorRT engine 与构建时的
TensorRT 版本及 GPU 绑定，请在目标平台上重新构建。阶段 3 策略导出见
`scripts/export_stage3_policy.py`。

## 引用

论文目前正在评审中，引用信息将在评审结束后补充。

<!-- TODO(anon): 评审结束后替换为 camera-ready 的 BibTeX。 -->

```bibtex
@inproceedings{searchworld2027,
  title     = {SearchWorld: Spatial Value-Grounded Imagination for UAV Object Search via World Models},
  author    = {Anonymous Authors},
  booktitle = {Submitted to the International Conference on Learning Representations (ICLR)},
  year      = {2027},
  note      = {Under review}
}
```

## 致谢

本代码库建立在已开源的 [X-Mobility](https://github.com/NVlabs/X-MOBILITY) 实现
（Apache-2.0）之上，并复用了其若干组件——循环状态空间世界模型、多头解码器栈与部署
工具链。在此感谢其作者。上游代码的部分实现又派生自
[MILE](https://github.com/wayveai/mile)、
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy)、
[Diffusers](https://github.com/huggingface/diffusers) 与
[DINOv2](https://github.com/facebookresearch/dinov2)。

同时感谢 **UAV-ON** 基准的作者提供的仿真器与数据。

## 许可证

本项目以 **Apache License 2.0** 发布，详见 [`LICENSE`](LICENSE)。
