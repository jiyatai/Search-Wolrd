#!/usr/bin/env bash
# ============================================================================
# setup_new_machine.sh — 新机器一键部署脚本(在目标机执行)
#
# 前置条件:
#   1. NVIDIA 驱动 >= 535 (CUDA 12.x), nvidia-smi 可用
#   2. conda 已安装 (miniconda 即可)
#   3. 已将以下内容放到 $BUNDLE_DIR:
#        searchworld_code.tar.gz / uavon_code.tar.gz /
#        stage2_epoch15_slim.ckpt / hf_cache.tar.gz
#   4. TEST_ENVS 目录(42GB)已 rsync 到 $WORKSPACE/UAV-ON/TEST_ENVS
#   5. 磁盘余量 >= 120GB
#
# 用法:  bash setup_new_machine.sh [BUNDLE_DIR]
# 默认 BUNDLE_DIR=/data/deploy_bundle, WORKSPACE=~/searchworld_deploy
# ============================================================================
set -euo pipefail

BUNDLE_DIR=${1:-/data/deploy_bundle}
WORKSPACE=${WORKSPACE:-$HOME/searchworld_deploy}
ENV_MODEL=sw_infer        # 模型推理环境 py3.10 + torch2.x + transformers4.48
ENV_SIM=uavon             # 仿真客户端环境 py3.8 (airsim msgpack-rpc)

# ───────────────────────── 0. 目录结构 ─────────────────────────
echo '==> [0] workspace layout'
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"
mkdir -p huggingface_cache checkpoints

# ───────────────────────── 1. 解包代码 ─────────────────────────
if [ ! -d SearchWorld ]; then
    tar -xzf "$BUNDLE_DIR/searchworld_code.tar.gz"
fi
if [ ! -d UAV-ON ]; then
    tar -xzf "$BUNDLE_DIR/uavon_code.tar.gz"
fi

# ───────────────────────── 2. ckpt + HF 缓存 ─────────────────────────
if [ ! -f checkpoints/stage2_epoch15_slim.ckpt ]; then
    cp "$BUNDLE_DIR/stage2_epoch15_slim.ckpt" checkpoints/
fi
if [ ! -d huggingface_cache/hub ]; then
    mkdir -p huggingface_cache/hub
    tar -C huggingface_cache/hub -xzf "$BUNDLE_DIR/hf_cache.tar.gz"
fi

# ───────────────────────── 3. 模型推理环境 ─────────────────────────
if ! conda env list | grep -q "^$ENV_MODEL "; then
    echo "==> [3] creating conda env: $ENV_MODEL (py3.10)"
    conda create -y -n "$ENV_MODEL" python=3.10
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_MODEL"

# 与训练容器一致的版本 (torch 官方 cu121 轮子等价于容器内 2.2.0a0)
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install \
    transformers==4.48.1 \
    pytorch-lightning==2.5.0.post0 \
    diffusers==0.29.2 einops==0.7.0 gin-config==0.5.0 \
    numpy==1.26.4 pandas==1.5.3 pyarrow==14.0.1 \
    timm==1.0.14 torcheval==0.0.7 wandb==0.19.4 \
    matplotlib==3.8.2 moviepy==2.1.2 tensorboardX==2.6.2.2 \
    av datasets huggingface-hub jsonlines
pip install polars==1.20.0

# ───────────────────────── 4. 仿真客户端环境 ─────────────────────────
if ! conda env list | grep -q "^$ENV_SIM "; then
    echo "==> [4] creating conda env: $ENV_SIM (py3.8, airsim)"
    conda create -y -n "$ENV_SIM" python=3.8
fi
conda activate "$ENV_SIM"
pip install -r "$WORKSPACE/UAV-ON/requirements.txt"
# UAV-ON README 记载的版本坑: 必须卸掉 msgpack-python 只留 msgpack-rpc-python
pip uninstall -y msgpack-python msgpack-rpc-python || true
pip install msgpack-rpc-python

# ───────────────────────── 5. headless 渲染依赖 ─────────────────────────
echo '==> [5] system deps for headless UE rendering (need sudo)'
echo '    若尚未安装, 请手动执行:'
echo '    sudo apt-get update && sudo apt-get install -y xvfb libvulkan1 vulkan-utils mesa-vulkan-drivers'

# ───────────────────────── 6. 自检 ─────────────────────────
conda activate "$ENV_MODEL"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$WORKSPACE/huggingface_cache"

echo
echo '==> [6] smoke test: load model on GPU'
cd "$WORKSPACE/SearchWorld"
python - <<'PY'
import torch, os
# 1) 注册 configurable (与 train.py 的 import 链一致)
from model.dataset.uav_dataset import UAVDataModule  # noqa
from model.dataset.uav_parquet_dataset import UAVParquetDataModule  # noqa
from model.trainer import SearchWorldTrainer  # noqa
# 2) 解析 gin 配置 (RSSM 等模块的构造参数都在里面)
import gin
gin.parse_config_file('configs/stage2_expert_parquet_config.gin', skip_unknown=True)
# 3) 加载瘦身 ckpt 并跑一次前向
ckpt = os.path.join('..', 'checkpoints', 'stage2_epoch15_slim.ckpt')
model = SearchWorldTrainer.load_from_checkpoint(ckpt, strict=False)
model = model.cuda().eval()
n = sum(p.numel() for p in model.parameters())
with torch.no_grad():
    batch = {
        'image':         torch.zeros(1, 1, 3, 224, 224).cuda(),
        'relative_pose': torch.zeros(1, 1, 4).cuda(),
        'text_feat':     torch.zeros(1, 1, 768).cuda(),
        'bev_memory':    torch.zeros(1, 1, 3, 256, 256).cuda(),
    }
    obs = model.model.observation_encoder(batch)
    print('encoder out:', obs['embedding'].shape)
print(f'OK: model loaded on {torch.cuda.get_device_name(0)}, {n/1e6:.1f}M params')
PY

echo
echo '=========================== 部署完成 ==========================='
echo '工作目录:  '
echo "  $WORKSPACE"
tree -L 2 -d "$WORKSPACE" 2>/dev/null || find "$WORKSPACE" -maxdepth 2 -type d | sort
echo
echo '下一步运行闭环评估:'
echo '  1) conda activate uavon'
echo '     bash UAV-ON/scripts/start_server.sh   # 启动 AirSim/UE 环境服务器'
echo '  2) conda activate sw_infer'
echo '     cd SearchWorld && python evaluate_uav.py --help   # 评估入口'
echo '================================================================'
