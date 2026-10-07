#!/usr/bin/env bash
# ============================================================================
# pack_deploy_bundle.sh — 在训练机(giga2504)上执行,打包闭环评估所需全部资产
#
# 产出: /shared_disk/users/wenhao.lu/JYT/WorldSearch_data/deploy_bundle/
#   ├── searchworld_code.tar.gz        SearchWorld 代码(不含数据/实验产物)
#   ├── uavon_code.tar.gz              UAV-ON 代码(含 DATASET json)
#   ├── stage2_epoch15_slim.ckpt       瘦身后的推理 ckpt (~2.5GB)
#   ├── hf_cache.tar.gz                dinov2-small + siglip2-base (~1.7GB)
#   └── (TEST_ENVS 42GB 不打包——直接 rsync/移动硬盘拷贝, 见 deploy README)
#
# 用法:  bash pack_deploy_bundle.sh
# ============================================================================
set -euo pipefail

SRC_CODE=${SEARCHWORLD_ROOT:-$(pwd)}
SRC_UAVON=${UAVON_ROOT:?Set UAVON_ROOT to the checked-out UAV-ON project}
SRC_ENVS=${UAVON_ENVS:?Set UAVON_ENVS to the UAV-ON TEST_ENVS directory}
SRC_CKPT=${SEARCHWORLD_CHECKPOINT:?Set SEARCHWORLD_CHECKPOINT to a trained checkpoint}
HF_HUB=${HF_HOME:-$HOME/.cache/huggingface}/hub

OUT=${DEPLOY_BUNDLE_DIR:-$PWD/deploy_bundle}
mkdir -p "$OUT"

echo '==> [1/4] Slim checkpoint (strip optimizer states, 4.3GB -> ~2.5GB)'
if [ ! -f "$OUT/stage2_epoch15_slim.ckpt" ]; then
    python "$SRC_CODE/deploy/slim_checkpoint.py" "$SRC_CKPT" \
        "$OUT/stage2_slim.ckpt"
else
    echo '    already exists, skip'
fi

echo '==> [2/4] HF model cache (dinov2-small + siglip2)'
tar -C "$HF_HUB" -czf "$OUT/hf_cache.tar.gz" \
    models--facebook--dinov2-small \
    models--google--siglip2-base-patch16-224

echo '==> [3/4] SearchWorld code'
tar -C "$(dirname "$SRC_CODE")" -czf "$OUT/searchworld_code.tar.gz" \
    --exclude='__pycache__' --exclude='.git' \
    --exclude='*.npz' --exclude='output' \
    "$(basename "$SRC_CODE")"

echo '==> [4/4] UAV-ON code + dataset json'
tar -C "$(dirname "$SRC_UAVON")" -czf "$OUT/uavon_code.tar.gz" \
    --exclude='__pycache__' --exclude='.git' \
    --exclude='TEST_ENVS' --exclude='TRAIN_ENVS' --exclude='logs' \
    --exclude='random_dataset' --exclude='CLIP_logs' --exclude='*.log' \
    "$(basename "$SRC_UAVON")"

echo
echo '================================ bundle contents ==============================='
ls -lh "$OUT"
echo
echo "TEST_ENVS (42GB, 拷贝而非打包): $SRC_ENVS"
echo "    rsync -aP $SRC_ENVS/ <目标机>:/data/UAV-ON/TEST_ENVS/"
echo "================================================================================"
du -sh "$OUT"
