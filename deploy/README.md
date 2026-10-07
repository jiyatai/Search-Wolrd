# 闭环评估新机器部署清单

> 目标:在另一台 GPU 机器上部署 UAV-ON 仿真环境 + SearchWorld 世界模型,
> 用 `stage2_expert_v2/epoch=15-step=1536.ckpt` 跑闭环评估。

---

## 1. 硬件要求

| 项目 | 最低 | 推荐 | 说明 |
|---|---|---|---|
| GPU | 8GB × 2 | **16GB 单卡**(3090/4080/4090/A5000) | 推理 batch=1 ~6GB + UE 渲染 ~4-6GB;分两卡各跑一边更稳 |
| 显卡驱动 | ≥ 535 (CUDA 12.1) | 同左 | torch 2.2.2+cu121 |
| CPU | 8 核 | 16 核 | UE 渲染进程 + RPC + 推理 |
| 内存 | 32GB | 64GB | UE 环境加载较吃内存 |
| 磁盘 | 120GB 空闲 | 200GB | 见下表 |

## 2. 需要迁移的资产(~48GB)

| 资产 | 体积 | 源路径(训练机) | 目标位置 |
|---|---|---|---|
| UE 测试环境(14 个场景) | 42GB | `/shared_disk/users/wenhao.lu/JYT/UAV-ON/TEST_ENVS` | `$WORKSPACE/UAV-ON/TEST_ENVS` |
| 推理 ckpt(瘦身版) | ~2.5GB | 由 `pack_deploy_bundle.sh` 生成 | `$WORKSPACE/checkpoints/` |
| HF 缓存(dinov2-small + siglip2) | ~1.7GB | `~/.cache/huggingface/hub` 两个目录 | `$WORKSPACE/huggingface_cache/hub` |
| SearchWorld 代码 | ~50MB | `/mnt/pfs/users/luwenhao/code_jyt/SearchWorld` | `$WORKSPACE/SearchWorld` |
| UAV-ON 代码 + DATASET json | ~150MB | `/mnt/pfs/users/luwenhao/code_jyt/UAV-ON` | `$WORKSPACE/UAV-ON` |

> ckpt 选 **`epoch=15-step=1536.ckpt`**(val_loss 最低点之一,泛化最好),
> 不用 last.ckpt(e50,action 过拟合,val action loss 28.8 vs e15 的 14.8)。

## 3. 部署步骤

### 3.1 训练机:打包(bundle 脚本做 4 件事)

```bash
bash /mnt/pfs/users/luwenhao/code_jyt/SearchWorld/deploy/pack_deploy_bundle.sh
# 产出 /shared_disk/.../WorldSearch_data/deploy_bundle/{4 个文件}
# TEST_ENVS 42GB 不打包,单独 rsync:
rsync -aP /shared_disk/users/wenhao.lu/JYT/UAV-ON/TEST_ENVS/ <新机>:/data/UAV-ON/TEST_ENVS/
```

### 3.2 新机:一键部署

```bash
bash setup_new_machine.sh /data/deploy_bundle
```

脚本自动完成:
1. 解包两份代码
2. 建 **两个 conda 环境**(版本坑见下)
3. 装系统依赖提示(xvfb + vulkan,headless UE 渲染必需)
4. GPU 冒烟测试:加载 ckpt 到显卡

### 3.3 运行闭环评估

```bash
# 终端 1:启动 AirSim/UE 环境服务器(headless)
conda activate uavon
bash UAV-ON/scripts/start_server.sh          # 默认 GPU 0, 端口 30000, Xvfb :100

# 终端 2:模型推理端
conda activate sw_infer
export HF_HOME=$WORKSPACE/huggingface_cache HF_HUB_OFFLINE=1
cd SearchWorld
python evaluate_uav.py ...   # 离线评估入口(闭环仍需按 UAV-ON 场景适配, 见 §5)
```

## 4. 版本坑(重点!)

| 坑 | 说明 | 解决 |
|---|---|---|
| **python 版本冲突** | UAV-ON 官方 py3.8 + `msgpack-rpc`;SearchWorld 需 py3.10(transformers 4.48 支持 SigLIP2) | 两个独立 conda 环境,`uavon` 只跑 AirSim 客户端/服务器,`sw_infer` 跑模型 |
| msgpack 版本 | `pip install msgpack-python` 会导致 "Ping returned false" | 只装 `msgpack-rpc-python`(setup 脚本已处理) |
| torch 版本 | 容器内 `2.2.0a0` 是 NVIDIA 内部轮子 | 用官方 `torch==2.2.2+cu121` 等价替代 |
| pyarrow | requirements 里是 NVIDIA 内部 dev 版 | 用官方 `14.0.1` |
| HF 下载 | 新机器若不能直连 huggingface.co | 已随 bundle 携带全部缓存,`HF_HUB_OFFLINE=1` 离线加载 |
| headless 渲染 | UE 无显示器机器上起不来 | `xvfb` + vulkan driver,`start_server.sh` 已内置 Xvfb 启动 |
| 指令集 | UE 环境二进制需要较新 CPU(如 NYC 场景) | 目标机 CPU 建议 ≥ Haswell(2014 后) |
| **gin 加载顺序** | 单独 `load_from_checkpoint` 会报 `RSSM.__init__() missing 7 required arguments` | 必须先 import 各 dataset 模块 + trainer 注册 configurable,再 `gin.parse_config_file`,最后加载。setup 脚本的冒烟测试已是正确顺序,照抄即可 |

## 5. 待办:评估入口适配

当前仓库只保留离线评估入口 `evaluate_uav.py`(在 parquet 切分上跑一次 `Trainer.test`),
闭环评估需要新增一条路径:UAV-ON AirSim 服务器提供观测 → SearchWorld 模型逐步推理
(observation_encoder → RSSM → action_policy)→ 动作回传 AirSim。
涉及:
- AirSim 观测(相机图、相对位姿)→ 模型输入格式的转换器
- `task.description` → SigLIP text embedding(离线批量预计算,新机也可直接用 `sw_infer` 环境算)
- BEV memory 的在线构建/滚动更新

## 6. 快速验证清单(部署完成后逐项打勾)

- [ ] `nvidia-smi` 正常,驱动 ≥ 535
- [ ] `conda activate sw_infer && python -c "import torch; print(torch.cuda.is_available())"` → True
- [ ] 冒烟测试通过:setup 脚本最后自动跑(加载 634M 模型到 GPU)
- [ ] `conda activate uavon && python -c "import msgpackrpc"` 成功
- [ ] `bash UAV-ON/scripts/start_server.sh` 后,UE 进程在跑(`nvidia-smi` 可见 ~4-6GB 渲染占用),server 日志无报错
- [ ] AirSim 客户端能连上 30000 端口("Ping returned true")
- [ ] 跑通 1 个场景(如 `DATASET/valset/Barnyard.json`)的 rollout
