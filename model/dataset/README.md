# Dataset

This directory contains the dataset loaders used for training and evaluating SearchWorld models.

## UAVDataset
Loader for the raw AirSim episode layout produced by
[`scripts/data_pipeline/`](../../scripts/data_pipeline/README.md): it reads images and
depth maps straight from disk, which is convenient for debugging, dataloader
development and small-scale experiments.
(The loader is layout-compatible with the upstream
[nvidia/X-Mobility](https://huggingface.co/datasets/nvidia/X-Mobility) ground-vehicle
dataset, which it grew out of.)

### Data Format
Each episode is a directory tree of PNG/NPY/JSON files:

```
<dataset_path>/
 - episode_0001/
    - episode_summary.json     # episode-level metadata
    - step_0000/
       - rgb_front.png         # front-camera RGB
       - depth_front.npy       # GT depth (256x256 float32)
       - semantic_front.npy    # semantic labels (h, w uint8), optional
       - state.json            # per-step state (pose, action, task...)
    - step_0001/ ... step_0049/
 - episode_0002/
    ...
```

## UAVParquetDataset
Parquet (`.pqt`) variant of the same data, and the loader used by every training run:
columnar storage keeps the three-stage curriculum comfortable to stream from disk, and
Parquet is the format emitted by the conversion scripts.

### Data Format
Each split folder holds one or more scenarios, with several runs per scenario. Rows
carry the encoded image/depth columns plus, optionally, the BEV and semantic columns
consumed by the BEV and semantic decoders; see
[`scripts/data_pipeline/README.md`](../../scripts/data_pipeline/README.md) for the
exact column set written by each converter.

### Data Folder Structure
The dataset is organized into train, validation, and test splits, with multiple scenarios in each:

```
data
 - train
   - scenario_0
      - run_0000.pqt
        run_0001.pqt
        ...
   - scenario_1
      - run_0000.pqt
        run_0001.pqt
        ...
 - val
    - scenario_0
      - run_0000.pqt
        run_0001.pqt
        ...
    - scenario_1
      - run_0000.pqt
        run_0001.pqt
        ...
 - test
    - scenario_0
      - run_0000.pqt
        run_0001.pqt
        ...
    - scenario_1
      - run_0000.pqt
        run_0001.pqt
        ...
```

## gin configurables
Both loaders are `@gin.configurable`. A training entry point must import
`model.dataset` (or either loader module) **before** `gin.parse_config_file(...)` is
called, otherwise the `UAVDataModule.*` / `UAVParquetDataModule.*` bindings are
silently skipped, because the entry points parse their configs with
`skip_unknown=True`.

## Semantic label palettes
`isaac_sim_semantic_label.py` defines `SemanticLabel` / `SEMANTIC_COLORS`, the class
id → colour mapping of the simulator segmentation export. It is used by
`model/visualization.py` and, for batches that carry segmentation ground truth, by
`model/eval/searchworld_metrics.py`.
