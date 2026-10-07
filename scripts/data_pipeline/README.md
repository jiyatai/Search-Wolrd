# Data pipeline

Everything that turns raw UAV-ON AirSim episodes into the training artifacts
(parquet episodes, BEV channels, text-feature caches, pseudo-semantic labels)
lives here.  Model / evaluation / deployment utilities stay in `../`.

## Flow

```
raw episodes (AirSim, 4-view RGB-D + pose + actions + target)
        │
        ├─ convert_uav_to_parquet.py      random-action rollouts  -> *.parquet
        │     (defines BEVGenerator + split_episodes, reused below)
        │
        ├─ convert_expert_to_parquet.py   expert demonstrations   -> *.parquet
        │     (paper path; writes `pose` and `target_rel`, required by the
        │      value-layer supervision in stage 1/2)
        │
        ├─ generate_text_feat_cache.py    SigLIP text features    -> cache
        │
        └─ generate_pseudo_semantic_labels_*.py    label variants (SAM mobile)
              (v2 / v3 / dino / improved / better; download_mobilesam.py
               fetches the SAM weights)
```

All converters share one BEV definition: `BEVGenerator` in
`convert_uav_to_parquet.py`, whose occluder/exploration geometry is aligned
with `model/loss/bev_grid.py` (0.4 m affine grid, single source of truth).
The value channel stored in a parquet is optional metadata: the training loss
rebuilds the target V* from channels 0/1 plus `target_rel`
(`model/loss/value_target.py::build_value_target`).

## Inspecting the data

| script | purpose |
|---|---|
| `analyze_uav_dataset.py` | dataset statistics |
| `sample_images.py` | dump sample RGB views |
| `check_labels_simple.py` | label sanity check |
| `test_bev_simple.py` | render BEV channels for a pose sample |
| `test_uav_dataset.py`, `smoke_test_uav_dataset.py` | dataset loader tests |
| `compare_semantic_labels.py`, `visualize_semantic_labels.py` | label comparison / visualization |

(Label *visualization / comparison* utilities that consume model outputs live
in `../`.)

## Running

```bash
cd /path/to/SearchWorld
python scripts/data_pipeline/convert_expert_to_parquet.py \
    --input  <expert_episodes_dir> \
    --output <out.parquet>
```

Each script is self-locating (it puts its own directory on `sys.path`), so the
group can be moved as a whole but should not be split up.

See `README_SEMANTIC.md` for the pseudo-semantic label design.
