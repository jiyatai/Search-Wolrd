#!/usr/bin/env bash
# =============================================================================
# Reorganize the data-processing code into scripts/data_pipeline/
#
#   bash scripts/reorganize_data_pipeline.sh
#
# Idempotent: safe to re-run; already-moved files are skipped.
# Uses `git mv` for tracked files and plain `mv` for untracked ones.
#
# The moved group is self-contained: its members import each other with
# self-locating paths (Path(__file__).parent), so moving them together keeps
# every import working.  Nothing outside the group imports them.
# =============================================================================
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
DST="scripts/data_pipeline"
PY="${PY:-python3}"

mkdir -p "$DST"

# ---- data-processing files (conversion / label generation / caches) --------
FILES=(
  # paper-path + base converters (share BEVGenerator / split_episodes)
  convert_uav_to_parquet.py
  convert_expert_to_parquet.py
  convert_with_bev.py
  convert_simple.py
  # standalone BEV memory copy used by the offline generators
  bev_memory_simple.py
  # BEV dataset generation / visualization of raw data
  generate_bev_data.py
  generate_bev_from_dataset.py
  generate_bev_expert_episode.py
  remap_v2_to_v3_bev.py
  # feature / label caches
  generate_text_feat_cache.py
  # pseudo-semantic label generation (+ its SAM download helper)
  generate_pseudo_semantic_labels_better.py
  generate_pseudo_semantic_labels_dino.py
  generate_pseudo_semantic_labels_improved.py
  generate_pseudo_semantic_labels_v2.py
  generate_pseudo_semantic_labels_v3.py
  download_mobilesam.py
  # raw-data inspection / sanity checks
  analyze_uav_dataset.py
  sample_images.py
  check_labels_simple.py
  test_bev_simple.py
  # dataset-side tests and label comparison / visualization
  test_uav_dataset.py
  smoke_test_uav_dataset.py
  compare_semantic_labels.py
  visualize_semantic_labels.py
)

echo "== moving ${#FILES[@]} data-processing files -> $DST"
moved=0; skipped=0
for f in "${FILES[@]}"; do
  if [ -f "$DST/$f" ] && [ ! -f "scripts/$f" ]; then
    skipped=$((skipped+1)); continue
  fi
  if [ ! -f "scripts/$f" ]; then
    echo "   [warn] scripts/$f not found, skipped"; skipped=$((skipped+1)); continue
  fi
  if git ls-files --error-unmatch "scripts/$f" >/dev/null 2>&1; then
    git mv "scripts/$f" "$DST/$f" && moved=$((moved+1))
  else
    mv "scripts/$f" "$DST/$f" && moved=$((moved+1))
  fi
done
echo "   moved: $moved   skipped: $skipped"

# ---- the semantic-label design note travels with its pipeline -------------
if [ -f scripts/README_SEMANTIC.md ] && [ ! -f "$DST/README_SEMANTIC.md" ]; then
  mv scripts/README_SEMANTIC.md "$DST/README_SEMANTIC.md" && echo "   moved README_SEMANTIC.md"
fi

# ---- pipeline README (written once) --------------------------------------
if [ ! -f "$DST/README.md" ]; then
cat > "$DST/README.md" <<'MD'
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
MD
  echo "   wrote $DST/README.md"
fi

# ---- post-move checks ----------------------------------------------------
echo "== syntax check"
if command -v "$PY" >/dev/null 2>&1; then
  "$PY" -m py_compile "$DST"/*.py && echo "   py_compile OK"
else
  echo "   [warn] $PY not found; skipped"
fi

echo "== stale references to old paths (should be empty)"
grep -rn --include='*.py' --include='*.md' --include='*.gin' --include='*.sh' \
     -e 'scripts/convert_uav_to_parquet' \
     -e 'scripts/convert_expert_to_parquet' \
     -e 'scripts/bev_memory_simple' \
     -e 'scripts/generate_pseudo_semantic' \
     -e 'scripts/generate_bev_' \
     -e 'scripts/README_SEMANTIC' \
     . 2>/dev/null | grep -v 'data_pipeline/' || echo "   none"

echo
echo "== done.  Layout:"
echo "   scripts/data_pipeline/   <- data processing (this move)"
echo "   scripts/                 <- model eval / visualization / export / profiling"
