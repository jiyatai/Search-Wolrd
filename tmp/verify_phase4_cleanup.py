#!/usr/bin/env python3
"""Phase-4 cleanup verification.

Checks (read-only):
  1) every .py still parses (avoids compileall, which trips on root-owned __pycache__)
  2) every `model.*` import target resolves to a real module
  3) every gin binding name has a matching @gin.configurable in the repo
  4) no reference to the deleted modules survives outside tmp/
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = {'tmp', '.git', '__pycache__', 'vis_results'}

DELETED = [
    'model/searchworld/vector_net.py',
    'model/eval/prediction_evaluator.py',
    'evaluate.py',
    'model/dataset/isaac_sim_dataset.py',
    'model/dataset/lerobot_dataset.py',
    'configs/lerobot_base_train_config.gin',
    'configs/pretrained_gwm_train_config.gin',
]

py_files = [
    p for p in sorted(ROOT.rglob('*.py'))
    if not any(part in SKIP for part in p.relative_to(ROOT).parts)
]

trees = {}
fail = 0

# ---------------------------------------------------------------- 1) syntax
print(f'--- 1) 语法检查: {len(py_files)} 个 .py ---')
for p in py_files:
    src = p.read_text(encoding='utf-8', errors='replace')
    try:
        trees[p] = ast.parse(src, filename=str(p))
    except SyntaxError as e:
        fail += 1
        print(f'  [SYNTAX] {p.relative_to(ROOT)}:{e.lineno} {e.msg}')
print('  OK — 全部可解析' if fail == 0 else f'  !! {fail} 个文件解析失败')

# ------------------------------------------------------- 2) import targets
print('\n--- 2) model.* 导入目标存在性 ---')
missing = set()
for p, tree in trees.items():
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('model'):
            mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods += [a.name for a in node.names if a.name.startswith('model')]
    for m in mods:
        f = ROOT / (m.replace('.', '/') + '.py')
        d = ROOT / m.replace('.', '/')
        if not f.exists() and not (d / '__init__.py').exists():
            missing.add(f'{p.relative_to(ROOT)} -> {m}')
for s in sorted(missing):
    print(f'  [MISSING] {s}')
if not missing:
    print('  OK — 所有 model.* 目标均存在')

# --------------------------------------------------------- 3) gin bindings
print('\n--- 3) gin 绑定 vs @gin.configurable 定义 ---')
defined = set()
for p, tree in trees.items():
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if any('gin.configurable' in ast.unparse(d) for d in node.decorator_list):
                defined.add(node.name)

unknown = set()
for g in sorted((ROOT / 'configs').glob('*.gin')):
    for i, line in enumerate(g.read_text(encoding='utf-8').splitlines(), 1):
        s = line.strip()
        if not s or s.startswith('#') or '=' not in s:
            continue
        name = s.split('=', 1)[0].strip().split('.')[0]
        if not re.match(r'^[A-Za-z_]\w*$', name) or name.isupper():
            continue  # macros like SEQUENCE_LENGTH=4
        if name not in defined:
            unknown.add(f'{g.name}:{i}  {name}')
for s in sorted(unknown):
    print(f'  [UNKNOWN] {s}')
if not unknown:
    print('  OK — 所有 gin 绑定都能找到 @gin.configurable 定义')

# ------------------------------------------------- 4) deleted-module residue
print('\n--- 4) 已删除模块的残留引用 (排除 tmp/) ---')
stems = [Path(d).stem for d in DELETED]
pat = re.compile('|'.join(re.escape(s) for s in stems))
resid = []
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or any(part in SKIP for part in p.relative_to(ROOT).parts):
        continue
    if p.suffix not in {'.py', '.gin', '.sh', '.md', '.txt', '.cfg', '.toml', '.yaml', '.yml'}:
        continue
    for i, line in enumerate(p.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
        if pat.search(line):
            resid.append(f'{p.relative_to(ROOT)}:{i}  {line.strip()[:100]}')
for s in resid:
    print(f'  [RESIDUE] {s}')
if not resid:
    print('  OK — 无残留')

print('\n--- 5) 已删除文件是否真的不存在 ---')
for d in DELETED:
    print(f'  {"gone " if not (ROOT / d).exists() else "STILL THERE"}  {d}')
