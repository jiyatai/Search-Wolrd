"""从入口脚本出发做传递可达性分析，找出不可达模块（真正死代码）。"""
from __future__ import annotations

import ast
import os
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", "__pycache__", "vis_results", "tmp", "node_modules"}

py_files: list[Path] = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for fn in filenames:
        if fn.endswith(".py"):
            py_files.append(Path(dirpath) / fn)
py_files.sort()


def module_name(p: Path) -> str:
    rel = p.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


modules = {module_name(p): p for p in py_files}

# 每个模块的 import 目标（解析到仓库内模块）
deps: dict[str, set[str]] = {}
for m, p in modules.items():
    try:
        tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"), filename=str(p))
    except SyntaxError:
        deps[m] = set()
        continue
    out: set[str] = set()
    for node in ast.walk(tree):
        base = None
        names: list[str] = []
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name)
            continue
        if isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            base = node.module
            if node.level:
                pkg = m.split(".")[:-1]
                up = pkg[: len(pkg) - node.level + 1] if node.level > 1 else pkg
                base = ".".join(up + ([node.module] if node.module else []))
            names = [a.name for a in node.names]
        if base is None:
            continue
        for cand in [base] + [f"{base}.{n}" for n in names]:
            parts = cand.split(".")
            for i in range(len(parts), 0, -1):
                c = ".".join(parts[:i])
                if c in modules:
                    out.add(c)
                    break
    deps[m] = out

# 入口集合 = 顶层可执行脚本 + 显式列出的测试/工具
ENTRIES = [
    "train", "evaluate", "evaluate_uav", "train_stage3", "setup", "arg_parser",
    "onnx_conversion", "trt_conversion",
]
entries: set[str] = set()
for m in modules:
    if m in ENTRIES:
        entries.add(m)
    rel = modules[m].relative_to(ROOT)
    if rel.parts[0] not in ("model", "ros2_deployment"):
        entries.add(m)          # deploy/ scripts/ tools/ 全部当入口
    if m.startswith("ros2_deployment"):
        entries.add(m)
    if "smoke_test" in modules[m].name:
        entries.add(m)


def reachable(root_set: set[str]) -> set[str]:
    seen: set[str] = set()
    dq = deque(root_set)
    while dq:
        cur = dq.popleft()
        if cur in seen or cur not in modules:
            continue
        seen.add(cur)
        for d in deps.get(cur, ()):
            if d not in seen:
                dq.append(d)
    return seen


reach = reachable(entries)
dead = sorted(m for m in modules if m not in reach and not m.endswith("__init__") and modules[m].stem != "__init__")

print("=" * 78)
print(f"模块 {len(modules)} | 入口 {len(entries)} | 可达 {len(reach)} | 不可达 {len(dead)}")
print("=" * 78)
print("\n### 不可达模块（从任何入口触发都到不了）")
total = 0
for m in dead:
    p = modules[m]
    sz = p.stat().st_size
    total += sz
    print(f"  {sz:>7d}B  {m:60s} {p.relative_to(ROOT)}")
print(f"\n合计 {len(dead)} 个文件 / {total/1024:.1f} KB")

print("\n### 可达但仅被 1 个文件引用（弱连接，需人工确认）")
for m in sorted(modules):
    if m in reach and m not in entries and not m.endswith("__init__"):
        # 计算入度
        indeg = sum(1 for other, ds in deps.items() if m in ds)
        if indeg <= 1:
            owners = [o for o, ds in deps.items() if m in ds]
            print(f"  {modules[m].stat().st_size:>7d}B  {m:55s} <- {owners}")
