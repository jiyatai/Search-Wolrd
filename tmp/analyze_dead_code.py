"""静态引用分析：找出仓库内从未被引用到的模块（候选死代码）。

不考虑 runtime 反射（gin 绑定靠全限定名，单独用 grep 覆盖）。
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {".git", "__pycache__", "vis_results", "tmp", "node_modules"}

# ---------------- 1. 收集所有 .py 模块 ----------------
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
mod_by_basename: dict[str, list[str]] = defaultdict(list)
for m, p in modules.items():
    mod_by_basename[p.stem].append(m)

# ---------------- 2. 解析每个文件的 import ----------------
imports_of: dict[str, set[str]] = {}
for m, p in modules.items():
    try:
        tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"), filename=str(p))
    except SyntaxError as e:
        print(f"[SYNTAX FAIL] {p}: {e}")
        continue
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            base = node.module
            if node.level:  # relative
                pkg = m.split(".")[: -1 or None]
                depth = node.level
                up = pkg[: len(pkg) - depth + 1] if depth > 1 else pkg
                base = ".".join(up + ([node.module] if node.module else []))
            out.add(base)
            for a in node.names:
                out.add(f"{base}.{a.name}")
    imports_of[m] = out

# ---------------- 3. 反向引用计数 ----------------
referenced: dict[str, set[str]] = defaultdict(set)
for m, imp in imports_of.items():
    for target in imp:
        # 精确模块
        if target in modules:
            referenced[target].add(m)
            continue
        # from pkg import name -> pkg.name
        parts = target.split(".")
        for i in range(len(parts), 0, -1):
            cand = ".".join(parts[:i])
            if cand in modules:
                referenced[cand].add(m)
                break

# ---------------- 4. 文本级引用（gin / md / sh / 字符串） ----------------
text_refs: dict[str, list[str]] = defaultdict(list)
grep_targets = ["*.gin", "*.md", "*.sh", "*.toml", "*.cfg", "*.yaml", "*.yml", "*.txt", "Dockerfile"]
all_text_files: list[Path] = []
for pat in grep_targets:
    all_text_files += [q for q in ROOT.rglob(pat) if not any(s in q.parts for s in SKIP_DIRS)]
for q in sorted(set(all_text_files)):
    try:
        txt = q.read_text(encoding="utf-8", errors="replace")
    except Exception:
        continue
    for m, p in modules.items():
        stem = p.stem
        if stem == "__init__":
            continue
        # 匹配 model.searchworld.x 或 x.py 或 类名
        if stem in txt:
            text_refs[m].append(str(q.relative_to(ROOT)))

# ---------------- 5. 报告 ----------------
print("=" * 78)
print("模块总数:", len(modules))
print("=" * 78)

entry_scripts = []
unreferenced = []
for m, p in sorted(modules.items()):
    if p.stem == "__init__":
        continue
    refs = referenced.get(m, set())
    trefs = text_refs.get(m, [])
    if not refs and not trefs:
        unreferenced.append((m, p))
    if not refs:
        entry_scripts.append((m, p, trefs))

print("\n### A. 完全无引用（无 import、无文本提及） —— 强死代码候选")
if not unreferenced:
    print("  (无)")
for m, p in unreferenced:
    print(f"  {m:60s} {p.stat().st_size:>7d}B  {p}")

print("\n### B. 无 import 但有文本提及（脚本/工具类，仅靠命令行调用）")
for m, p, trefs in entry_scripts:
    if trefs:
        print(f"  {m:60s}  <- {', '.join(trefs[:3])}{' ...' if len(trefs) > 3 else ''}")

print("\n### C. 被 import 的模块（活代码）—— 仅列被引用次数")
live = [(m, len(referenced[m])) for m in modules if referenced.get(m) and m not in [x[0] for x in entry_scripts]]
live.sort(key=lambda x: -x[1])
for m, n in live:
    print(f"  {n:>3d}x  {m}")
