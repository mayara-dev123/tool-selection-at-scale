"""Step 0: check the server before running anything.

Prints GPU / VRAM, Python packages, Ollama models, and whether the
configured models are available. Send the output to adjust config.py.
"""
import importlib
import json
import subprocess
import sys
import urllib.request

import config as C

print("python:", sys.version.split()[0])

try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
         "--format=csv,noheader"], capture_output=True, text=True, check=True)
    print("GPU:", out.stdout.strip())
except Exception as e:  # noqa: BLE001
    print("GPU: nvidia-smi not available ->", e)

for pkg in ["torch", "datasets", "sentence_transformers",
            "numpy", "matplotlib"]:
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg}: {getattr(m, '__version__', 'ok')}")
    except ImportError:
        print(f"{pkg}: MISSING  (pip install -r requirements.txt)")

try:
    import torch
    print("torch.cuda.is_available():", torch.cuda.is_available())
except ImportError:
    pass

try:
    with urllib.request.urlopen(C.OLLAMA_URL + "/api/tags", timeout=10) as r:
        models = [m["name"] for m in json.loads(r.read())["models"]]
    print("ollama models:", models)
    for m in [C.REWRITE_MODEL] + C.SELECTION_MODELS:
        ok = any(x == m or x.startswith(m + ":") or x == m + ":latest"
                 for x in models)
        print(f"  {m}: {'OK' if ok else 'NOT PULLED -> ollama pull ' + m}")
except Exception as e:  # noqa: BLE001
    print("ollama: not reachable at", C.OLLAMA_URL, "->", e)

print("offline data/models (hf_local):", "FOUND" if C.OFFLINE else "MISSING -> run download_on_pc.py on the PC")
for name in ("QUERIES_DATASET", "TOOLS_DATASET", "EMBED_MODEL", "RERANK_MODEL"):
    print(f"  {name}: {getattr(C, name)}")
for m, (mo, _) in C.TOOL_DENSE.items():
    import os as _os
    print(f"  {m}: {mo}  [{'OK' if _os.path.isdir(mo) else 'MISSING'}]")

# quick Ollama test: deterministic answer without a <think> trace
try:
    from common import ollama_chat
    out, _, _ = ollama_chat(C.REWRITE_MODEL, "Reply with the single word OK.")
    print(f"ollama test ({C.REWRITE_MODEL}):", repr(out[:40]))
except Exception as e:  # noqa: BLE001
    print("ollama test failed ->", e)
