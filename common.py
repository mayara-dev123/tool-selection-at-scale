"""Shared helpers: IO, metrics, Ollama client, caching."""
import hashlib
import json
import math
import os
import random
import re
import time
import urllib.request

import config as C


# ---------------------------------------------------------------- IO
def ensure_dirs():
    for d in (C.DATA_DIR, C.RESULTS_DIR, C.TABLES_DIR, C.CACHE_DIR):
        os.makedirs(d, exist_ok=True)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_data():
    """Returns (queries, tools) prepared by 01_prepare_data.py."""
    queries = read_jsonl(os.path.join(C.DATA_DIR, "queries_sample.jsonl"))
    tools = read_jsonl(os.path.join(C.DATA_DIR, "tools.jsonl"))
    return queries, tools


def load_pools():
    with open(os.path.join(C.DATA_DIR, "pools.json"), encoding="utf-8") as f:
        return json.load(f)


def pool_key(size, seed):
    return f"{size}_s{seed}"


# ---------------------------------------------------------------- metrics
def per_query_metrics(ranked_ids, gold_ids):
    """Binary-relevance metrics for one query. ranked_ids is best-first."""
    gold = set(gold_ids)
    out = {}
    for k in C.TOP_KS:
        out[f"recall@{k}"] = len(gold & set(ranked_ids[:k])) / len(gold)
    dcg = sum(1.0 / math.log2(i + 2)
              for i, t in enumerate(ranked_ids[:C.NDCG_K]) if t in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), C.NDCG_K)))
    out[f"ndcg@{C.NDCG_K}"] = dcg / idcg
    # NOTE: callers pass the first 100 results only, so this value is MRR@100
    # (a query whose first correct tool is below rank 100 gets 0). MRR is not
    # reported in the paper (it was only used by the E1 table, now commented out).
    rr = 0.0
    for i, t in enumerate(ranked_ids):
        if t in gold:
            rr = 1.0 / (i + 1)
            break
    out["mrr@100"] = rr
    return out


def bootstrap_ci(values, iters=None, seed=0, alpha=0.05):
    """Mean and percentile bootstrap CI over queries."""
    iters = iters or C.BOOTSTRAP_ITERS
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(alpha / 2 * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return sum(values) / n, lo, hi


# ---------------------------------------------------------------- Ollama
def _cache_path(name):
    return os.path.join(C.CACHE_DIR, name)


class JsonCache:
    """Append-only key -> value cache, so interrupted runs resume."""

    def __init__(self, name):
        self.path = _cache_path(name)
        self.data = {}
        if os.path.exists(self.path):
            for r in read_jsonl(self.path):
                self.data[r["k"]] = r["v"]

    def get(self, k):
        return self.data.get(k)

    def put(self, k, v):
        self.data[k] = v
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")


def ollama_chat(model, prompt, system=None, json_mode=False, retries=3,
                max_tokens=None, timeout=None):
    """Deterministic single-turn call. Returns (text, prompt_tokens, gen_tokens).

    max_tokens caps the generation length, so a model that starts looping
    cannot block the run; timeout is per attempt (seconds)."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    body = {
        "model": model,
        "messages": msgs,
        "stream": False,
        "options": {"temperature": C.LLM_TEMPERATURE, "seed": C.LLM_SEED,
                    "num_predict": max_tokens or getattr(C, "LLM_MAX_TOKENS", 512)},
        "think": False,   # qwen3: disable the thinking trace
    }
    if json_mode:
        body["format"] = "json"
    data = json.dumps(body).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                C.OLLAMA_URL + "/api/chat", data=data,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout or getattr(C, "LLM_TIMEOUT", 180)) as r:
                out = json.loads(r.read())
            text = re.sub(r"<think>.*?</think>", "", out["message"]["content"],
                          flags=re.S).strip()
            return (text,
                    out.get("prompt_eval_count", 0),
                    out.get("eval_count", 0))
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"  ollama error ({e}); retrying")
            time.sleep(3)


def key_of(*parts):
    return hashlib.sha1("\x1f".join(map(str, parts)).encode()).hexdigest()
