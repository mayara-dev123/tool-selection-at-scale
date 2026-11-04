"""Step 2: experiments E1 (method comparison) and E2 (library size).

For every query, method, and library size (pool), ranks the candidate
tools and stores per-query metrics in results/retrieval.jsonl.
Latency is measured on the full library only. Latency is not reported in the
paper (it was only used by the E1 table, now commented out in 05_make_tables.py).

Usage:  python 02_retrieve.py [--methods bm25 dense dense_rerank llm_rewrite
                                         dense_toolllm dense_toolret]
"""
import argparse
import os
import time

import numpy as np

import config as C
from common import (ensure_dirs, load_data, load_pools, per_query_metrics,
                    write_jsonl)
from retrievers import BM25, Dense, LLMRewrite, Reranker

ALL = ["bm25", "dense", "dense_rerank", "llm_rewrite"] + list(C.TOOL_DENSE)


def build(methods, tool_ids, tool_texts):
    r = {}
    if "bm25" in methods:
        t0 = time.time()
        r["bm25"] = BM25(tool_texts)
        print(f"bm25 index built in {time.time() - t0:.0f}s")
    if any(m in methods for m in ("dense", "dense_rerank", "llm_rewrite")):
        r["dense"] = Dense(tool_ids, tool_texts)
    if "dense_rerank" in methods:
        r["dense_rerank"] = Reranker(r["dense"], tool_texts)
    if "llm_rewrite" in methods:
        r["llm_rewrite"] = LLMRewrite(r["dense"])
    for m, (model, prefix) in C.TOOL_DENSE.items():
        if m in methods:
            r[m] = Dense(tool_ids, tool_texts, model=model, prefix=prefix)
    return r


def pool_indices(pool, qid, id2idx, n_tools):
    if pool is None:
        return np.arange(n_tools)
    return np.array([id2idx[t] for t in pool[qid]])


def run(methods):
    ensure_dirs()
    queries, tools = load_data()
    pools = load_pools()
    tool_ids = [t["id"] for t in tools]
    tool_texts = [t["text"] for t in tools]
    id2idx = {t: i for i, t in enumerate(tool_ids)}
    R = build(methods, tool_ids, tool_texts)
    # full library first, so latency is measured before any caching
    pool_keys = sorted(pools, key=lambda k: (k != "full_s0", k))

    rows = []
    for qi, q in enumerate(queries):
        if qi % 50 == 0:
            print(f"query {qi}/{len(queries)}")
        full = {}
        lat = {}
        for m in methods:
            base = "dense" if m == "dense_rerank" else m
            if m == "llm_rewrite":
                # FIX: the rewrite is generated (or read from the cache) outside
                # the timer, and its original generation time is added once.
                # Before, on a cache miss the LLM time was counted twice.
                rw = R[m].rewrite(q["query"])
                t0 = time.perf_counter()
                full[m] = R["dense"].full_scores(q["id"], q["query"] + "\n" + rw)
                lat[m] = time.perf_counter() - t0 + R[m].last_llm_sec
            else:
                t0 = time.perf_counter()
                full[m] = R[base].full_scores(q["id"], q["query"])
                lat[m] = time.perf_counter() - t0
        for pk in pool_keys:
            idx = pool_indices(pools[pk], q["id"], id2idx, len(tool_ids))
            for m in methods:
                s = full[m][idx]
                t0 = time.perf_counter()
                if m == "dense_rerank":
                    order = R[m].rank(s, idx, q["query"])
                else:
                    order = np.argsort(-s)
                extra = time.perf_counter() - t0
                ranked = [tool_ids[idx[j]] for j in order[:100]]
                row = {"qid": q["id"], "subset": q["subset"], "method": m,
                       "pool": pk, **per_query_metrics(ranked, q["gold"]),
                       "top10": ranked[:10]}
                if pk == "full_s0":
                    row["latency_ms"] = 1000 * (lat[m] + extra)
                    row["top100"] = ranked  # used by 04_selection.py
                rows.append(row)

    out = os.path.join(C.RESULTS_DIR, "retrieval.jsonl")
    if os.path.exists(out):  # keep rows of methods not rerun now
        from common import read_jsonl
        rows = [r for r in read_jsonl(out) if r["method"] not in methods] + rows
    write_jsonl(out, rows)
    print("wrote", out, len(rows), "rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=ALL, choices=ALL)
    run(ap.parse_args().methods)
