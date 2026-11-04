"""Step 1: download ToolRet, draw the stratified query sample, build pools.

Outputs (in DATA_DIR):
  tools.jsonl            {"id", "text"} for every tool in the corpus
  queries_sample.jsonl   {"id", "subset", "query", "gold"} for N_QUERIES queries
  pools.json             candidate tool ids per (library size, seed, query)
  data_stats.json        counts reported in the paper's setup section
"""
import json
import os
import random
from collections import defaultdict

import config as C
from common import ensure_dirs, write_jsonl

DOC_KEYS = ["documentation", "doc", "text", "description", "content"]


def doc_to_text(doc):
    if isinstance(doc, str):
        return doc
    return json.dumps(doc, ensure_ascii=False)


def load_tools():
    from datasets import get_dataset_config_names, load_dataset
    tools = {}
    configs = get_dataset_config_names(C.TOOLS_DATASET)
    print("tool configs:", configs)
    for cfg in configs:
        dd = load_dataset(C.TOOLS_DATASET, cfg)
        for split in dd:
            ds = dd[split]
            print(f"  {cfg}/{split}: {len(ds)} rows, fields={ds.column_names}")
            key = next((k for k in DOC_KEYS if k in ds.column_names), None)
            if key is None:
                raise SystemExit(f"No documentation field in {cfg}: "
                                 f"{ds.column_names}. Add it to DOC_KEYS.")
            for row in ds:
                tools[row["id"]] = doc_to_text(row[key])
    return tools


def load_queries():
    from datasets import get_dataset_config_names, load_dataset
    rows = []
    for subset in get_dataset_config_names(C.QUERIES_DATASET):
        ds = load_dataset(C.QUERIES_DATASET, subset, split="queries")
        for r in ds:
            labels = r["labels"]
            if isinstance(labels, str):
                labels = json.loads(labels)
            gold = [l["id"] for l in labels
                    if float(l.get("relevance", 1)) > 0]
            q = r["query"]
            if C.USE_INSTRUCTION and r.get("instruction"):
                q = r["instruction"].strip() + "\n" + q
            rows.append({"id": r["id"], "subset": subset, "query": q,
                         "gold": gold})
    return rows


def stratified_sample(rows, n, min_per, seed):
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["subset"]].append(r)
    total = len(rows)
    quota = {s: max(min_per, round(n * len(v) / total)) for s, v in by.items()}
    quota = {s: min(q, len(by[s])) for s, q in quota.items()}
    # trim the largest quotas until we hit n exactly
    while sum(quota.values()) > n:
        s = max(quota, key=lambda x: quota[x])
        quota[s] -= 1
    # top up (rounding can leave us short) from subsets with spare queries
    while sum(quota.values()) < min(n, total):
        s = max((x for x in quota if quota[x] < len(by[x])),
                key=lambda x: len(by[x]) - quota[x])
        quota[s] += 1
    out = []
    for s in sorted(by):
        out += rng.sample(by[s], quota[s])
    return out


def build_pools(queries, tool_ids):
    pools = {}
    all_ids = sorted(tool_ids)
    for size in C.POOL_SIZES:
        if size == "full":
            pools["full_s0"] = None  # None = whole corpus
            continue
        for seed in C.POOL_SEEDS:
            rng = random.Random(f"{seed}-{size}")
            per_q = {}
            for q in queries:
                gold = set(q["gold"])
                need = max(0, size - len(gold))
                distract = []
                while len(distract) < need:
                    for t in rng.sample(all_ids, need - len(distract)):
                        if t not in gold:
                            distract.append(t)
                    distract = list(dict.fromkeys(distract))
                per_q[q["id"]] = sorted(gold) + distract
            pools[f"{size}_s{seed}"] = per_q
    return pools


def main():
    ensure_dirs()
    tools = load_tools()
    print("tools:", len(tools))
    rows = load_queries()
    print("queries (all):", len(rows))
    valid = [r for r in rows if r["gold"] and all(g in tools for g in r["gold"])]
    print("queries with all gold tools in corpus:", len(valid))
    sample = stratified_sample(valid, C.N_QUERIES, C.MIN_PER_SUBSET,
                               C.SAMPLE_SEED)
    print("sampled:", len(sample))

    write_jsonl(os.path.join(C.DATA_DIR, "tools.jsonl"),
                [{"id": k, "text": v} for k, v in tools.items()])
    write_jsonl(os.path.join(C.DATA_DIR, "queries_sample.jsonl"), sample)
    pools = build_pools(sample, tools.keys())
    with open(os.path.join(C.DATA_DIR, "pools.json"), "w") as f:
        json.dump(pools, f)

    per_subset = defaultdict(int)
    for r in sample:
        per_subset[r["subset"]] += 1
    stats = {
        "n_tools": len(tools),
        "n_queries_all": len(rows),
        "n_queries_valid": len(valid),
        "n_queries_sampled": len(sample),
        "avg_gold_per_query": sum(len(r["gold"]) for r in sample) / len(sample),
        "per_subset": dict(per_subset),
        "sample_seed": C.SAMPLE_SEED,
    }
    with open(os.path.join(C.DATA_DIR, "data_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "per_subset"},
                     indent=2))


if __name__ == "__main__":
    main()
