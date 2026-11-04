"""Step 3: experiment E3, robustness under meaning-preserving perturbations.

Two perturbations, each generated once by REWRITE_MODEL and cached:
  query_paraphrase   the user request is reworded
  desc_rewrite       the documentation of each gold tool is reworded
                     (distractor tools are left untouched)

Run at library size ROBUSTNESS_POOL_SIZE, seed ROBUSTNESS_SEED. For each
method we store clean vs perturbed metrics and whether the top-1 changed.
The perturbations are saved in results/perturbations.jsonl so they can be
inspected and released with the paper.
"""
import argparse
import os

import numpy as np

import config as C
from common import (JsonCache, ensure_dirs, key_of, load_data, load_pools,
                    ollama_chat, per_query_metrics, pool_key, write_jsonl)
from retrievers import BM25, Dense, LLMRewrite, Reranker, truncate

ALL = ["bm25", "dense", "dense_rerank", "llm_rewrite"] + list(C.TOOL_DENSE)

PARA_SYS = ("Paraphrase the user request. Keep exactly the same meaning, "
            "every entity, number and constraint. Change the wording and "
            "sentence structure. Output only the paraphrase.")
DESC_SYS = ("Rewrite this tool documentation so that it has exactly the same "
            "meaning and the same function name and parameter names, but uses "
            "different wording for every natural-language description. Keep "
            "the same format (JSON stays valid JSON). Output only the "
            "rewritten documentation.")


FAILED = []  # (qid, kind) pairs whose perturbation could not be generated
REWRITE_FAILED = set()  # queries where the llm_rewrite expansion failed


def perturb(cache, system, text, qid, kind):
    """Returns (perturbed_text, ok). On a persistent LLM failure the original
    text is returned with ok=False and the query is excluded from E3."""
    k = key_of(C.REWRITE_MODEL, system, text)
    v = cache.get(k)
    if v is None:
        try:
            v, _, _ = ollama_chat(C.REWRITE_MODEL, text, system=system,
                                  max_tokens=700)
        except Exception as e:  # noqa: BLE001
            print(f"  perturbation failed for {qid} ({kind}): {type(e).__name__}")
            FAILED.append((qid, kind))
            return text, False
        v = v.strip()
        if not v:
            FAILED.append((qid, kind))
            return text, False
        cache.put(k, v)
    return v, True


def main(methods):
    ensure_dirs()
    queries, tools = load_data()
    pools = load_pools()
    pk = pool_key(C.ROBUSTNESS_POOL_SIZE, C.ROBUSTNESS_SEED)
    pool = pools[pk]
    tool_ids = [t["id"] for t in tools]
    tool_texts = [t["text"] for t in tools]
    id2idx = {t: i for i, t in enumerate(tool_ids)}

    bm25 = BM25(tool_texts) if "bm25" in methods else None
    dense = Dense(tool_ids, tool_texts)
    rer = Reranker(dense, tool_texts) if "dense_rerank" in methods else None
    llm = LLMRewrite(dense) if "llm_rewrite" in methods else None
    tdense = {m: Dense(tool_ids, tool_texts, model=mo, prefix=pre)
              for m, (mo, pre) in C.TOOL_DENSE.items() if m in methods}
    cache = JsonCache(f"perturb_{C.REWRITE_MODEL.replace(':', '_')}.jsonl")

    pert_rows, rows = [], []
    for qi, q in enumerate(queries):
        if qi % 50 == 0:
            print(f"query {qi}/{len(queries)}")
        idx = np.array([id2idx[t] for t in pool[q["id"]]])
        gold_pos = [j for j, t in enumerate(pool[q["id"]]) if t in set(q["gold"])]
        para, ok_para = perturb(cache, PARA_SYS, q["query"], q["id"], "query")
        new_docs, ok_desc = {}, True
        for j in gold_pos:
            d, ok = perturb(cache, DESC_SYS, truncate(tool_texts[idx[j]]),
                            q["id"], "desc")
            new_docs[j] = d
            ok_desc = ok_desc and ok
        pert_rows.append({"qid": q["id"], "query": q["query"],
                          "paraphrase": para,
                          "gold_rewrites": {pool[q["id"]][j]: d
                                            for j, d in new_docs.items()}})

        def scores(m, text, docs_override):
            """Scores over the pool, optionally with gold docs replaced."""
            if m == "bm25":
                s = bm25.full_scores(q["id"], text)[idx].copy()
                for j, d in docs_override.items():
                    s[j] = bm25.score_new_doc(text, d)
                return s, text
            if m == "llm_rewrite":
                text = text + "\n" + llm.rewrite(text)
                if not llm.last_ok:
                    REWRITE_FAILED.add(q["id"])
            enc = tdense.get(m, dense)
            qv = enc.encode_query(text)
            s = enc.doc_emb[idx] @ qv
            if docs_override:
                js = list(docs_override)
                s[js] = enc.encode_docs([docs_override[j] for j in js]) @ qv
            return s, text

        def rank(m, text, docs_override):
            s, qtext = scores(m, text, docs_override)
            if m != "dense_rerank":
                return np.argsort(-s)
            order = np.argsort(-s)
            head = order[:C.RERANK_DEPTH]
            pairs = [(qtext, truncate(docs_override.get(int(j), tool_texts[idx[j]])))
                     for j in head]
            ce = rer.ce.predict(pairs, batch_size=C.BATCH_SIZE)
            return np.concatenate([head[np.argsort(-ce)], order[C.RERANK_DEPTH:]])

        for m in methods:
            ids = pool[q["id"]]
            clean = [ids[j] for j in rank(m, q["query"], {})[:100]]
            base = per_query_metrics(clean, q["gold"])
            for cond, text, ov, ok in (("query_paraphrase", para, {}, ok_para),
                                       ("desc_rewrite", q["query"], new_docs, ok_desc)):
                pert = [ids[j] for j in rank(m, text, ov)[:100]]
                pm = per_query_metrics(pert, q["gold"])
                rows.append({
                    "qid": q["id"], "method": m, "condition": cond,
                    **{f"clean_{k}": v for k, v in base.items()},
                    **{f"pert_{k}": v for k, v in pm.items()},
                    "top1_flip": int(clean[0] != pert[0]),
                    "pert_ok": int(ok and not (m == "llm_rewrite"
                                               and q["id"] in REWRITE_FAILED)),
                })

    write_jsonl(os.path.join(C.RESULTS_DIR, "perturbations.jsonl"), pert_rows)
    out = os.path.join(C.RESULTS_DIR, "robustness.jsonl")
    if os.path.exists(out):  # keep rows of methods not rerun now
        from common import read_jsonl
        rows = [r for r in read_jsonl(out) if r["method"] not in methods] + rows
    write_jsonl(out, rows)
    print("done:", len(rows), "rows;", len(FAILED),
          "perturbations failed (excluded from E3):", FAILED[:10])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=ALL, choices=ALL)
    main(ap.parse_args().methods)
