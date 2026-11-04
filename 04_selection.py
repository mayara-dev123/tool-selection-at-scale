"""[NOT USED IN THE PAPER - disabled, see the end of the file]

Step 4: experiment E4, how many retrieved candidates should the LLM see?

The best retriever (default dense_rerank, full library) gives its top-k
tools; each SELECTION_MODELS model picks one of them (or "none").
For each k we store: retrieval hit (gold in top-k), whether the model
picked a gold tool, and prompt tokens.

Candidates are shown in retrieval order and numbered 1..k, so tool ids
(which contain dataset names) are never shown to the model.

Usage:  python 04_selection.py [--retriever dense_rerank]
"""
import argparse
import json
import os
import random
import re

import config as C
from common import (JsonCache, ensure_dirs, key_of, load_data, ollama_chat,
                    read_jsonl, write_jsonl)

SYS = ("You are the tool-selection module of an AI agent. Given a user "
       "request and a numbered list of candidate tools, choose the single "
       "tool that should be called first to fulfil the request. If none of "
       "the tools fits, choose 0. Answer with JSON only: {\"choice\": <number>}")


def build_prompt(query, cands, texts):
    lines = [f"User request: {query}", "", "Candidate tools:"]
    for i, t in enumerate(cands, 1):
        lines.append(f"[{i}] {texts[t][:C.SELECTION_DESC_CHARS]}")
    return "\n".join(lines)


def parse_choice(out, k):
    try:
        c = int(json.loads(out)["choice"])
    except Exception:  # noqa: BLE001
        m = re.search(r"\d+", out)
        c = int(m.group()) if m else -1
    return c if 0 <= c <= k else -1   # -1 = invalid output


def main(retriever):
    ensure_dirs()
    queries, tools = load_data()
    texts = {t["id"]: t["text"] for t in tools}
    ret = {r["qid"]: r for r in read_jsonl(os.path.join(C.RESULTS_DIR, "retrieval.jsonl"))
           if r["method"] == retriever and r["pool"] == "full_s0"}
    rng = random.Random(C.SAMPLE_SEED)
    sub = rng.sample(queries, min(C.SELECTION_N_QUERIES, len(queries)))
    rows = []
    for model in C.SELECTION_MODELS:
        cache = JsonCache(f"select_{model.replace(':', '_')}.jsonl")
        print("model:", model)
        for qi, q in enumerate(sub):
            if qi % 50 == 0:
                print(f"  query {qi}/{len(sub)}")
            ranked = ret[q["id"]]["top100"]
            gold = set(q["gold"])
            for k in C.SELECTION_KS:
                cands = ranked[:k]
                prompt = build_prompt(q["query"], cands, texts)
                ck = key_of(model, SYS, prompt)
                v = cache.get(ck)
                if v is None:
                    try:
                        out, ptok, gtok = ollama_chat(model, prompt, system=SYS,
                                                      json_mode=True, max_tokens=64)
                    except Exception as e:  # noqa: BLE001
                        # counted as an invalid answer; not cached, so a rerun retries it
                        print(f"  selection failed for {q['id']} k={k}: {type(e).__name__}")
                        out, ptok, gtok = "", 0, 0
                    v = {"out": out, "ptok": ptok, "gtok": gtok}
                    if out:
                        cache.put(ck, v)
                c = parse_choice(v["out"], k)
                chosen = cands[c - 1] if c >= 1 else None
                rows.append({
                    "qid": q["id"], "model": model, "k": k,
                    "hit": int(bool(gold & set(cands))),
                    "correct": int(chosen in gold),
                    "abstain": int(c == 0), "invalid": int(c == -1),
                    "prompt_tokens": v["ptok"],
                })
    write_jsonl(os.path.join(C.RESULTS_DIR, "selection.jsonl"), rows)
    print("done:", len(rows), "rows")


# ---------------------------------------------------------------------------
# NOT USED IN THE PAPER: experiment E4 (number of candidates k) was removed
# from the paper, so this script is disabled. The security experiment
# (06_security.py) has its own selection step and does not need it.
# ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--retriever", default="dense_rerank")
#     main(ap.parse_args().retriever)
