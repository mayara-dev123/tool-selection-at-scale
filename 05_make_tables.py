"""Step 5: build the tables reported in the paper from the per-query results.

Tables written to tables/ (same layout and numbers as in the paper):
  table_scaling.tex             Recall@5 for each library size (+ loss)
  table_robustness.tex          Recall@5 before/after rewording, top tool changed
  table_security_retrieval.tex  malicious tool ranked first / in first five
  table_security_selection.tex  choice of the LLM with and without attack
It also prints every number quoted in the text of the paper, and the
largest 95% bootstrap confidence half-width of each table (used in the
captions), so that all values can be checked.

The scaling figure of the paper is drawn with pgfplots directly from the
values of table_scaling.tex.

Tables of experiments that are NOT in the paper (E1 method comparison on the
full library with NDCG/MRR/latency, the NDCG scaling table, and E4 number of
candidates) are kept at the end of this file as comments.
"""
import os
import random
from collections import defaultdict

import config as C
from common import bootstrap_ci, read_jsonl

NAMES = {"bm25": "BM25", "dense": "Dense (bi-encoder)",
         "dense_toolllm": "ToolLLM retriever", "dense_toolret": "ToolRet-trained dense",
         "dense_rerank": "Dense + cross-encoder", "llm_rewrite": "LLM rewrite + dense"}
ORDER = ["bm25", "dense", "dense_toolllm", "dense_toolret", "dense_rerank", "llm_rewrite"]


def pct(x):
    return f"{100 * x:.1f}\\%"


def save(name, lines):
    path = os.path.join(C.TABLES_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)


def half_width(values):
    _, lo, hi = bootstrap_ci(values)
    return 100 * (hi - lo) / 2


def mean(v):
    return sum(v) / len(v)


# ---------------------------------------------------------------- scaling
def table_scaling(rows):
    """Recall@5 per library size; each query is first averaged over seeds."""
    acc = defaultdict(lambda: defaultdict(list))
    subset = {}
    for r in rows:
        acc[(r["method"], r["pool"].split("_s")[0])][r["qid"]].append(r["recall@5"])
        subset[r["qid"]] = r["subset"]
    sizes = [str(s) for s in C.POOL_SIZES]
    vals, hw = {}, 0.0
    for m in ORDER:
        for s in sizes:
            per_q = [mean(v) for v in acc[(m, s)].values()]
            vals[(m, s)] = round(100 * mean(per_q), 1)
            hw = max(hw, half_width(per_q))
    best = {s: max(vals[(m, s)] for m in ORDER) for s in sizes}
    lines = ["\\begin{tabular}{lccccc}", "\\toprule",
             "\\textbf{Method} & \\textbf{100 tools} & \\textbf{1,000 tools} & "
             "\\textbf{10,000 tools} & \\textbf{Full (44,453)} & \\textbf{Loss} \\\\",
             "\\midrule"]
    for m in ORDER:
        cells = []
        for s in sizes:
            c = f"{vals[(m, s)]:.1f}\\%"
            cells.append(f"\\textbf{{{c}}}" if vals[(m, s)] == best[s] else c)
        loss = vals[(m, sizes[-1])] - vals[(m, sizes[0])]
        lines.append(" & ".join([NAMES[m]] + cells + [f"${loss:.1f}$"]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    save("table_scaling.tex", lines)
    print(f"  [caption] largest CI half-width: {hw:.1f} points")
    print("  [figure] pgfplots coordinates:")
    for m in ORDER:
        xs = [100, 1000, 10000, 44453]
        print(f"    {NAMES[m]}: " + " ".join(f"({x},{vals[(m, s)]})" for x, s in zip(xs, sizes)))
    # numbers quoted in the text
    full = {m: {q: mean(v) for q, v in acc[(m, "full")].items()} for m in ORDER}
    tb = [q for q in full["dense"] if subset[q] in ("toolbench", "toolbench-sam")]
    other = [q for q in full["dense"] if q not in set(tb)]
    print(f"  [text] ToolBench queries: {len(tb)}; ToolLLM {100 * mean([full['dense_toolllm'][q] for q in tb]):.1f}% "
          f"vs dense {100 * mean([full['dense'][q] for q in tb]):.1f}%")
    print(f"  [text] other sources: ToolLLM {100 * mean([full['dense_toolllm'][q] for q in other]):.1f}% "
          f"vs dense {100 * mean([full['dense'][q] for q in other]):.1f}%")
    for s in (sizes[0], sizes[-1]):
        print(f"  [text] size {s}: dense - BM25 = {vals[('dense', s)] - vals[('bm25', s)]:.1f} points")


# ---------------------------------------------------------------- robustness
def table_robustness(rows):
    rows = [r for r in rows if r.get("pert_ok", 1)]   # drop failed perturbations
    lines = ["\\begin{tabular}{lccccc}", "\\toprule",
             " & & \\multicolumn{2}{c}{\\textbf{User request reworded}} & "
             "\\multicolumn{2}{c}{\\textbf{Tool description reworded}} \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
             "\\textbf{Method} & \\textbf{Recall@5 before} & \\textbf{Recall@5 after} & "
             "\\textbf{Top tool changed} & \\textbf{Recall@5 after} & \\textbf{Top tool changed} \\\\",
             "\\midrule"]
    hw = 0.0
    for m in ORDER:
        qp = [r for r in rows if r["method"] == m and r["condition"] == "query_paraphrase"]
        dr = [r for r in rows if r["method"] == m and r["condition"] == "desc_rewrite"]
        cols = [[r["clean_recall@5"] for r in qp], [r["pert_recall@5"] for r in qp],
                [r["top1_flip"] for r in qp], [r["pert_recall@5"] for r in dr],
                [r["top1_flip"] for r in dr]]
        hw = max([hw] + [half_width(c) for c in cols])
        lines.append(" & ".join([NAMES[m]] + [pct(mean(c)) for c in cols]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    save("table_robustness.tex", lines)
    print(f"  [caption] largest CI half-width: {hw:.1f} points")


# ---------------------------------------------------------------- security
def table_security(rows, srows):
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             " & \\multicolumn{2}{c}{\\textbf{Competing clone}} & "
             "\\multicolumn{2}{c}{\\textbf{Shadow query}} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "\\textbf{Method} & \\textbf{Ranked first} & \\textbf{In first five} & "
             "\\textbf{Ranked first} & \\textbf{In first five} \\\\",
             "\\midrule"]
    worst_drop = 0.0
    for m in ORDER:
        cells = []
        for a in ("clone", "shadow"):
            rs = [r for r in rows if r["method"] == m and r["attack"] == a]
            cells += [pct(mean([r["poison_top1"] for r in rs])),
                      pct(mean([r["poison_top5"] for r in rs]))]
            d = mean([r["pert_recall@5"] - r["clean_recall@5"] for r in rs])
            worst_drop = min(worst_drop, 100 * d)
        lines.append(" & ".join([NAMES[m]] + cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    save("table_security_retrieval.tex", lines)
    print(f"  [text] largest change of Recall@5 under attack: {worst_drop:.1f} points")

    if not srows:
        return
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             "\\textbf{Situation} & \\textbf{\\shortstack{Malicious tool\\\\among the five}} & "
             "\\textbf{\\shortstack{LLM chose the\\\\malicious tool}} & "
             "\\textbf{\\shortstack{LLM chose a\\\\correct tool}} \\\\",
             "\\midrule"]
    hw = 0.0
    for a, name in (("none", "No attack"), ("clone", "Competing clone"),
                    ("shadow", "Shadow query")):
        rs = [r for r in srows if r["attack"] == a]
        cols = [[r["poison_in_list"] for r in rs], [r["chose_poison"] for r in rs],
                [r["chose_gold"] for r in rs]]
        hw = max([hw] + [half_width(c) for c in cols])
        cells = ["--", "--", pct(mean(cols[2]))] if a == "none" else [pct(mean(c)) for c in cols]
        lines.append(" & ".join([name] + cells) + " \\\\")
        both = [r for r in rs if r["poison_in_list"] and r["gold_in_list"]]
        if both:
            print(f"  [text] {name}: correct and malicious both in list for {len(both)} queries; "
                  f"chose malicious {100 * mean([r['chose_poison'] for r in both]):.0f}%, "
                  f"chose correct {100 * mean([r['chose_gold'] for r in both]):.0f}%")
    lines += ["\\bottomrule", "\\end{tabular}"]
    save("table_security_selection.tex", lines)
    print(f"  [caption] largest CI half-width: {hw:.1f} points")


def main():
    os.makedirs(C.TABLES_DIR, exist_ok=True)
    p = lambda n: os.path.join(C.RESULTS_DIR, n)  # noqa: E731
    print("== scaling (E2)")
    table_scaling(read_jsonl(p("retrieval.jsonl")))
    print("== robustness (E3)")
    table_robustness(read_jsonl(p("robustness.jsonl")))
    print("== security (E5)")
    sel = p("security_selection.jsonl")
    table_security(read_jsonl(p("security_retrieval.jsonl")),
                   read_jsonl(sel) if os.path.exists(sel) else [])


if __name__ == "__main__":
    main()


# ===========================================================================
# NOT USED IN THE PAPER (kept as comments for transparency):
#   table_e1           method comparison on the full library (R@k, NDCG, MRR, latency)
#   table_and_fig_e2   was also called with NDCG@10 (only Recall@5 is reported)
#   table_and_fig_e4   number of candidates shown to the LLM (experiment E4)
# Note: in these tables "mrr" was MRR@100 and the latency of llm_rewrite
# counted the LLM time twice on a cache miss (both fixed in common.py and
# 02_retrieve.py, but these values are not reported).
# ===========================================================================
# def table_e1(rows):
#     metrics = [f"recall@{k}" for k in C.TOP_KS] + [f"ndcg@{C.NDCG_K}", "mrr"]
#     full = [r for r in rows if r["pool"] == "full_s0"]
#     methods = [m for m in ORDER if any(r["method"] == m for r in full)]
#     label = lambda x: x.replace("recall", "R").replace("ndcg", "NDCG").upper() \
#         if x != "mrr" else "MRR"  # noqa: E731
#     head = " & ".join(["Method"] + [label(m) for m in metrics]
#                       + ["Median latency (ms)"])
#     lines = ["\\begin{tabular}{l" + "c" * (len(metrics) + 1) + "}", "\\toprule",
#              head + " \\\\", "\\midrule"]
#     for m in methods:
#         rs = [r for r in full if r["method"] == m]
#         cells = [fmt(*bootstrap_ci([r[x] for r in rs])) for x in metrics]
#         lat = sorted(r["latency_ms"] for r in rs)
#         cells.append(f"{lat[len(lat)//2]:.0f}")
#         lines.append(" & ".join([NAMES[m]] + cells) + " \\\\")
#     lines += ["\\bottomrule", "\\end{tabular}"]
#     save("table_e1_methods.tex", "\n".join(lines) + "\n")
#
#
# def per_query_over_seeds(rows, metric):
#     """{(method, size): {qid: mean over seeds}}"""
#     acc = defaultdict(lambda: defaultdict(list))
#     for r in rows:
#         acc[(r["method"], size_of(r["pool"]))][r["qid"]].append(r[metric])
#     return {k: {q: sum(v) / len(v) for q, v in d.items()} for k, d in acc.items()}
#
#
# def table_and_fig_e2(rows, metric="recall@5"):
#     agg = per_query_over_seeds(rows, metric)
#     sizes = [str(s) for s in C.POOL_SIZES]
#     methods = [m for m in ORDER if any(k[0] == m for k in agg)]
#     lines = ["\\begin{tabular}{l" + "c" * len(sizes) + "}", "\\toprule",
#              " & ".join(["Method"] + [f"$N={s}$" if s != "full" else "Full"
#                                       for s in sizes]) + " \\\\", "\\midrule"]
#     curves = {}
#     for m in methods:
#         cells, ys = [], []
#         for s in sizes:
#             vals = list(agg.get((m, s), {}).values())
#             mean, lo, hi = bootstrap_ci(vals)
#             cells.append(fmt(mean, lo, hi))
#             ys.append((mean, lo, hi))
#         curves[m] = ys
#         lines.append(" & ".join([NAMES[m]] + cells) + " \\\\")
#     lines += ["\\bottomrule", "\\end{tabular}"]
#     save(f"table_e2_scaling_{metric.replace('@', '')}.tex", "\n".join(lines) + "\n")
#
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     n_full = len(read_jsonl(os.path.join(C.DATA_DIR, "tools.jsonl")))
#     xs = [int(s) if s != "full" else n_full for s in sizes]
#     fig, ax = plt.subplots(figsize=(4.6, 3.2))
#     for m, ys in curves.items():
#         ax.plot(xs, [100 * y[0] for y in ys], marker="o", label=NAMES[m])
#         ax.fill_between(xs, [100 * y[1] for y in ys], [100 * y[2] for y in ys],
#                         alpha=0.15)
#     ax.set_xscale("log")
#     ax.set_xlabel("Tool library size (log scale)")
#     ax.set_ylabel(metric.replace("recall", "Recall") + " (%)")
#     ax.grid(alpha=0.3)
#     ax.legend(fontsize=7, frameon=False)
#     fig.tight_layout()
#     path = os.path.join(C.TABLES_DIR, f"fig_e2_scaling_{metric.replace('@', '')}.pdf")
#     fig.savefig(path)
#     print("wrote", path)
#
#
# def table_and_fig_e4(rows):
#     models = sorted({r["model"] for r in rows})
#     lines = ["\\begin{tabular}{lccccc}", "\\toprule",
#              "Model & $k$ & Gold in top-$k$ & Selection acc. & Acc. $\\mid$ gold in list & Prompt tokens \\\\",
#              "\\midrule"]
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots(figsize=(4.6, 3.2))
#     for mo in models:
#         ys = []
#         for k in C.SELECTION_KS:
#             rs = [r for r in rows if r["model"] == mo and r["k"] == k]
#             hit = bootstrap_ci([r["hit"] for r in rs])
#             acc = bootstrap_ci([r["correct"] for r in rs])
#             cond = bootstrap_ci([r["correct"] for r in rs if r["hit"]])
#             tok = sum(r["prompt_tokens"] for r in rs) / len(rs)
#             lines.append(f"{mo} & {k} & {fmt(*hit)} & {fmt(*acc)} & "
#                          f"{fmt(*cond)} & {tok:.0f} \\\\")
#             ys.append(acc)
#         lines.append("\\midrule")
#         ax.plot(C.SELECTION_KS, [100 * y[0] for y in ys], marker="o", label=mo)
#         ax.fill_between(C.SELECTION_KS, [100 * y[1] for y in ys],
#                         [100 * y[2] for y in ys], alpha=0.15)
#     lines[-1] = "\\bottomrule"
#     lines.append("\\end{tabular}")
#     save("table_e4_selection.tex", "\n".join(lines) + "\n")
#     ax.set_xlabel("Candidates shown to the LLM ($k$)")
#     ax.set_ylabel("Selection accuracy (%)")
#     ax.set_xticks(C.SELECTION_KS)
#     ax.grid(alpha=0.3)
#     ax.legend(fontsize=7, frameon=False)
#     fig.tight_layout()
#     path = os.path.join(C.TABLES_DIR, "fig_e4_selection.pdf")
#     fig.savefig(path)
#     print("wrote", path)
