"""Central configuration for the tool-selection experiments.

Edit this file before running. Every script reads from here, so the
values below are exactly what gets reported in the paper.
"""
import os

# Hugging Face is not reachable from the server. Data and models are
# downloaded on a PC with download_on_pc.py into `hf_local/` and copied here.
# When that folder exists, everything is loaded from disk, fully offline.
HF_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_local")
OFFLINE = os.path.isdir(HF_LOCAL)
if OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def _src(repo):
    """Local copy if downloaded, otherwise the Hugging Face repo id."""
    local = os.path.join(HF_LOCAL, repo.split("/")[1])
    return local if os.path.isdir(local) else repo

# ---------------------------------------------------------------- paths
DATA_DIR = "data"          # downloaded + sampled ToolRet data
RESULTS_DIR = "results"    # per-query results (jsonl) and logs
TABLES_DIR = "tables"      # LaTeX tables and figures for the paper
CACHE_DIR = "cache"        # embeddings, LLM outputs (so reruns are free)

# ---------------------------------------------------------------- data
QUERIES_DATASET = _src("mangopy/ToolRet-Queries")
TOOLS_DATASET = _src("mangopy/ToolRet-Tools")
N_QUERIES = 1000           # stratified sample over ToolRet subsets
MIN_PER_SUBSET = 5         # every subset keeps at least this many queries
SAMPLE_SEED = 42
USE_INSTRUCTION = False    # False = query only (ToolRet "w/o inst." setting)

# Library sizes for the scalability experiment (E2). "full" = whole corpus.
POOL_SIZES = [100, 1000, 10000, "full"]
POOL_SEEDS = [0, 1, 2]     # distractor sampling seeds

# ---------------------------------------------------------------- retrievers
EMBED_MODEL = _src("BAAI/bge-base-en-v1.5")
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = _src("BAAI/bge-reranker-base")

# Published dense retrievers trained on tool data (used as released, no
# fine-tuning by us). method name -> (model, query prefix)
#   dense_toolllm : ToolLLM retriever (Qin et al., 2024), trained on ToolBench,
#                   i.e. on a different tool library (out-of-domain)
#   dense_toolret : bge-base fine-tuned on ToolRet-train (Shi et al., 2025),
#                   same base model as `dense`, same tool sources (in-domain);
#                   trained with the same bge query instruction
TOOL_DENSE = {
    "dense_toolllm": (_src("ToolBench/ToolBench_IR_bert_based_uncased"), ""),
    "dense_toolret": (_src("mangopy/ToolRet-trained-bge-base-en-v1.5"),
                      "Represent this sentence for searching relevant passages: "),
}
RERANK_DEPTH = 50          # reranker rescoring depth (top-50 from dense)
MAX_DOC_CHARS = 1500       # tool documentation is truncated to this length
BATCH_SIZE = 64
DEVICE = "cuda"            # "cpu" also works, only slower

# ---------------------------------------------------------------- LLM (Ollama)
OLLAMA_URL = "http://localhost:11434"
REWRITE_MODEL = "qwen3:14b"                     # query rewriting + perturbations
SELECTION_MODELS = ["qwen3:14b", "qwen3:32b"]   # E4 (both already on the server)
LLM_TEMPERATURE = 0.0
LLM_SEED = 1234
LLM_MAX_TOKENS = 512       # generation cap per call (stops runaway outputs)
LLM_TIMEOUT = 180          # seconds per attempt

# ---------------------------------------------------------------- experiments
TOP_KS = [1, 5, 10]                  # Recall@k
NDCG_K = 10
ROBUSTNESS_POOL_SIZE = 10000         # E3 is run at this library size
ROBUSTNESS_SEED = 0
SELECTION_N_QUERIES = 300            # E4 subset (sampled from the 1000)
SELECTION_KS = [1, 3, 5, 10, 20]
SELECTION_DESC_CHARS = 400           # per-tool text shown to the LLM
BOOTSTRAP_ITERS = 1000
