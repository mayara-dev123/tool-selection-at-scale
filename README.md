# Tool selection at scale: supporting experiments

Code and results of the supporting experiments of the review paper
*"Tool selection at scale: How large language model agents choose the right
tool from large libraries"* (Section "Gaps and future directions").

The experiments use the public ToolRet benchmark (44,453 tools) and test three
claims of the review:

| Experiment | Question | Script | Paper table |
|---|---|---|---|
| Scaling | How does retrieval degrade as the library grows (100, 1k, 10k, 44k tools)? | `02_retrieve.py` | Recall@5 for each library size (+ figure) |
| Robustness | Does the first retrieved tool change when the request or the tool description is reworded? | `03_robustness.py` | Robustness on the 10,000-tool library |
| Security | Is one malicious tool retrieved and chosen by the LLM? | `06_security.py` | Tool poisoning (retrieval and LLM choice) |

Six retrievers: BM25, bge-base-en-v1.5, bge-base + bge-reranker-base,
a simplified Re-Invoke (Qwen3-14B query rewrite + dense), the ToolLLM retriever,
and bge-base trained on ToolRet. LLM: Qwen3-14B through Ollama, temperature 0.

## 1. Check the tables (1 minute, no GPU)
```bash
unzip results.zip            # creates results/
pip install numpy
python 05_make_tables.py
```
This rebuilds the four tables of the paper in `tables/` from the per-query
results and prints every number quoted in the text.

`results.zip` contains:
- `retrieval.jsonl`: one line per query, method and library size
- `robustness.jsonl`, `perturbations.jsonl`: robustness results and the reworded texts
- `security_retrieval.jsonl`, `security_selection.jsonl`, `poison_tools.jsonl`: security results and the generated malicious tools

## 2. Run everything again (GPU + Ollama)
```bash
pip install -r requirements.txt
ollama pull qwen3:14b
python download_on_pc.py        # data and models into hf_local/ (optional, for offline use)
python 00_check_env.py
python 01_prepare_data.py       # sample of 1,000 queries (seed 42) and libraries
python 02_retrieve.py           # scaling
python 03_robustness.py         # robustness
python 06_security.py           # security
python 05_make_tables.py        # tables
```
All settings (seeds, models, library sizes) are in `config.py`. LLM outputs
are cached, and the generated texts used in the paper are released in
`results.zip`, so the tables can be rebuilt exactly even if an LLM produces
slightly different text on other hardware.

`04_selection.py` (number of candidates shown to the LLM) is an additional
experiment that is not reported in the paper and is disabled.

## License
MIT
