"""Run this on the PC (with VPN), NOT on the server.

Downloads everything the experiments need from Hugging Face into the
folder `hf_local/`. Then copy the whole `hf_local` folder to the server,
next to config.py. The server then works fully offline.

    pip install -U huggingface_hub
    python download_on_pc.py
"""
import os
import time

# Plain HTTP downloads instead of the Xet CDN (Xet breaks behind some VPNs).
# Must be set before huggingface_hub is imported.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

from huggingface_hub import list_repo_files, snapshot_download  # noqa: E402

OUT = "hf_local"

DATASETS = ["mangopy/ToolRet-Queries", "mangopy/ToolRet-Tools"]
MODELS = ["BAAI/bge-base-en-v1.5", "BAAI/bge-reranker-base",
          # published tool-trained retrievers (step 2 of the revision)
          "ToolBench/ToolBench_IR_bert_based_uncased",
          "mangopy/ToolRet-trained-bge-base-en-v1.5"]

# small config/tokenizer files + one copy of the weights (no ONNX, no duplicates)
MODEL_FILES = ["*.json", "*.txt", "*.model", "1_Pooling/*"]


def retry(fn, tries=5):
    """Re-run a download after a network error; finished files are skipped."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            print(f"  network error ({type(e).__name__}); retry {i + 1}/{tries - 1} in 10 s")
            time.sleep(10)


def main():
    os.makedirs(OUT, exist_ok=True)
    for repo in DATASETS:
        print("dataset:", repo)
        retry(lambda: snapshot_download(repo_id=repo, repo_type="dataset",
                                        local_dir=os.path.join(OUT, repo.split("/")[1])))
    for repo in MODELS:
        print("model:", repo)
        files = retry(lambda: list_repo_files(repo))
        weights = ["model.safetensors"] if "model.safetensors" in files \
            else ["pytorch_model.bin"]
        retry(lambda: snapshot_download(repo_id=repo,
                                        local_dir=os.path.join(OUT, repo.split("/")[1]),
                                        allow_patterns=MODEL_FILES + weights))
    total = 0
    for root, _, fs in os.walk(OUT):
        for f in fs:
            total += os.path.getsize(os.path.join(root, f))
    for repo in MODELS:
        d = os.path.join(OUT, repo.split("/")[1])
        w = [f for f in os.listdir(d) if f.endswith((".safetensors", ".bin"))]
        size = sum(os.path.getsize(os.path.join(d, f)) for f in w) / 1e6
        print(f"  {repo}: weights {w} = {size:.0f} MB")
    print(f"done: {OUT}/ ({total / 1e6:.0f} MB). Copy this folder to the server.")


if __name__ == "__main__":
    main()
