"""Download the two tool-trained retrievers directly on the server.

Tries, for each model: ModelScope (same repo id), then hf-mirror.com,
then huggingface.co. Files go to hf_local/<name>/ exactly like
download_on_pc.py, so config.py finds them.

    pip install -i https://mirrors.cloud.tencent.com/pypi/simple modelscope
    python download_on_server.py
"""
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_local")
MODELS = ["ToolBench/ToolBench_IR_bert_based_uncased",
          "mangopy/ToolRet-trained-bge-base-en-v1.5"]
PATTERNS = ["*.json", "*.txt", "*.model", "1_Pooling/*",
            "model.safetensors", "pytorch_model.bin"]


def has_weights(d):
    return os.path.isdir(d) and any(f.endswith((".safetensors", ".bin"))
                                    and os.path.getsize(os.path.join(d, f)) > 1e7
                                    for f in os.listdir(d))


def try_modelscope(repo, d):
    from modelscope import snapshot_download
    snapshot_download(repo, local_dir=d, allow_patterns=PATTERNS)


def try_hf(repo, d, endpoint):
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, local_dir=d, allow_patterns=PATTERNS,
                      endpoint=endpoint)


def main():
    os.makedirs(OUT, exist_ok=True)
    ok_all = True
    for repo in MODELS:
        d = os.path.join(OUT, repo.split("/")[1])
        if has_weights(d):
            print(f"{repo}: already there")
            continue
        for name, fn in (("modelscope", lambda: try_modelscope(repo, d)),
                         ("hf-mirror", lambda: try_hf(repo, d, "https://hf-mirror.com")),
                         ("huggingface", lambda: try_hf(repo, d, "https://huggingface.co"))):
            print(f"{repo}: trying {name} ...")
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                print(f"  {name} failed: {type(e).__name__}: {str(e)[:150]}")
                continue
            if has_weights(d):
                print(f"  OK via {name}: {sorted(os.listdir(d))}")
                break
            print(f"  {name}: no weights downloaded")
        else:
            ok_all = False
            print(f"{repo}: NOT DOWNLOADED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
