"""The four retrieval families compared in the paper.

  bm25          lexical baseline (Okapi BM25, k1=1.5, b=0.75)
  dense         bi-encoder (EMBED_MODEL), cosine similarity
  dense_rerank  dense top-RERANK_DEPTH rescored by a cross-encoder
  llm_rewrite   LLM rewrites the query into a tool description, then dense
  dense_toolllm / dense_toolret   published tool-trained bi-encoders
                (config.TOOL_DENSE), same Dense class with another model

Every retriever exposes full_scores(qid, text) -> np.array over the whole
corpus (index aligned with `tool_ids`), or rank(qid, text, pool_idx) for the
reranker. Scores over the full corpus are computed once per query and then
restricted to each pool, so all library sizes share the same computation.
"""
import math
import os
import re
import time
from collections import Counter, defaultdict

import numpy as np

import config as C
from common import JsonCache, key_of, ollama_chat

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    # split camelCase / snake_case so "getWeather" -> get weather
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return TOKEN_RE.findall(text.lower())


def truncate(text):
    return text[:C.MAX_DOC_CHARS]


class BM25:
    name = "bm25"

    def __init__(self, tool_texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.n = len(tool_texts)
        self.doc_len = np.zeros(self.n, dtype=np.float32)
        self.postings = defaultdict(list)  # term -> [(doc, tf)]
        df = Counter()
        for i, t in enumerate(tool_texts):
            tf = Counter(tokenize(truncate(t)))
            self.doc_len[i] = sum(tf.values())
            for term, c in tf.items():
                self.postings[term].append((i, c))
                df[term] += 1
        self.avgdl = float(self.doc_len.mean())
        self.idf = {t: math.log(1 + (self.n - d + 0.5) / (d + 0.5))
                    for t, d in df.items()}

    def full_scores(self, qid, text):
        s = np.zeros(self.n, dtype=np.float32)
        for term in set(tokenize(text)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                s[i] += idf * tf * (self.k1 + 1) / norm
        return s

    def score_new_doc(self, text, doc_text):
        """BM25 score of an unseen document under the corpus statistics."""
        tf = Counter(tokenize(truncate(doc_text)))
        dl = sum(tf.values())
        s = 0.0
        for term in set(tokenize(text)):
            if term in tf and term in self.idf:
                norm = tf[term] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += self.idf[term] * tf[term] * (self.k1 + 1) / norm
        return s


class Dense:
    name = "dense"

    def __init__(self, tool_ids, tool_texts, model=None, prefix=None):
        from sentence_transformers import SentenceTransformer
        model = model or C.EMBED_MODEL
        self.prefix = C.EMBED_QUERY_PREFIX if prefix is None else prefix
        if os.path.isdir(model) and not os.path.exists(os.path.join(model, "modules.json")):
            # plain transformers checkpoint (e.g. trained with FlagEmbedding):
            # sentence-transformers would default to mean pooling, but bge
            # models use the [CLS] vector, so build the model explicitly
            from sentence_transformers import models
            word = models.Transformer(model, max_seq_length=512)
            pool = models.Pooling(word.get_word_embedding_dimension(),
                                  pooling_mode="cls")
            self.model = SentenceTransformer(modules=[word, pool], device=C.DEVICE)
            print(f"{model}: no sentence-transformers config, using CLS pooling")
        else:
            self.model = SentenceTransformer(model, device=C.DEVICE)
        tag = os.path.basename(model.rstrip("/"))
        path = os.path.join(C.CACHE_DIR, f"emb_{tag}_{len(tool_ids)}.npy")
        if os.path.exists(path):
            self.doc_emb = np.load(path)
        else:
            print(f"encoding {len(tool_texts)} tools with {model} ...")
            self.doc_emb = self.encode_docs(tool_texts, progress=True)
            np.save(path, self.doc_emb)

    def encode_docs(self, texts, progress=False):
        return self.model.encode([truncate(t) for t in texts],
                                 batch_size=C.BATCH_SIZE,
                                 normalize_embeddings=True,
                                 show_progress_bar=progress).astype(np.float32)

    def encode_query(self, text):
        return self.model.encode([self.prefix + text],
                                 normalize_embeddings=True)[0].astype(np.float32)

    def full_scores(self, qid, text):
        return self.doc_emb @ self.encode_query(text)


class Reranker:
    name = "dense_rerank"

    def __init__(self, dense, tool_texts):
        from sentence_transformers import CrossEncoder
        self.dense = dense
        self.texts = tool_texts
        self.ce = CrossEncoder(C.RERANK_MODEL, device=C.DEVICE, max_length=512)
        self.cache = {}  # (query text hash, doc index) -> score, per process

    def ce_scores(self, text, doc_idx, doc_texts=None):
        qh = key_of(text)
        todo = [i for i in doc_idx if (qh, i) not in self.cache]
        if todo:
            pairs = [(text, truncate(self.texts[i])) for i in todo]
            for i, s in zip(todo, self.ce.predict(pairs, batch_size=C.BATCH_SIZE)):
                self.cache[(qh, i)] = float(s)
        return np.array([self.cache[(qh, i)] for i in doc_idx], dtype=np.float32)

    def rank(self, dense_scores_in_pool, pool_idx, text):
        """Returns pool positions ordered best-first."""
        order = np.argsort(-dense_scores_in_pool)
        head = order[:C.RERANK_DEPTH]
        ce = self.ce_scores(text, [int(pool_idx[j]) for j in head])
        head = head[np.argsort(-ce)]
        return np.concatenate([head, order[C.RERANK_DEPTH:]])


REWRITE_SYSTEM = (
    "You help a search engine find software tools (APIs/functions). "
    "Given a user request, write a short description of the tool or tools "
    "that would fulfil it: likely function names, what each does, and its "
    "key parameters. Do not answer the request itself. At most 80 words.")


class LLMRewrite:
    name = "llm_rewrite"

    def __init__(self, dense):
        self.dense = dense
        self.cache = JsonCache(f"rewrite_{C.REWRITE_MODEL.replace(':', '_')}.jsonl")

        self.last_llm_sec = 0.0
        self.last_ok = True

    def rewrite(self, text):
        """Returns the rewrite; its original generation time is kept in the
        cache so latency stays correct when a cached rewrite is reused."""
        k = key_of(C.REWRITE_MODEL, text)
        out = self.cache.get(k)
        self.last_ok = True
        if out is None:
            t0 = time.perf_counter()
            try:
                txt, _, _ = ollama_chat(C.REWRITE_MODEL, text, system=REWRITE_SYSTEM,
                                        max_tokens=200)
            except Exception as e:  # noqa: BLE001
                # fall back to the plain query; not cached, so a rerun retries it
                print(f"  query rewrite failed ({type(e).__name__}); using the query alone")
                self.last_ok = False
                self.last_llm_sec = 0.0
                return ""
            out = {"text": txt, "sec": time.perf_counter() - t0}
            self.cache.put(k, out)
        self.last_llm_sec = out["sec"]
        return out["text"]

    def full_scores(self, qid, text):
        # original query + generated description, as in query-expansion work
        return self.dense.full_scores(qid, text + "\n" + self.rewrite(text))
