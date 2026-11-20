"""Step 4: embed each company's business description with approved pretrained models, run locally.

Models are loaded from the local Hugging Face cache at the exact revisions in the handout.
A document is split into ~300-word chunks (first 8); chunk embeddings are L2-normalised and averaged.
Only the text of the company's own dated filing is embedded; no model is asked about a company.
Writes cache/emb_<name>.npy (rows in companies.csv order).
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = os.path.join(HERE, "cache", "hf")
MODELS = {
    "bge": ("BAAI/bge-base-en-v1.5", "70e8c52057794ed37381f7da00a03097927500a7", ""),
    "gte": ("thenlper/gte-base", "7cb59c7765dcf58720e0e63bdbbf8c10cbba8668", ""),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", "5681fe04da6e48e851d5dd1af673670cdb299753", ""),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "7dbbc90392e2f80f3d3c277d6e90027e55de9125", ""),
    "e5": ("intfloat/e5-large-v2", "b322e09026e4ea05f42beadf4d661fb4e101d311", "query: "),
}
CHUNK, MAX_CHUNKS = 300, 8


def snapshot(repo, rev):
    return os.path.join(HF, "models--" + repo.replace("/", "--"), "snapshots", rev)


def main(names):
    torch.manual_seed(0)
    comp = pd.read_csv(os.path.join(HERE, "data", "companies.csv"), dtype=str)
    tx = pd.read_parquet(os.path.join(HERE, "cache", "company_texts.parquet")).set_index("stock_id").loc[comp.stock_id]
    for name in names:
        repo, rev, prefix = MODELS[name]
        model = SentenceTransformer(snapshot(repo, rev), device="cpu").float()  # gte ships float16 weights
        model.max_seq_length = 512
        chunks, owner = [], []
        for i, t in enumerate(tx.text):
            w = t.split()
            for j in range(0, min(len(w), CHUNK * MAX_CHUNKS), CHUNK):
                chunks.append(prefix + " ".join(w[j:j + CHUNK]))
                owner.append(i)
        E = model.encode(chunks, batch_size=16, normalize_embeddings=True, show_progress_bar=False,
                         convert_to_numpy=True)
        owner = np.array(owner)
        D = np.stack([E[owner == i].mean(0) for i in range(len(tx))])
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        np.save(os.path.join(HERE, "cache", f"emb_{name}.npy"), D.astype(np.float64))
        print(name, D.shape, len(chunks), "chunks", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
