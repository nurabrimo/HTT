"""Frozen BiomedBERT embeddings of whole TCGA reports (512-token chunks, mean of chunk mean-pools).
Row order = rows of tcga_cdr.pkl. Usage: embed.py <tcga_cdr.pkl> <raw|redacted_all> <out.npy>"""
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

M = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


@torch.no_grad()
def main(pkl, view, out):
    df = pd.read_pickle(pkl).reset_index(drop=True)
    col = "text" if view == "raw" else "text_redacted_all"
    tok = AutoTokenizer.from_pretrained(M)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(M).to(dev).eval()
    embs = []
    for i, t in enumerate(df[col].astype(str)):
        ids = tok(t, add_special_tokens=False)["input_ids"] or [tok.unk_token_id]
        chunks = [ids[k:k + 510] for k in range(0, len(ids), 510)][:16]
        batch = [[tok.cls_token_id] + c + [tok.sep_token_id] for c in chunks]
        L = max(len(b) for b in batch)
        x = torch.tensor([b + [0] * (L - len(b)) for b in batch], device=dev)
        m = (x != 0).long()
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            h = model(input_ids=x, attention_mask=m).last_hidden_state.float()
        v = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)
        embs.append(v.mean(0).cpu().numpy())
        if i % 1000 == 0:
            print(i, flush=True)
    np.save(out, np.stack(embs).astype(np.float32))
    print("saved", out, len(embs))


if __name__ == "__main__":
    main(*sys.argv[1:4])
