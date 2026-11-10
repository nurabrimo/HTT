"""Data, training and evaluation shared by the synthetic and TCGA runners.
Training semantics follow OncoLocus HTTTrainer: AdamW (lr 2e-4, wd 0.01, trainable params only),
LinearLR warm-up from 0.1x over the first 10% of optimizer steps, batch 4 x 4 accumulation,
grad-norm clip 1.0, AMP, early stopping on validation AUROC (patience 5), best checkpoint restored.
"""
import copy
import math
import random
import re
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

SIZE_CUR = re.compile(r"(\d+(?:\.\d+)?) cm")
SIZE_PREV = re.compile(r"previously (\d+(?:\.\d+)?) cm")


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def parse_sizes(text):
    """Oracle numeracy: current and previous size read with a regex from the report text."""
    c = SIZE_CUR.search(text)
    p = SIZE_PREV.search(text)
    cur = float(c.group(1)) if c else 1.0
    if p:
        prev = float(p.group(1))
        return [math.log(cur), math.log(prev), math.log(cur / prev), 0.0]
    return [math.log(cur), 0.0, 0.0, 1.0]


class Patients:
    """Pre-tokenised patient sequences. df columns: patient_id, cancer_type, visit_index,
    days_from_start, report_text, label (patient label = max visit label, as in OncoLocus)."""

    def __init__(self, df, tok, type_ids, max_len=128, max_visits=10, with_feats=False, unseen_id=None):
        self.items = []
        for pid, g in df.groupby("patient_id", sort=False):
            g = g.sort_values("visit_index").iloc[:max_visits]
            texts = g.report_text.astype(str).tolist()
            enc = tok(texts, max_length=max_len, truncation=True, padding="max_length", return_tensors="pt") \
                if tok is not None else None
            ct = g.cancer_type.iloc[0]
            self.items.append(dict(
                pid=pid,
                ids=enc["input_ids"] if enc is not None else None,
                mask=enc["attention_mask"] if enc is not None else None,
                days=torch.tensor(g.days_from_start.values, dtype=torch.float),
                vlab=torch.tensor(g.label.values, dtype=torch.long),
                plab=int(g.label.max()),
                type_id=type_ids.get(ct, unseen_id if unseen_id is not None else len(type_ids)),
                ctype=ct,
                feats=torch.tensor([parse_sizes(t) for t in texts], dtype=torch.float) if with_feats else None,
            ))
        self.max_len, self.max_visits = max_len, max_visits

    def __len__(self):
        return len(self.items)


def collate(items, max_visits, max_len, days_override=None):
    B = len(items)
    S = max(len(it["days"]) for it in items)
    S = min(S, max_visits)
    has_text = items[0]["ids"] is not None
    has_f = items[0]["feats"] is not None
    ids = torch.zeros(B, S, max_len, dtype=torch.long) if has_text else None
    am = torch.zeros(B, S, max_len, dtype=torch.long) if has_text else None
    days = torch.zeros(B, S)
    sm = torch.zeros(B, S, dtype=torch.bool)
    vl = torch.full((B, S), -1, dtype=torch.long)
    feats = torch.zeros(B, S, items[0]["feats"].shape[1]) if has_f else None
    for i, it in enumerate(items):
        n = min(len(it["days"]), S)
        if has_text:
            ids[i, :n] = it["ids"][:n]; am[i, :n] = it["mask"][:n]
        d = it["days"][:n] if days_override is None else days_override(n)
        days[i, :n] = d
        sm[i, :n] = True
        vl[i, :n] = it["vlab"][:n]
        if has_f:
            feats[i, :n] = it["feats"][:n]
    if has_text:                                          # dynamic padding: identical under the attention mask
        L = int(am.sum(-1).max().clamp(min=1))
        ids, am = ids[:, :, :L].contiguous(), am[:, :, :L].contiguous()
    yrs = days / 365.0                                    # OncoLocus normalises days to years
    elapsed = yrs
    interval = torch.zeros_like(yrs)
    interval[:, 1:] = yrs[:, 1:] - yrs[:, :-1]
    interval = interval * sm
    order = torch.arange(S).float().unsqueeze(0).expand(B, S) * 66.0 / 365.0 * sm
    return dict(input_ids=ids, attention_mask=am, seq_mask=sm, elapsed=elapsed, interval=interval,
                order=order, visit_labels=vl,
                patient_label=torch.tensor([it["plab"] for it in items]),
                type_id=torch.tensor([it["type_id"] for it in items]), feats=feats)


def batches(ds, bs, shuffle, rng=None, days_override=None):
    idx = list(range(len(ds)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield collate([ds.items[j] for j in idx[i:i + bs]], ds.max_visits, ds.max_len, days_override)


def to_dev(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan")


@torch.no_grad()
def predict(model, ds, dev, bs=32, days_override=None, want_summary=False):
    model.eval()
    probs, summ = [], []
    for b in batches(ds, bs, False, days_override=days_override):
        b = to_dev(b, dev)
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            out = model(b, return_summary=want_summary)
        probs.append(torch.softmax(out[0].float(), -1)[:, 1].cpu())
        if want_summary:
            summ.append(out[2].float().cpu())
    p = torch.cat(probs).numpy()
    if want_summary:
        return p, torch.cat(summ).numpy()
    return p


def train(model, tr, va, dev, seed, epochs=10, patience=5, lr=2e-4, bs=4, accum=4, visit_w=0.3,
          log=print):
    rng = random.Random(seed)
    model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(tr) / bs)
    total = steps_per_epoch * epochs // accum
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=max(1, total // 10))
    crit = nn.CrossEntropyLoss(ignore_index=-1)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    yv = [it["plab"] for it in va.items]
    best, best_ep, best_state, wait = -1.0, 0, None, 0
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        tl, nb = 0.0, 0
        for step, b in enumerate(batches(tr, bs, True, rng)):
            b = to_dev(b, dev)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                pl, vl = model(b)
                loss = crit(pl.float(), b["patient_label"])
                if visit_w > 0:
                    loss = loss + visit_w * crit(vl.float().reshape(-1, 2), b["visit_labels"].reshape(-1))
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
            tl += loss.item(); nb += 1
        va_auc = auc(yv, predict(model, va, dev))
        hist.append(dict(epoch=ep, train_loss=tl / max(nb, 1), val_auroc=va_auc))
        log(f"epoch {ep} loss {tl / max(nb, 1):.4f} val_auroc {va_auc:.4f}")
        if va_auc > best:
            best, best_ep, wait = va_auc, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if patience and wait >= patience:
                log(f"early stop at {ep}")
                break
    model.load_state_dict(best_state)
    return dict(history=hist, best_val=best, best_epoch=best_ep)
