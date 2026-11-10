"""HTT model family, following OncoLocus@47719d1 (src/models/*) with these switches:

time_band : 'orig'     base frequencies exp(-i ln20 / d) rad/yr, t in years (periods 6.3-126 y)
            'clinical' periods log-spaced from 7 d to 5 y (HTT-IA)
time_input: 'elapsed' | 'interval' | 'both' | 'none' | 'order'
numbers   : add a linear projection of sizes parsed from the text (oracle numeracy)
reader    : 'text'  BiomedBERT+LoRA report encoder (Level 1)
            'struct' no text; per-visit input is a projection of the parsed sizes only
encoder   : 'biomedbert' (pre-trained, LoRA), 'bert' (general-domain, LoRA),
            'random' (BiomedBERT architecture, random init, all weights trained)
"""
import math
import torch
import torch.nn as nn

BIOMEDBERT = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
GENERAL_BERT = "google-bert/bert-base-uncased"


class TimeCode(nn.Module):
    def __init__(self, d_model, band="orig"):
        super().__init__()
        d = d_model // 2
        if band == "orig":
            freqs = torch.exp(torch.arange(d).float() * (-math.log(20.0) / d))  # rad / year
        elif band == "clinical":
            periods = torch.logspace(math.log10(7 / 365.25), math.log10(5.0), d)  # years
            freqs = 2 * math.pi / periods
        else:
            raise ValueError(band)
        self.register_buffer("freqs", freqs)
        self.freq_scale = nn.Parameter(torch.ones(d))

    def forward(self, t_years):
        ang = t_years.unsqueeze(-1) * (self.freqs * self.freq_scale.abs())
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class ReportEncoder(nn.Module):
    def __init__(self, encoder="biomedbert", hidden=512, lora_r=16, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        from transformers import AutoModel, AutoConfig
        name = GENERAL_BERT if encoder == "bert" else BIOMEDBERT
        if encoder == "random":
            cfg = AutoConfig.from_pretrained(name)
            base = AutoModel.from_config(cfg)
            self.encoder = base                       # all weights trainable
        else:
            from peft import get_peft_model, LoraConfig, TaskType
            base = AutoModel.from_pretrained(name, dtype=torch.float32)
            lcfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=lora_r, lora_alpha=lora_alpha,
                              lora_dropout=lora_dropout, target_modules=["query", "value"], bias="none")
            self.encoder = get_peft_model(base, lcfg)
        self.projection = nn.Sequential(nn.Linear(base.config.hidden_size, hidden), nn.LayerNorm(hidden), nn.GELU())

    def forward(self, ids, mask):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        return self.projection((h * m).sum(1) / m.sum(1).clamp(min=1))


class StructEncoder(nn.Module):
    def __init__(self, n_feat=4, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, hidden), nn.LayerNorm(hidden), nn.GELU())

    def forward(self, feats):
        return self.net(feats)


def _encode_visits(enc, ids, mask, seq_mask, hidden):
    """Encode only real visits, scatter back to (B, S, H)."""
    B, S, L = ids.shape
    flat = seq_mask.reshape(-1)
    out = torch.zeros(B * S, hidden, device=ids.device, dtype=torch.float32)
    e = enc(ids.reshape(B * S, L)[flat], mask.reshape(B * S, L)[flat])
    out[flat] = e.float()
    return out.view(B, S, hidden)


class HTT(nn.Module):
    def __init__(self, reader="text", encoder="biomedbert", time_band="orig", time_input="elapsed",
                 numbers=False, n_types=5, hidden=512, n_heads=8, n_layers=4, ff=2048,
                 temporal_dropout=0.1, cls_dropout=0.2, n_feat=4):
        super().__init__()
        self.reader, self.hidden, self.time_input, self.numbers = reader, hidden, time_input, numbers
        if reader == "text":
            self.enc = ReportEncoder(encoder, hidden)
        else:
            self.enc = StructEncoder(n_feat, hidden)
        if numbers:
            self.num_proj = nn.Linear(n_feat, hidden)
        self.time = TimeCode(hidden, time_band)
        if time_input == "both":
            self.time2 = TimeCode(hidden, time_band)
        self.type_emb = nn.Embedding(n_types + 1, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        layer = nn.TransformerEncoderLayer(hidden, n_heads, ff, temporal_dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(hidden),
                                                 enable_nested_tensor=False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.patient_head = nn.Sequential(nn.Dropout(cls_dropout), nn.Linear(hidden, hidden // 2), nn.GELU(),
                                          nn.Dropout(cls_dropout), nn.Linear(hidden // 2, 2))
        self.visit_head = nn.Sequential(nn.Dropout(cls_dropout), nn.Linear(hidden, 2))

    def time_code(self, b):
        ti = self.time_input
        if ti == "none":
            return 0.0
        if ti == "elapsed":
            return self.time(b["elapsed"])
        if ti == "interval":
            return self.time(b["interval"])
        if ti == "order":
            return self.time(b["order"])
        if ti == "both":
            return self.time(b["elapsed"]) + self.time2(b["interval"])
        raise ValueError(ti)

    def embed(self, b):
        sm = b["seq_mask"]
        if self.reader == "text":
            x = _encode_visits(self.enc, b["input_ids"], b["attention_mask"], sm, self.hidden)
        else:
            x = self.enc(b["feats"])
        if self.numbers:
            x = x + self.num_proj(b["feats"])
        return x

    def forward(self, b, return_summary=False):
        sm = b["seq_mask"]
        x = self.embed(b)
        x = x + self.time_code(b)
        x = x + self.type_emb(b["type_id"]).unsqueeze(1)
        B = x.shape[0]
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], 1)
        full = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), sm], 1)
        out = self.transformer(self.input_norm(x), src_key_padding_mask=~full)
        summ, visits = out[:, 0], out[:, 1:]
        pl, vl = self.patient_head(summ), self.visit_head(visits)
        return (pl, vl, summ) if return_summary else (pl, vl)


class Baseline(nn.Module):
    """Most recent report only; same Level 1; head also applied to every report (visit loss)."""

    def __init__(self, encoder="biomedbert", hidden=512, cls_dropout=0.2, **_):
        super().__init__()
        self.hidden = hidden
        self.enc = ReportEncoder(encoder, hidden)
        self.head = nn.Sequential(nn.Dropout(cls_dropout), nn.Linear(hidden, hidden // 2), nn.GELU(),
                                  nn.Dropout(cls_dropout), nn.Linear(hidden // 2, 2))

    def forward(self, b, return_summary=False):
        sm = b["seq_mask"]
        x = _encode_visits(self.enc, b["input_ids"], b["attention_mask"], sm, self.hidden)
        B = x.shape[0]
        last = sm.long().sum(1).clamp(min=1) - 1
        e = x[torch.arange(B, device=x.device), last]
        pl, vl = self.head(e), self.head(x)
        return (pl, vl, e) if return_summary else (pl, vl)
