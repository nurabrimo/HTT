"""Growth benchmark: generator and Bayes-optimal (ceiling) readers.

Re-implementation from the manuscript's specification (Section 2.4, Fig. 4):
  Y ~ Bernoulli(0.40)                      rapid progression
  DT | Y ~ LogNormal(log(120 d if Y else 600 d), 0.35)
  s0 ~ LogNormal(log 2.0 cm, 0.60)         baseline size
  c ~ U(14, 180) d                         patient-level visit cadence (independent of Y)
  n_target ~ U{2..8}; gaps J_k = c * U(0.5, 1.5); visits stop at n_target or when t_k > 365 d
  s_k = s0 * 2^(t_k / DT)
  m_k = round(s_k * exp(N(0, 0.08)), 0.1 cm)   reported size (8% multiplicative noise)
Report k states m_k, the previous reported size m_{k-1} (same variable) and a verdict word
derived from their ratio. No date appears in the text.
"""
import math
import random
import numpy as np
import pandas as pd

PREV = 0.40
DT_MED = {1: 120.0, 0: 600.0}
DT_SD = 0.35
S0_MED, S0_SD = 2.0, 0.60
NOISE = 0.08
CAD_LO, CAD_HI = 14.0, 180.0
JIT_LO, JIT_HI = 0.5, 1.5
FOLLOW = 365.0
MAX_VISITS = 10
ROUND = 0.1
PREV_INDEP = False

TYPES = ["NSCLC", "BrCa", "CRC", "Prostate", "PANC"]
TOKENS = {"NSCLC": "[CANCER:LUNG]", "BrCa": "[CANCER:BREAST]", "CRC": "[CANCER:COLORECTAL]",
          "PANC": "[CANCER:PANCREATIC]", "Prostate": "[CANCER:PROSTATE]"}

TEMPLATES = {
    "NSCLC": [
        "CT chest. The {lobe} mass measures {cur} cm, {prev}. Mediastinal nodes {nodes}. Impression: {verdict}.",
        "Interval CT chest. Target lesion in the {lobe} measures {cur} cm ({prev}). No pleural effusion. Impression: {verdict}.",
        "CT chest with contrast. Primary lung lesion {cur} cm, {prev}. Hilar nodes {nodes}. Findings {verdict}.",
        "Restaging CT. Known NSCLC, dominant lesion now {cur} cm, {prev}. Adrenal glands unremarkable. Impression: {verdict}.",
    ],
    "BrCa": [
        "MRI breast with contrast. Index lesion in the {side} breast measures {cur} cm, {prev}. Axillary nodes {nodes}. Impression: {verdict}.",
        "Breast MRI. Enhancing mass {side} breast {cur} cm ({prev}). No new satellite lesions. Findings {verdict}.",
        "Contrast-enhanced breast MRI. Known carcinoma, {side} breast, now {cur} cm, {prev}. Impression: {verdict}.",
        "MRI breast. Target lesion {cur} cm in the {side} breast, {prev}. Skin and chest wall intact. Impression: {verdict}.",
    ],
    "CRC": [
        "CT abdomen and pelvis. Dominant hepatic metastasis segment {seg} measures {cur} cm, {prev}. Impression: {verdict}.",
        "CT abdomen/pelvis with contrast. Liver lesion segment {seg} {cur} cm ({prev}). No ascites. Findings {verdict}.",
        "Restaging CT. Colorectal primary resected. Target hepatic lesion {cur} cm, {prev}. Portal nodes {nodes}. Impression: {verdict}.",
        "CT abdomen and pelvis. Peritoneal nodule {cur} cm, {prev}. Anastomosis unremarkable. Impression: {verdict}.",
    ],
    "Prostate": [
        "Multiparametric MRI prostate. Index lesion in the {zone} zone measures {cur} cm, {prev}. Impression: {verdict}.",
        "MRI prostate. {zone} zone lesion {cur} cm ({prev}). No extracapsular extension. Findings {verdict}.",
        "Prostate mpMRI. Target lesion now {cur} cm, {prev}. Seminal vesicles clear. Impression: {verdict}.",
        "MRI pelvis, prostate protocol. Dominant {zone} zone lesion {cur} cm, {prev}. Pelvic nodes {nodes}. Impression: {verdict}.",
    ],
    "PANC": [
        "CT abdomen and pelvis with contrast. Pancreatic head mass {cur} cm, {prev}. SMV {vessel}. Impression: {verdict}.",
        "MRI abdomen. Known pancreatic adenocarcinoma, primary tumour {cur} cm ({prev}). Findings {verdict}.",
        "CT for restaging. Pancreatic mass now {cur} cm, {prev}. Peripancreatic nodes {nodes}. Impression: {verdict}.",
        "CT abdomen and pelvis. Pancreatic body lesion {cur} cm, {prev}. Celiac axis {vessel}. Impression: {verdict}.",
    ],
}
FILL = {
    "lobe": ["right upper lobe", "right lower lobe", "left upper lobe", "left lower lobe"],
    "side": ["left", "right"], "seg": [str(i) for i in range(2, 9)],
    "zone": ["peripheral", "transition", "central"],
    "nodes": ["not enlarged", "stable", "mildly prominent"],
    "vessel": ["patent", "abutted", "encased without occlusion"],
}
VERDICT = {
    "up": ["increased in size", "interval enlargement", "enlarged compared with prior"],
    "same": ["stable in size", "no significant interval change", "similar to prior"],
    "down": ["decreased in size", "interval decrease"],
}


def _fmt(x):
    return f"{x:.1f}"


def _round(x):
    return max(ROUND, round(x / ROUND) * ROUND)


N_LO, N_HI = 2, 8


def sample_times(rng):
    c = rng.uniform(CAD_LO, CAD_HI)
    n = int(rng.integers(N_LO, N_HI + 1))
    t = [0.0]
    while len(t) < min(n, MAX_VISITS):
        nxt = t[-1] + c * rng.uniform(JIT_LO, JIT_HI)
        if nxt > FOLLOW:
            break
        t.append(nxt)
    return np.array(t)


def make_patient(pid, ctype, rng, prng):
    y = int(rng.random() < PREV)
    dt = math.exp(math.log(DT_MED[y]) + DT_SD * rng.normal())
    s0 = math.exp(math.log(S0_MED) + S0_SD * rng.normal())
    t = sample_times(rng)
    s = s0 * 2.0 ** (t / dt)
    m = [_round(v * math.exp(NOISE * rng.normal())) for v in s]
    if PREV_INDEP:
        mp = [np.nan] + [_round(v * math.exp(NOISE * rng.normal())) for v in s[:-1]]
    else:
        mp = [np.nan] + m[:-1]
    rows = []
    for k in range(len(t)):
        if k == 0:
            prev = "baseline examination, no prior for comparison"
            verdict = prng.choice(["baseline measurement", "baseline study"])
        else:
            prev = f"previously {_fmt(mp[k])} cm"
            r = m[k] / mp[k]
            verdict = prng.choice(VERDICT["up" if r >= 1.2 else ("down" if r <= 0.8 else "same")])
        tpl = prng.choice(TEMPLATES[ctype])
        txt = tpl.format(cur=_fmt(m[k]), prev=prev, verdict=verdict,
                         **{k2: prng.choice(v) for k2, v in FILL.items()})
        rows.append(dict(patient_id=pid, cancer_type=ctype, visit_index=k,
                         days_from_start=float(t[k]), report_text=f"{TOKENS[ctype]} {txt}",
                         label=y, size=m[k], size_prev=mp[k], dt_true=dt))
    return rows


def generate(n_per_type, types, seed, prefix):
    rng = np.random.default_rng(seed)
    prng = random.Random(seed + 7)
    rows = []
    for ct in types:
        for i in range(n_per_type):
            rows += make_patient(f"{prefix}-{ct}-{i:05d}", ct, rng, prng)
    return pd.DataFrame(rows)


def build_benchmark(n_train_per_type=70, seed=42):
    """Splits mirroring the phrase benchmark: 4 training types x (70 train, 15 val,
    15+15+15 test) and the held-out PANC type (100 + 200)."""
    train_types = ["NSCLC", "BrCa", "CRC", "Prostate"]
    sets = {}
    # validation and test pools never change with n_train
    ev = generate(75, train_types, seed + 1000, "G")
    pid = ev.patient_id.str.split("-").str[-1].astype(int)
    sets["val"] = ev[pid < 15]
    sets["test60"] = ev[(pid >= 15) & (pid < 30)]
    sets["test120"] = ev[(pid >= 15) & (pid < 45)]
    sets["test180"] = ev[(pid >= 15) & (pid < 60)]
    pa = generate(300, ["PANC"], seed + 2000, "G")
    ppid = pa.patient_id.str.split("-").str[-1].astype(int)
    sets["PANC100"] = pa[ppid < 100]
    sets["PANC300"] = pa
    # nested training pool: prefixes of one large pool, so larger N contains smaller N
    tr = generate(max(n_train_per_type, 70), train_types, seed, "T")
    tpid = tr.patient_id.str.split("-").str[-1].astype(int)
    sets["train"] = tr[tpid < n_train_per_type]
    return {k: v.reset_index(drop=True) for k, v in sets.items()}


# ----------------------------------------------------------------------------------------
# Bayes readers. In log space y_k = a + b t_k + e_k with a = log s0 ~ N(mu_a, tau^2),
# b = ln2 / DT and e_k ~ N(0, sigma_k^2), sigma_k^2 = NOISE^2 + (ROUND^2/12)/m_k^2
# (Sheppard correction for rounding).  a is integrated out analytically; DT on a grid.
# ----------------------------------------------------------------------------------------
LOGDT = np.linspace(math.log(5.0), math.log(50000.0), 1500)
B = math.log(2.0) / np.exp(LOGDT)
_d = LOGDT[1] - LOGDT[0]


def _prior_logdt(y):
    z = (LOGDT - math.log(DT_MED[y])) / DT_SD
    return np.exp(-0.5 * z * z) / (DT_SD * math.sqrt(2 * math.pi)) * _d


PRI = {1: _prior_logdt(1), 0: _prior_logdt(0)}


def _loglik_given_times(y, sig2, T):
    """y,sig2: (n,), T: (M, n) candidate time vectors in days -> (M, G) log-lik over grid of b."""
    mu_a, tau2 = math.log(S0_MED), S0_SD ** 2
    w = 1.0 / sig2                                 # (n,)
    # residual r = y - mu_a - b t ; marginal over a: N(r | 0, D + tau2 11')
    # log N = -0.5[ r'Wr - tau2 (w'r)^2/(1+tau2 sum w) ] - 0.5 log det - n/2 log 2pi
    sw = w.sum()
    c = tau2 / (1 + tau2 * sw)
    logdet = np.log(sig2).sum() + math.log(1 + tau2 * sw)
    r0 = (y - mu_a)[None, None, :]                 # (1,1,n)
    bt = B[None, :, None] * T[:, None, :]          # (M,G,n)
    r = r0 - bt
    q = (r * r * w).sum(-1) - c * ((r * w).sum(-1)) ** 2
    return -0.5 * q - 0.5 * logdet - 0.5 * len(y) * math.log(2 * math.pi)


def _post(ll):
    """ll: (M,G) -> posterior P(Y=1) averaging likelihood over M time samples."""
    mx = ll.max()
    L = np.exp(ll - mx).mean(0)                    # (G,)
    l1 = (L * PRI[1]).sum() * PREV
    l0 = (L * PRI[0]).sum() * (1 - PREV)
    return l1 / (l1 + l0)


def _obs(sizes):
    m = np.asarray(sizes, float)
    return np.log(m), NOISE ** 2 + (ROUND ** 2 / 12.0) / m ** 2


def _obs_patient(g):
    """All size observations of a patient with the visit index each refers to."""
    cur = g["size"].values
    vals, idx = list(cur), list(range(len(cur)))
    if PREV_INDEP:
        pv = g["size_prev"].values
        vals += list(pv[1:]); idx += list(range(len(cur) - 1))
    y, s2 = _obs(vals)
    return y, s2, np.array(idx)


def _obs_last(g):
    """Observations in the last report only: current size (time index 1 of last2) and
    the stated previous size (time index 0)."""
    vals = [g["size_prev"].values[-1], g["size"].values[-1]]
    y, s2 = _obs(vals)
    return y, s2, np.array([0, 1])


def time_bank(n_samples, seed):
    rng = np.random.default_rng(seed)
    return [sample_times(rng) for _ in range(n_samples)]


def ceilings(df, bank_size=1500, seed=123):
    """Posterior P(Y=1) for the three readers, per patient."""
    bank = time_bank(bank_size * 4, seed)
    by_n = {}
    for t in bank:
        by_n.setdefault(len(t), []).append(t)
    by_n = {n: np.array(v[:bank_size]) for n, v in by_n.items()}
    last2 = np.array([t[-2:] for t in bank[:bank_size]])
    out = []
    for pid, g in df.groupby("patient_id", sort=False):
        g = g.sort_values("visit_index")
        y, s2, idx = _obs_patient(g)
        t = g["days_from_start"].values
        p_dates = _post(_loglik_given_times(y, s2, t[None, idx]))
        p_nodates = _post(_loglik_given_times(y, s2, by_n[len(t)][:, idx]))
        yl, sl, il = _obs_last(g)
        p_last = _post(_loglik_given_times(yl, sl, last2[:, il]))
        out.append(dict(patient_id=pid, label=int(g.label.iloc[0]), p_last=p_last,
                        p_nodates=p_nodates, p_dates=p_dates))
    return pd.DataFrame(out)


def redate_ceiling_shift(df, g1=21, g2=90, bank_size=1500):
    """Mean |P(g1 spacing) - P(g2 spacing)| for the with-dates Bayes reader."""
    shifts = []
    for pid, g in df.groupby("patient_id", sort=False):
        g = g.sort_values("visit_index")
        y, s2, idx = _obs_patient(g)
        k = np.arange(len(g))[idx]
        p1 = _post(_loglik_given_times(y, s2, (k * g1)[None, :].astype(float)))
        p2 = _post(_loglik_given_times(y, s2, (k * g2)[None, :].astype(float)))
        shifts.append(abs(p1 - p2))
    return float(np.mean(shifts))


def ols_rate_reader(df):
    """Training-free reader: OLS slope of log(size) on days (dates required)."""
    out = []
    for pid, g in df.groupby("patient_id", sort=False):
        g = g.sort_values("visit_index")
        t = g.days_from_start.values; y = np.log(g["size"].values)
        b = np.polyfit(t, y, 1)[0] if len(t) > 1 else 0.0
        out.append(dict(patient_id=pid, label=int(g.label.iloc[0]), score=b))
    return pd.DataFrame(out)
