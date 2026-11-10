"""Download GENIE BPC public studies (patient clinical attributes + timeline clinical events) from
genie.cbioportal.org with the user's data-access token. Data stay on DCC.
Usage: genie_fetch.py <token_file> <out_dir> [study ...]"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

API = "https://genie.cbioportal.org/api"
STUDIES = ["nsclc_public_genie_bpc", "crc_public_genie_bpc", "brca_public_genie_bpc", "panc_public_genie_bpc",
           "prostate_public_genie_bpc", "bladder_public_genie_bpc"]


def main(token_file, out, studies):
    tok = [l.split(":", 1)[1].strip() for l in open(token_file) if l.startswith("token")][0]
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {tok}", "Accept": "application/json"})

    def get(path, **params):
        for k in range(5):
            r = s.get(API + path, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            time.sleep(2 * (k + 1))
        r.raise_for_status()

    os.makedirs(out, exist_ok=True)
    for st in studies:
        path = os.path.join(out, f"{st}.json")
        if os.path.exists(path):
            print("exists", st); continue
        pts = get(f"/studies/{st}/patients", pageSize=100000)
        pclin = get(f"/studies/{st}/clinical-data", clinicalDataType="PATIENT", pageSize=10000000)
        sclin = get(f"/studies/{st}/clinical-data", clinicalDataType="SAMPLE", pageSize=10000000)
        ids = [p["patientId"] for p in pts]

        def ev(pid):
            return pid, get(f"/studies/{st}/patients/{pid}/clinical-events", pageSize=100000)
        with ThreadPoolExecutor(8) as ex:
            events = dict(ex.map(ev, ids))
        json.dump(dict(study=st, patients=ids, patient_clinical=pclin, sample_clinical=sclin, events=events),
                  open(path + ".tmp", "w"))
        os.replace(path + ".tmp", path)
        n_ev = sum(len(v) for v in events.values())
        types = {}
        for v in events.values():
            for e in v:
                types[e.get("eventType")] = types.get(e.get("eventType"), 0) + 1
        print(st, "patients", len(ids), "events", n_ev, types, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or STUDIES)
