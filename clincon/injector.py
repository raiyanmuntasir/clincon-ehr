"""Controlled injection of patient-level clinical inconsistencies.

Each injected error is logged in a ground-truth table so that detectors can be
scored with precision / recall. At most one error is injected per patient, which
keeps per-type evaluation unambiguous.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .generator import EHR
from .knowledge import ALL_TREATMENT_DRUGS, CONDITIONS, MEASUREMENTS, indications_for

# type id -> (category, human description)
INCONSISTENCY_TYPES: dict[str, tuple[str, str]] = {
    "T1": ("temporal", "Discharge time precedes admission time"),
    "T2": ("temporal", "Clinical event recorded after date of death"),
    "D1": ("demographic", "Sex-specific diagnosis recorded for the other sex"),
    "D2": ("demographic", "Diagnosis outside its plausible age range"),
    "P1": ("plausibility", "Physiologically implausible measurement value"),
    "U1": ("plausibility", "Unit mismatch (value recorded in the wrong unit)"),
    "C1": ("integrity", "Conflicting duplicate measurements at the same timestamp"),
    "M1": ("pharmacological", "Disease-specific drug without a supporting diagnosis"),
    "K1": ("causal", "Diagnosis contradicted by its expected physiological evidence"),
}


def inject(ehr: EHR, rate: float = 0.2, seed: int = 7) -> tuple[EHR, pd.DataFrame]:
    """Return a corrupted copy of `ehr` and a ground-truth log."""
    rng = np.random.default_rng(seed)
    e = ehr.copy()
    pts = e.patients.set_index("subject_id")
    ids = list(pts.index)
    rng.shuffle(ids)
    n_target = int(rate * len(ids))
    types = list(INCONSISTENCY_TYPES)
    per_type = {t: n_target // len(types) for t in types}

    dx_by_sid = e.diagnoses_icd.groupby("subject_id")["icd_code"].apply(set).to_dict()
    truth = []
    used: set[int] = set()
    new_meas, new_dx, new_rx = [], [], []

    def admissions_of(sid):
        return e.admissions[e.admissions.subject_id == sid]

    for t in types:
        for sid in ids:
            if per_type[t] == 0:
                break
            if sid in used:
                continue
            sex, age = pts.at[sid, "gender"], int(pts.at[sid, "anchor_age"])
            adm = admissions_of(sid)
            detail = None

            if t == "T1":
                idx = adm.index[int(rng.integers(0, len(adm)))]
                admit = e.admissions.at[idx, "admittime"]
                e.admissions.at[idx, "dischtime"] = admit - pd.Timedelta(hours=int(rng.integers(1, 72)))
                detail = f"hadm {e.admissions.at[idx, 'hadm_id']}"

            elif t == "T2":
                last_admit = adm.admittime.max()
                e.patients.loc[e.patients.subject_id == sid, "dod"] = (
                    last_admit.normalize() - pd.Timedelta(days=int(rng.integers(1, 30)))
                )
                detail = "dod moved before last admission"

            elif t == "D1":
                code = "O80" if sex == "M" else "C61"
                lo, hi = CONDITIONS[code].age_range
                if not (lo <= age <= hi):
                    continue
                new_dx.append((sid, int(adm.hadm_id.iloc[0]), code, 10))
                detail = code

            elif t == "D2":
                if sex == "F" and age > 55:
                    code = "O80"
                elif sex == "M" and age < 35:
                    code = "C61"
                else:
                    continue
                new_dx.append((sid, int(adm.hadm_id.iloc[0]), code, 10))
                detail = f"{code} at age {age}"

            elif t in {"P1", "U1", "C1"}:
                m = e.measurements
                rows = m[m.subject_id == sid]
                if t == "P1":
                    rows = rows[rows.label != "glucose"]
                    idx = rows.index[int(rng.integers(0, len(rows)))]
                    name = m.at[idx, "label"]
                    hi = MEASUREMENTS[name].plausible[1]
                    m.at[idx, "valuenum"] = round(hi * float(rng.uniform(1.3, 2.5)), 2)
                    detail = name
                elif t == "U1":
                    rows = rows[(rows.label == "glucose") & (rows.valuenum < 350)]
                    if rows.empty:
                        continue
                    idx = rows.index[int(rng.integers(0, len(rows)))]
                    m.at[idx, "valuenum"] = round(m.at[idx, "valuenum"] / 18.0, 1)
                    detail = "glucose mmol/L stored as mg/dL"
                else:  # C1
                    rows = rows[rows.kind == "lab"]
                    idx = rows.index[int(rng.integers(0, len(rows)))]
                    r = m.loc[idx].copy()
                    name = r.label
                    # a conflicting value that is still individually plausible, so the
                    # error is only detectable by comparing the two records
                    lo, hi = MEASUREMENTS[name].plausible
                    factor = float(rng.uniform(1.2, 1.5))
                    v = r.valuenum * factor
                    if not lo <= v <= hi:
                        v = r.valuenum / factor
                    if not lo <= v <= hi:
                        v = r.valuenum * 1.15
                    r.valuenum = round(v, 2)
                    new_meas.append(r)
                    detail = name

            elif t == "M1":
                have = dx_by_sid.get(sid, set())
                options = [d for d in ALL_TREATMENT_DRUGS if not (indications_for(d) & have)]
                if not options:
                    continue
                drug = options[int(rng.integers(0, len(options)))]
                row = adm.iloc[0]
                new_rx.append((sid, int(row.hadm_id), drug, row.admittime + pd.Timedelta(hours=2)))
                detail = drug

            elif t == "K1":
                have = [c for c in sorted(dx_by_sid.get(sid, set())) if CONDITIONS[c].effects]
                if not have:
                    continue
                code = have[int(rng.integers(0, len(have)))]
                m = e.measurements
                for eff in CONDITIONS[code].effects:
                    lo, hi = MEASUREMENTS[eff.measurement].normal_for(sex)
                    mid = (lo + hi) / 2
                    sel = (m.subject_id == sid) & (m.label == eff.measurement)
                    # values drawn from the 70% of the reference interval furthest from
                    # the expected direction: most, but not all, sit on the "wrong" side
                    # of the midpoint, so some injected cases are genuinely ambiguous
                    span = 0.7 * (hi - lo)
                    if eff.direction == "high":
                        vals = rng.uniform(lo, lo + span, sel.sum())
                    else:
                        vals = rng.uniform(hi - span, hi, sel.sum())
                    m.loc[sel, "valuenum"] = np.round(vals, 2)
                detail = code

            used.add(sid)
            per_type[t] -= 1
            truth.append((sid, t, INCONSISTENCY_TYPES[t][0], detail))

    if new_meas:
        e.measurements = pd.concat([e.measurements, pd.DataFrame(new_meas)], ignore_index=True)
    if new_dx:
        e.diagnoses_icd = pd.concat(
            [e.diagnoses_icd, pd.DataFrame(new_dx, columns=e.diagnoses_icd.columns)], ignore_index=True
        )
    if new_rx:
        e.prescriptions = pd.concat(
            [e.prescriptions, pd.DataFrame(new_rx, columns=e.prescriptions.columns)], ignore_index=True
        )
    gt = pd.DataFrame(truth, columns=["subject_id", "type", "category", "detail"])
    return e, gt
