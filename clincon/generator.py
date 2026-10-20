"""Synthetic EHR generator with a MIMIC-IV-inspired relational schema.

Tables produced (column names follow MIMIC-IV conventions where practical):

* ``patients``       subject_id, gender, anchor_age, dod
* ``admissions``     hadm_id, subject_id, admittime, dischtime
* ``diagnoses_icd``  subject_id, hadm_id, icd_code, icd_version
* ``prescriptions``  subject_id, hadm_id, drug, starttime
* ``measurements``   subject_id, hadm_id, label, valuenum, valueuom, charttime, kind

The generator produces *internally consistent* records. Inconsistencies are added
afterwards by :mod:`clincon.injector`, which keeps a ground-truth log so detectors
can be evaluated. No real patient data is used anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .knowledge import CONDITIONS, MEASUREMENTS, SUPPORTIVE_DRUGS


@dataclass
class EHR:
    patients: pd.DataFrame
    admissions: pd.DataFrame
    diagnoses_icd: pd.DataFrame
    prescriptions: pd.DataFrame
    measurements: pd.DataFrame

    def copy(self) -> "EHR":
        return EHR(*(getattr(self, t).copy() for t in self.tables()))

    @staticmethod
    def tables() -> tuple[str, ...]:
        return ("patients", "admissions", "diagnoses_icd", "prescriptions", "measurements")

    def to_csv(self, directory) -> None:
        from pathlib import Path

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for t in self.tables():
            getattr(self, t).to_csv(d / f"{t}.csv", index=False)


# Fraction of diagnosed patients whose labs are currently controlled / resolved
# (e.g. treated anaemia). These are *not* errors and are a realistic source of
# false positives for evidence-based checks.
RESOLVED_FRACTION = 0.06
TREATMENT_PROB = 0.85


def _normal_value(rng: np.random.Generator, name: str, sex: str) -> float:
    m = MEASUREMENTS[name]
    lo, hi = m.normal_for(sex)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    return float(rng.normal(mid, half / 2.2))


def _clip(name: str, v: float) -> float:
    lo, hi = MEASUREMENTS[name].plausible
    return float(np.clip(v, lo, hi))


def generate(n_patients: int = 2000, seed: int = 42) -> EHR:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2019-01-01")

    patients, admissions, dx, rx, meas = [], [], [], [], []
    hadm_counter = 20_000_000

    for i in range(n_patients):
        sid = 10_000_000 + i
        sex = "F" if rng.random() < 0.52 else "M"
        age = int(rng.integers(18, 91))

        # --- conditions (eligibility-aware) ---------------------------------
        conds = []
        for code, c in CONDITIONS.items():
            if c.sex and c.sex != sex:
                continue
            if c.age_range and not (c.age_range[0] <= age <= c.age_range[1]):
                continue
            if rng.random() < c.prevalence:
                conds.append(code)
        resolved = {c for c in conds if CONDITIONS[c].effects and rng.random() < RESOLVED_FRACTION}

        # --- admissions -----------------------------------------------------
        t = base + pd.Timedelta(days=int(rng.integers(0, 4 * 365)))
        n_adm = int(rng.integers(1, 4))
        last_disch = None
        for _ in range(n_adm):
            hadm_counter += 1
            hadm = hadm_counter
            admit = t + pd.Timedelta(hours=int(rng.integers(0, 24)))
            los_h = int(rng.integers(24, 14 * 24))
            disch = admit + pd.Timedelta(hours=los_h)
            admissions.append((hadm, sid, admit, disch))
            last_disch = disch

            for code in conds:
                dx.append((sid, hadm, code, 10))
            # treatments
            for code in conds:
                tx = CONDITIONS[code].treatments
                if tx and rng.random() < TREATMENT_PROB:
                    drug = tx[int(rng.integers(0, len(tx)))]
                    rx.append((sid, hadm, drug, admit + pd.Timedelta(hours=int(rng.integers(1, 12)))))
            for drug in SUPPORTIVE_DRUGS:
                if rng.random() < 0.3:
                    rx.append((sid, hadm, drug, admit + pd.Timedelta(hours=int(rng.integers(1, los_h)))))

            # measurements
            effect_of = {
                e.measurement: e
                for code in conds if code not in resolved
                for e in CONDITIONS[code].effects
            }
            for name, m in MEASUREMENTS.items():
                n_obs = int(rng.integers(2, 5)) if m.kind == "vital" else int(rng.integers(1, 3))
                # distinct chart times so that genuine duplicates never occur by chance
                minutes = rng.choice(np.arange(30, los_h * 60 - 30), size=n_obs, replace=False)
                for minute in minutes:
                    if name in effect_of:
                        e = effect_of[name]
                        v = rng.normal(e.gen_mean, e.gen_sd)
                    else:
                        v = _normal_value(rng, name, sex)
                    ct = admit + pd.Timedelta(minutes=int(minute))
                    meas.append((sid, hadm, name, round(_clip(name, v), 2), m.unit, ct, m.kind))

            t = disch + pd.Timedelta(days=int(rng.integers(20, 400)))

        # --- death ----------------------------------------------------------
        dod = pd.NaT
        if rng.random() < 0.08:
            dod = (last_disch + pd.Timedelta(days=int(rng.integers(0, 365)))).normalize() + pd.Timedelta(days=1)
        patients.append((sid, sex, age, dod))

    return EHR(
        patients=pd.DataFrame(patients, columns=["subject_id", "gender", "anchor_age", "dod"]),
        admissions=pd.DataFrame(admissions, columns=["hadm_id", "subject_id", "admittime", "dischtime"]),
        diagnoses_icd=pd.DataFrame(dx, columns=["subject_id", "hadm_id", "icd_code", "icd_version"]),
        prescriptions=pd.DataFrame(rx, columns=["subject_id", "hadm_id", "drug", "starttime"]),
        measurements=pd.DataFrame(
            meas, columns=["subject_id", "hadm_id", "label", "valuenum", "valueuom", "charttime", "kind"]
        ),
    )
