"""Deterministic, interpretable rules for patient-level clinical inconsistencies.

Each rule is a small function that takes an :class:`~clincon.generator.EHR` and
returns a list of :class:`Finding` objects. Every finding carries a plain-English
explanation plus the exact evidence that triggered it, so a reviewer can verify
it without re-running the code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import pandas as pd

from .generator import EHR
from .knowledge import CONDITIONS, GLUCOSE_MMOL_PLAUSIBLE, MEASUREMENTS, indications_for


@dataclass
class Finding:
    subject_id: int
    rule: str
    category: str
    severity: str
    message: str
    hadm_id: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)


Rule = Callable[[EHR], list[Finding]]
RULES: dict[str, Rule] = {}


def rule(rule_id: str) -> Callable[[Rule], Rule]:
    def deco(fn: Rule) -> Rule:
        RULES[rule_id] = fn
        return fn

    return deco


@rule("T1")
def discharge_before_admission(e: EHR) -> list[Finding]:
    bad = e.admissions[e.admissions.dischtime < e.admissions.admittime]
    return [
        Finding(int(r.subject_id), "T1", "temporal", "high",
                f"Admission {r.hadm_id}: discharge {r.dischtime} is before admission {r.admittime}.",
                int(r.hadm_id))
        for r in bad.itertuples()
    ]


@rule("T2")
def event_after_death(e: EHR) -> list[Finding]:
    dead = e.patients.dropna(subset=["dod"])[["subject_id", "dod"]]
    m = e.measurements.merge(dead, on="subject_id")
    # date of death is day-resolution; allow events on the day of death itself
    late = m[m.charttime >= m.dod + pd.Timedelta(days=1)]
    out = []
    for sid, g in late.groupby("subject_id"):
        first = g.sort_values("charttime").iloc[0]
        out.append(Finding(int(sid), "T2", "temporal", "high",
                           f"{len(g)} measurement(s) recorded after date of death {first.dod.date()} "
                           f"(first: {first.label} at {first.charttime}).", int(first.hadm_id)))
    return out


def _dx_with_demographics(e: EHR) -> pd.DataFrame:
    return e.diagnoses_icd.merge(e.patients[["subject_id", "gender", "anchor_age"]], on="subject_id")


@rule("D1")
def sex_specific_diagnosis(e: EHR) -> list[Finding]:
    out = []
    for r in _dx_with_demographics(e).drop_duplicates(["subject_id", "icd_code"]).itertuples():
        c = CONDITIONS.get(r.icd_code)
        if c and c.sex and c.sex != r.gender:
            out.append(Finding(int(r.subject_id), "D1", "demographic", "high",
                               f"{r.icd_code} ({c.name}) is specific to sex {c.sex} but patient is {r.gender}.",
                               int(r.hadm_id)))
    return out


@rule("D2")
def age_specific_diagnosis(e: EHR) -> list[Finding]:
    out = []
    for r in _dx_with_demographics(e).drop_duplicates(["subject_id", "icd_code"]).itertuples():
        c = CONDITIONS.get(r.icd_code)
        if not c or not c.age_range:
            continue
        if c.sex and c.sex != r.gender:
            continue  # already reported by D1
        lo, hi = c.age_range
        if not lo <= r.anchor_age <= hi:
            out.append(Finding(int(r.subject_id), "D2", "demographic", "medium",
                               f"{r.icd_code} ({c.name}) expected at ages {lo}-{hi}; patient age {r.anchor_age}.",
                               int(r.hadm_id)))
    return out


def _unit_mismatch_mask(m: pd.DataFrame) -> pd.Series:
    lo, hi = GLUCOSE_MMOL_PLAUSIBLE
    g = MEASUREMENTS["glucose"]
    return (m.label == "glucose") & (m.valueuom == g.unit) & (m.valuenum < g.plausible[0]) & m.valuenum.between(lo, hi)


@rule("U1")
def unit_mismatch(e: EHR) -> list[Finding]:
    m = e.measurements
    out = []
    for r in m[_unit_mismatch_mask(m)].itertuples():
        out.append(Finding(int(r.subject_id), "U1", "plausibility", "medium",
                           f"Glucose {r.valuenum} {r.valueuom} at {r.charttime} is impossible in mg/dL but "
                           f"plausible in mmol/L (= {r.valuenum * 18:.0f} mg/dL); likely unit error.",
                           int(r.hadm_id)))
    return out


@rule("P1")
def implausible_value(e: EHR) -> list[Finding]:
    m = e.measurements
    lo = m.label.map(lambda x: MEASUREMENTS[x].plausible[0])
    hi = m.label.map(lambda x: MEASUREMENTS[x].plausible[1])
    bad = m[((m.valuenum < lo) | (m.valuenum > hi)) & ~_unit_mismatch_mask(m)]
    return [
        Finding(int(r.subject_id), "P1", "plausibility", "high",
                f"{r.label} = {r.valuenum} {r.valueuom} at {r.charttime} is outside the physiologically "
                f"plausible range {MEASUREMENTS[r.label].plausible}.", int(r.hadm_id))
        for r in bad.itertuples()
    ]


@rule("C1")
def conflicting_duplicates(e: EHR, rel_tol: float = 0.10) -> list[Finding]:
    m = e.measurements
    g = m.groupby(["subject_id", "hadm_id", "label", "charttime"]).valuenum.agg(["min", "max", "count"])
    g = g[(g["count"] > 1) & ((g["max"] - g["min"]) / g["max"].abs() > rel_tol)].reset_index()
    return [
        Finding(int(r.subject_id), "C1", "integrity", "medium",
                f"{r.label} has {r['count']} conflicting values ({r['min']} vs {r['max']}) at {r.charttime}.",
                int(r.hadm_id))
        for _, r in g.iterrows()
    ]


@rule("M1")
def drug_without_indication(e: EHR) -> list[Finding]:
    dx = e.diagnoses_icd.groupby("subject_id").icd_code.apply(set).to_dict()
    out = []
    for r in e.prescriptions.drop_duplicates(["subject_id", "drug"]).itertuples():
        ind = indications_for(r.drug)
        if ind and not ind & dx.get(r.subject_id, set()):
            out.append(Finding(int(r.subject_id), "M1", "pharmacological", "medium",
                               f"{r.drug} prescribed but none of its indications {sorted(ind)} is coded.",
                               int(r.hadm_id)))
    return out


def run_rules(e: EHR, only: list[str] | None = None) -> list[Finding]:
    findings: list[Finding] = []
    for rid, fn in RULES.items():
        if only is None or rid in only:
            findings.extend(fn(e))
    return findings
