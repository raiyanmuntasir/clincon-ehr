"""Knowledge-driven causal consistency checking.

Clinical knowledge is encoded as a directed acyclic graph::

    condition ──causes──▶ physiological effect (measurement, direction)
    condition ──treated_by──▶ drug

If a patient carries a diagnosis, its causal descendants should leave a trace in
the record. ClinCon-EHR scores that trace per effect and flags diagnoses whose
expected evidence is not merely absent but *points the other way* (for example an
anaemia diagnosis while every haemoglobin value sits in the upper half of the
normal range).

The per-effect evidence score is

    s = sign * (v* - mid) / half_width

where ``v*`` is the median of the patient's plausible values for that
measurement (robust to a single stray reading), ``mid`` and ``half_width`` describe the sex-specific reference
interval, and ``sign`` is +1 for "high" effects and -1 for "low" effects. ``s = 1``
means the typical value sits exactly on the reference limit in the expected
direction; ``s < 0`` means the typical value lies on the opposite side of the
reference midpoint. A diagnosis is flagged when the best score across all of its
measured effects is below ``threshold`` (default 0).

This is a consistency test against a causal *knowledge* graph, not causal
discovery or effect estimation from data.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd

from .generator import EHR
from .knowledge import CONDITIONS, MEASUREMENTS
from .rules import Finding, _unit_mismatch_mask


def build_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    for code, c in CONDITIONS.items():
        g.add_node(code, kind="condition", label=c.name)
        for e in c.effects:
            g.add_node(e.measurement, kind="measurement", unit=MEASUREMENTS[e.measurement].unit)
            g.add_edge(code, e.measurement, relation="causes", direction=e.direction)
        for d in c.treatments:
            g.add_node(d, kind="drug")
            g.add_edge(code, d, relation="treated_by")
    assert nx.is_directed_acyclic_graph(g)
    return g


def _plausible(m: pd.DataFrame) -> pd.DataFrame:
    lo = m.label.map(lambda x: MEASUREMENTS[x].plausible[0])
    hi = m.label.map(lambda x: MEASUREMENTS[x].plausible[1])
    return m[m.valuenum.between(lo, hi) & ~_unit_mismatch_mask(m)]


def evidence_scores(e: EHR, graph: nx.DiGraph | None = None) -> pd.DataFrame:
    """One row per (patient, diagnosis, effect) with its evidence score."""
    g = graph or build_graph()
    meas = _plausible(e.measurements)
    sex = e.patients.set_index("subject_id").gender.to_dict()
    dx = e.diagnoses_icd.drop_duplicates(["subject_id", "icd_code"])
    by_patient = {sid: grp for sid, grp in meas.groupby("subject_id")}

    rows = []
    for r in dx.itertuples():
        if r.icd_code not in g:
            continue
        pm = by_patient.get(r.subject_id)
        if pm is None:
            continue
        for _, target, attrs in g.out_edges(r.icd_code, data=True):
            if attrs["relation"] != "causes":
                continue
            vals = pm.loc[pm.label == target, "valuenum"]
            if vals.empty:
                continue
            lo, hi = MEASUREMENTS[target].normal_for(sex[r.subject_id])
            mid, half = (lo + hi) / 2, (hi - lo) / 2
            v = float(vals.median())
            s = (v - mid) / half if attrs["direction"] == "high" else (mid - v) / half
            rows.append((r.subject_id, r.icd_code, target, attrs["direction"], v, lo, hi, s, len(vals)))
    return pd.DataFrame(
        rows,
        columns=["subject_id", "icd_code", "effect", "direction", "median_value", "ref_lo", "ref_hi", "score", "n_obs"],
    )


def diagnosis_support(scores: pd.DataFrame) -> pd.DataFrame:
    """Best evidence score per (patient, diagnosis)."""
    return scores.groupby(["subject_id", "icd_code"], as_index=False).score.max()


def causal_findings(e: EHR, threshold: float = 0.0, scores: pd.DataFrame | None = None) -> list[Finding]:
    scores = evidence_scores(e) if scores is None else scores
    out = []
    for (sid, code), grp in scores.groupby(["subject_id", "icd_code"]):
        if grp.score.max() >= threshold:
            continue
        parts = [
            f"{r.effect} expected {r.direction}, median {r.median_value:g} "
            f"(ref {r.ref_lo:g}-{r.ref_hi:g}, score {r.score:.2f}, n={r.n_obs})"
            for r in grp.itertuples()
        ]
        out.append(Finding(int(sid), "K1", "causal", "medium",
                           f"{code} ({CONDITIONS[code].name}) is not supported by its causal evidence: "
                           + "; ".join(parts) + "."))
    return out
