"""Detection pipeline, unsupervised baseline and evaluation against ground truth."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .causal_graph import causal_findings, evidence_scores
from .generator import EHR
from .injector import INCONSISTENCY_TYPES
from .rules import run_rules


def detect(e: EHR, causal_threshold: float = 0.0) -> pd.DataFrame:
    findings = run_rules(e) + causal_findings(e, threshold=causal_threshold)
    cols = ["subject_id", "rule", "category", "severity", "message", "hadm_id"]
    return pd.DataFrame([f.as_dict() for f in findings], columns=cols)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def per_type_metrics(findings: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Patient-level precision/recall/F1 for every inconsistency type."""
    rows = []
    for t, (cat, desc) in INCONSISTENCY_TYPES.items():
        gt = set(truth.loc[truth.type == t, "subject_id"])
        pred = set(findings.loc[findings.rule == t, "subject_id"])
        tp, fp, fn = len(gt & pred), len(pred - gt), len(gt - pred)
        p, r, f = _prf(tp, fp, fn)
        rows.append((t, cat, desc, len(gt), tp, fp, fn, p, r, f))
    return pd.DataFrame(rows, columns=["type", "category", "description", "n_injected", "tp", "fp", "fn",
                                       "precision", "recall", "f1"])


def patient_level(flagged: set[int], truth: pd.DataFrame, all_ids) -> dict:
    gt = set(truth.subject_id)
    tp, fp, fn = len(gt & flagged), len(flagged - gt), len(gt - flagged)
    p, r, f = _prf(tp, fp, fn)
    return {"patients": len(all_ids), "injected": len(gt), "flagged": len(flagged),
            "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


# ---------------------------------------------------------------------------
# Unsupervised baseline: Isolation Forest on patient-level aggregate features
# ---------------------------------------------------------------------------
def patient_features(e: EHR) -> pd.DataFrame:
    m = e.measurements
    agg = m.groupby(["subject_id", "label"]).valuenum.agg(["min", "max", "mean", "count"]).unstack("label")
    agg.columns = [f"{lab}_{stat}" for stat, lab in agg.columns]
    p = e.patients.set_index("subject_id")
    feats = pd.DataFrame(index=p.index)
    feats["age"] = p.anchor_age
    feats["female"] = (p.gender == "F").astype(int)
    feats["dead"] = p.dod.notna().astype(int)
    adm = e.admissions.assign(los_h=(e.admissions.dischtime - e.admissions.admittime).dt.total_seconds() / 3600)
    feats["min_los_h"] = adm.groupby("subject_id").los_h.min()
    feats["n_adm"] = adm.groupby("subject_id").size()
    last_event = m.groupby("subject_id").charttime.max()
    feats["days_event_after_dod"] = ((last_event - p.dod).dt.total_seconds() / 86400).fillna(-999)
    dx = pd.crosstab(e.diagnoses_icd.subject_id, e.diagnoses_icd.icd_code).clip(upper=1)
    rx = pd.crosstab(e.prescriptions.subject_id, e.prescriptions.drug).clip(upper=1).add_prefix("rx_")
    feats = feats.join(agg).join(dx).join(rx).fillna(0)
    return feats


def isolation_forest_flags(e: EHR, contamination: float, seed: int = 0) -> set[int]:
    X = patient_features(e)
    Xs = (X - X.mean()) / X.std().replace(0, 1)
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed)
    pred = model.fit_predict(Xs.values)
    return set(X.index[pred == -1])


def causal_threshold_sweep(e: EHR, truth: pd.DataFrame, thresholds=None) -> pd.DataFrame:
    thresholds = np.round(np.arange(-1.0, 1.01, 0.1), 2) if thresholds is None else thresholds
    scores = evidence_scores(e)
    gt = set(truth.loc[truth.type == "K1", "subject_id"])
    rows = []
    for th in thresholds:
        pred = {f.subject_id for f in causal_findings(e, threshold=th, scores=scores)}
        p, r, f = _prf(len(gt & pred), len(pred - gt), len(gt - pred))
        rows.append((th, p, r, f, len(pred)))
    return pd.DataFrame(rows, columns=["threshold", "precision", "recall", "f1", "n_flagged"])
