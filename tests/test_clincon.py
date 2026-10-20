import unittest

import pandas as pd

from clincon.causal_graph import build_graph, causal_findings
from clincon.evaluate import detect, per_type_metrics
from clincon.generator import generate
from clincon.injector import INCONSISTENCY_TYPES, inject
from clincon.rules import run_rules


class TestClinCon(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = generate(n_patients=400, seed=1)
        cls.corrupted, cls.truth = inject(cls.clean, rate=0.3, seed=2)

    def test_schema(self):
        self.assertEqual(self.clean.patients.subject_id.is_unique, True)
        self.assertTrue((self.clean.admissions.dischtime > self.clean.admissions.admittime).all())

    def test_clean_data_has_no_rule_violations(self):
        self.assertEqual(run_rules(self.clean), [])

    def test_injection_is_logged_once_per_patient(self):
        self.assertTrue(self.truth.subject_id.is_unique)
        self.assertEqual(set(self.truth.type), set(INCONSISTENCY_TYPES))

    def test_every_rule_based_type_is_recovered(self):
        m = per_type_metrics(detect(self.corrupted), self.truth).set_index("type")
        for t in ["T1", "T2", "D1", "D2", "P1", "U1", "C1", "M1"]:
            self.assertEqual(m.at[t, "recall"], 1.0, t)
            self.assertEqual(m.at[t, "precision"], 1.0, t)

    def test_unit_mismatch_not_double_counted_as_implausible(self):
        f = detect(self.corrupted)
        u1 = set(self.truth.loc[self.truth.type == "U1", "subject_id"])
        p1_on_u1 = f[(f.rule == "P1") & f.subject_id.isin(u1)]
        self.assertTrue(p1_on_u1.empty)

    def test_knowledge_graph_is_acyclic(self):
        import networkx as nx

        self.assertTrue(nx.is_directed_acyclic_graph(build_graph()))

    def test_causal_check_flags_contradicting_diagnosis(self):
        e = generate(n_patients=50, seed=3)
        sid = int(e.patients.subject_id.iloc[0])
        hadm = int(e.admissions.loc[e.admissions.subject_id == sid, "hadm_id"].iloc[0])
        e.diagnoses_icd = e.diagnoses_icd[e.diagnoses_icd.subject_id != sid]
        e.diagnoses_icd = pd.concat(
            [e.diagnoses_icd, pd.DataFrame([(sid, hadm, "D64.9", 10)], columns=e.diagnoses_icd.columns)]
        )
        m = e.measurements
        m.loc[(m.subject_id == sid) & (m.label == "hemoglobin"), "valuenum"] = 17.0  # high, not low
        flagged = {f.subject_id for f in causal_findings(e) if f.rule == "K1"}
        self.assertIn(sid, flagged)


if __name__ == "__main__":
    unittest.main()
