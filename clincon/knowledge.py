"""Clinical knowledge base used by both the synthetic generator and the detectors.

Everything here is deliberately small, explicit and human-readable so that every
finding ClinCon-EHR produces can be traced back to a single line of clinical
knowledge. Reference ranges are typical adult values; they are illustrative and
are not intended for clinical use.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Measurements (labs + vitals)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Measurement:
    name: str
    unit: str
    normal: tuple[float, float]          # adult reference interval
    plausible: tuple[float, float]       # physiologically survivable / recordable
    kind: str = "lab"                    # "lab" or "vital"
    normal_female: tuple[float, float] | None = None

    def normal_for(self, sex: str) -> tuple[float, float]:
        if sex == "F" and self.normal_female is not None:
            return self.normal_female
        return self.normal


MEASUREMENTS: dict[str, Measurement] = {
    m.name: m
    for m in [
        Measurement("glucose", "mg/dL", (70, 110), (20, 2000)),
        Measurement("hba1c", "%", (4.0, 5.6), (3.0, 20.0)),
        Measurement("creatinine", "mg/dL", (0.6, 1.2), (0.1, 25.0)),
        Measurement("hemoglobin", "g/dL", (13.5, 17.5), (2.0, 25.0), normal_female=(12.0, 15.5)),
        Measurement("potassium", "mmol/L", (3.5, 5.0), (1.5, 10.0)),
        Measurement("sodium", "mmol/L", (135, 145), (100, 190)),
        Measurement("tsh", "mIU/L", (0.4, 4.0), (0.005, 200.0)),
        Measurement("wbc", "K/uL", (4.0, 11.0), (0.1, 300.0)),
        Measurement("heart_rate", "bpm", (60, 100), (20, 300), kind="vital"),
        Measurement("sbp", "mmHg", (90, 130), (40, 300), kind="vital"),
        Measurement("temperature", "degC", (36.1, 37.5), (25.0, 45.0), kind="vital"),
        Measurement("spo2", "%", (95, 100), (50, 100), kind="vital"),
    ]
}

# Glucose is the classic unit-confusion analyte (mg/dL vs mmol/L).
GLUCOSE_MGDL_PER_MMOL = 18.0
GLUCOSE_MMOL_PLAUSIBLE = (1.1, 33.0)


# ---------------------------------------------------------------------------
# Conditions (ICD-10-CM) and their expected downstream effects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Effect:
    measurement: str
    direction: str            # "high" or "low"
    # distribution used by the *generator* for affected patients (mean, sd)
    gen_mean: float
    gen_sd: float


@dataclass(frozen=True)
class Condition:
    code: str
    name: str
    prevalence: float
    effects: tuple[Effect, ...] = ()
    treatments: tuple[str, ...] = ()
    sex: str | None = None                     # restrict to "M" / "F"
    age_range: tuple[int, int] | None = None   # inclusive
    tags: frozenset[str] = field(default_factory=frozenset)


CONDITIONS: dict[str, Condition] = {
    c.code: c
    for c in [
        Condition(
            "E11.9", "Type 2 diabetes mellitus", 0.20,
            effects=(Effect("hba1c", "high", 7.9, 1.2), Effect("glucose", "high", 175, 45)),
            treatments=("metformin", "insulin glargine"),
        ),
        Condition(
            "N18.3", "Chronic kidney disease, stage 3", 0.10,
            effects=(Effect("creatinine", "high", 1.9, 0.35),),
            treatments=("lisinopril",),
        ),
        Condition(
            "D64.9", "Anemia, unspecified", 0.12,
            effects=(Effect("hemoglobin", "low", 9.6, 1.1),),
            treatments=("ferrous sulfate",),
        ),
        Condition(
            "E03.9", "Hypothyroidism, unspecified", 0.10,
            effects=(Effect("tsh", "high", 9.5, 3.0),),
            treatments=("levothyroxine",),
        ),
        Condition(
            "I10", "Essential (primary) hypertension", 0.30,
            effects=(Effect("sbp", "high", 152, 12),),
            treatments=("lisinopril", "amlodipine"),
        ),
        Condition(
            "A41.9", "Sepsis, unspecified organism", 0.05,
            effects=(
                Effect("wbc", "high", 17.0, 4.0),
                Effect("heart_rate", "high", 112, 10),
                Effect("temperature", "high", 38.6, 0.5),
            ),
            treatments=("piperacillin-tazobactam",),
        ),
        Condition("C61", "Malignant neoplasm of prostate", 0.04, sex="M", age_range=(40, 90),
                  treatments=("leuprolide",)),
        Condition("O80", "Encounter for full-term uncomplicated delivery", 0.06, sex="F",
                  age_range=(18, 45), tags=frozenset({"pregnancy"})),
        Condition("N95.1", "Menopausal and female climacteric states", 0.08, sex="F",
                  age_range=(45, 90)),
    ]
}

# Drugs that are routinely given without a specific coded indication.
SUPPORTIVE_DRUGS = ("acetaminophen", "ondansetron", "heparin", "docusate")


def indications_for(drug: str) -> set[str]:
    """ICD-10 codes that justify prescribing `drug` (empty for supportive drugs)."""
    return {code for code, c in CONDITIONS.items() if drug in c.treatments}


ALL_TREATMENT_DRUGS = sorted({d for c in CONDITIONS.values() for d in c.treatments})
