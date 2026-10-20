"""Command-line entry point: ``python -m clincon --help``."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402

from .causal_graph import build_graph  # noqa: E402
from .evaluate import (  # noqa: E402
    causal_threshold_sweep,
    detect,
    isolation_forest_flags,
    patient_level,
    per_type_metrics,
)
from .generator import generate  # noqa: E402
from .injector import inject  # noqa: E402


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def run_once(n: int, seed: int, rate: float, threshold: float):
    clean = generate(n, seed)
    corrupted, truth = inject(clean, rate=rate, seed=seed + 1)
    findings = detect(corrupted, causal_threshold=threshold)
    ids = corrupted.patients.subject_id
    ours = patient_level(set(findings.subject_id), truth, ids)
    base = patient_level(isolation_forest_flags(corrupted, contamination=len(truth) / len(ids), seed=seed),
                         truth, ids)
    clean_fp = detect(clean, causal_threshold=threshold)
    return clean, corrupted, truth, findings, ours, base, clean_fp


def plot_graph(path: Path) -> None:
    g = build_graph()
    cond = [n for n, d in g.nodes(data=True) if d["kind"] == "condition" and g.out_degree(n)]
    meas = [n for n, d in g.nodes(data=True) if d["kind"] == "measurement"]
    drugs = [n for n, d in g.nodes(data=True) if d["kind"] == "drug"]
    g = g.subgraph(cond + meas + drugs)
    pos = {}
    for col, nodes in enumerate([drugs, cond, meas]):
        for i, n in enumerate(nodes):
            pos[n] = (col, -i * (len(cond) / max(len(nodes), 1)))
    fig, ax = plt.subplots(figsize=(10, 6))
    for nodes, color in [(cond, "#4c72b0"), (meas, "#55a868"), (drugs, "#c44e52")]:
        nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=color, node_size=500, ax=ax, alpha=0.85)
    causes = [(u, v) for u, v, d in g.edges(data=True) if d["relation"] == "causes"]
    treats = [(u, v) for u, v, d in g.edges(data=True) if d["relation"] == "treated_by"]
    nx.draw_networkx_edges(g, pos, edgelist=causes, ax=ax, edge_color="#55a868", arrows=True)
    nx.draw_networkx_edges(g, pos, edgelist=treats, ax=ax, edge_color="#c44e52", style="dashed", arrows=True)
    nx.draw_networkx_labels(g, pos, font_size=8, ax=ax,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.set_title("ClinCon-EHR clinical knowledge graph\n(drug ◀ treated_by ─ condition ─ causes ▶ measurement)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="clincon", description=__doc__)
    ap.add_argument("--n-patients", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--rate", type=float, default=0.2, help="fraction of patients receiving one injected error")
    ap.add_argument("--threshold", type=float, default=0.0, help="causal evidence threshold")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--save-data", action="store_true", help="also write the synthetic EHR tables as CSV")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    per_type_runs, ours_runs, base_runs, clean_fp_runs = [], [], [], []
    for i, seed in enumerate(args.seeds):
        clean, corrupted, truth, findings, ours, base, clean_fp = run_once(
            args.n_patients, seed, args.rate, args.threshold)
        per_type_runs.append(per_type_metrics(findings, truth).assign(seed=seed))
        ours_runs.append({**ours, "seed": seed})
        base_runs.append({**base, "seed": seed})
        clean_fp_runs.append({"seed": seed, "flagged_patients_in_clean_data": clean_fp.subject_id.nunique()})
        print(f"seed {seed}: ClinCon F1={ours['f1']:.3f}  IsolationForest F1={base['f1']:.3f}")
        if i == 0:  # keep artefacts from the first seed for inspection
            findings.merge(truth.rename(columns={"type": "injected_type"})[["subject_id", "injected_type"]],
                           on="subject_id", how="left").to_csv(args.out / f"findings_seed{seed}.csv", index=False)
            truth.to_csv(args.out / f"ground_truth_seed{seed}.csv", index=False)
            sweep = causal_threshold_sweep(corrupted, truth)
            sweep.to_csv(args.out / "causal_threshold_sweep.csv", index=False)
            if args.save_data:
                corrupted.to_csv(args.out / "synthetic_ehr")

    pt = pd.concat(per_type_runs)
    summary = (pt.groupby(["type", "category", "description"], sort=False)
                 .agg(n_injected=("n_injected", "mean"), precision=("precision", "mean"),
                      precision_sd=("precision", "std"), recall=("recall", "mean"),
                      recall_sd=("recall", "std"), f1=("f1", "mean"))
                 .reset_index())
    ours_df, base_df = pd.DataFrame(ours_runs), pd.DataFrame(base_runs)
    overall = pd.DataFrame({
        "method": ["ClinCon-EHR (rules + causal graph)", "Isolation Forest baseline"],
        "precision": [ours_df.precision.mean(), base_df.precision.mean()],
        "precision_sd": [ours_df.precision.std(), base_df.precision.std()],
        "recall": [ours_df.recall.mean(), base_df.recall.mean()],
        "recall_sd": [ours_df.recall.std(), base_df.recall.std()],
        "f1": [ours_df.f1.mean(), base_df.f1.mean()],
    })
    summary.to_csv(args.out / "per_type_metrics.csv", index=False)
    overall.to_csv(args.out / "overall_metrics.csv", index=False)

    # ---- figures ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(len(summary))
    ax.bar([i - 0.2 for i in x], summary.precision, 0.4, yerr=summary.precision_sd, label="precision")
    ax.bar([i + 0.2 for i in x], summary.recall, 0.4, yerr=summary.recall_sd, label="recall")
    ax.set_xticks(list(x), summary.type)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-type detection, mean ± sd over {len(args.seeds)} seeds")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(args.out / "per_type_metrics.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(sweep.recall, sweep.precision, marker="o")
    for r in sweep.itertuples():
        if r.threshold in (-0.5, 0.0, 0.5, 1.0):
            ax.annotate(f"t={r.threshold:g}", (r.recall, r.precision), fontsize=8)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Causal check (K1): threshold sweep")
    fig.tight_layout()
    fig.savefig(args.out / "causal_threshold_sweep.png", dpi=150)
    plt.close(fig)
    plot_graph(args.out / "knowledge_graph.png")

    # ---- markdown report ---------------------------------------------------
    clean_fp_df = pd.DataFrame(clean_fp_runs)
    report = [
        "# ClinCon-EHR results",
        "",
        f"Generated by `python -m clincon --n-patients {args.n_patients} --seeds {' '.join(map(str, args.seeds))}"
        f" --rate {args.rate} --threshold {args.threshold}`.",
        "",
        "## Patient-level detection (any inconsistency)",
        "",
        _md(overall),
        "",
        "## Per-type detection",
        "",
        _md(summary),
        "",
        "## False alarms on the clean (un-injected) data",
        "",
        _md(clean_fp_df),
        "",
        "## Causal-check threshold sweep (first seed)",
        "",
        _md(sweep),
        "",
    ]
    (args.out / "RESULTS.md").write_text("\n".join(report))
    print(f"wrote results to {args.out}/")


if __name__ == "__main__":
    main()
