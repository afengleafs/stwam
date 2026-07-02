"""Aggregate LIBERO-PRO results with Ori baseline from logs/eval_libero_*.csv."""
from __future__ import annotations

import csv
from pathlib import Path

STWAM_ROOT = Path(__file__).resolve().parent.parent
EVAL_PLUS = Path(__file__).resolve().parent
LOGS = EVAL_PLUS / "logs"
ORI_LOGS = STWAM_ROOT / "logs"
RESULTS = EVAL_PLUS / "results"

BASE_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
PERTS = ["object", "swap", "language", "task", "environment"]
ORI_FILES = {
    "libero_spatial": ORI_LOGS / "eval_libero_spatial.csv",
    "libero_object": ORI_LOGS / "eval_libero_object.csv",
    "libero_goal": ORI_LOGS / "eval_libero_goal.csv",
    "libero_10": ORI_LOGS / "eval_libero_10.csv",
}


def read_suite_rate(path: Path) -> float | None:
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        return None
    succ = sum(int(r["success"]) for r in rows)
    total = sum(int(r["episodes"]) for r in rows)
    return succ / max(total, 1)


def read_pro_rate(suite: str, pert: str) -> float | None:
    path = LOGS / f"eval_pro_{suite}_{pert}.csv"
    return read_suite_rate(path)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    matrix_rows: list[dict] = []

    for suite in BASE_SUITES:
        ori = read_suite_rate(ORI_FILES[suite])
        row = {"suite": suite, "condition": "ori", "success_rate": ori, "retention_vs_ori": 1.0 if ori else None}
        matrix_rows.append(row)
        for pert in PERTS:
            pro = read_pro_rate(suite, pert)
            retention = (pro / ori) if (pro is not None and ori and ori > 0) else None
            matrix_rows.append({
                "suite": suite,
                "condition": pert,
                "success_rate": pro,
                "retention_vs_ori": retention,
            })

    matrix_path = RESULTS / "libero_pro_matrix.csv"
    with open(matrix_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["suite", "condition", "success_rate", "retention_vs_ori"])
        w.writeheader()
        for r in matrix_rows:
            w.writerow(r)

    lines = ["# LIBERO-PRO Robustness Summary (STWAM)", ""]
    lines.append("| Suite | Ori | Object | Position | Semantic | Task | Environment |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    labels = {"object": "Object", "swap": "Position", "language": "Semantic", "task": "Task", "environment": "Environment"}

    for suite in BASE_SUITES:
        cells = []
        by_cond = {r["condition"]: r for r in matrix_rows if r["suite"] == suite}
        ori = by_cond.get("ori", {}).get("success_rate")
        cells.append(f"{ori*100:.1f}%" if ori is not None else "n/a")
        for pert in PERTS:
            rate = by_cond.get(pert, {}).get("success_rate")
            ret = by_cond.get(pert, {}).get("retention_vs_ori")
            if rate is None:
                cells.append("n/a")
            else:
                cells.append(f"{rate*100:.1f}% ({ret*100:.0f}% ret)" if ret is not None else f"{rate*100:.1f}%")
        lines.append(f"| {suite} | " + " | ".join(cells) + " |")

    avg_ori = [r["success_rate"] for r in matrix_rows if r["condition"] == "ori" and r["success_rate"] is not None]
    lines.extend(["", f"**Mean Ori success rate:** {sum(avg_ori)/len(avg_ori)*100:.1f}%" if avg_ori else ""])
    for pert in PERTS:
        rates = [r["success_rate"] for r in matrix_rows if r["condition"] == pert and r["success_rate"] is not None]
        rets = [r["retention_vs_ori"] for r in matrix_rows if r["condition"] == pert and r["retention_vs_ori"] is not None]
        if rates:
            lines.append(f"- **{labels[pert]}**: mean {sum(rates)/len(rates)*100:.1f}%, "
                         f"mean retention {sum(rets)/len(rets)*100:.0f}%")

    summary_path = RESULTS / "libero_pro_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {matrix_path}")
    print(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
