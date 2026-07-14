"""Aggregate VTWAM LIBERO and LIBERO-PRO rollout CSVs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


BASE_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
PERTS = ["object", "swap", "language", "task", "environment"]
PERT_LABELS = {
    "object": "Object",
    "swap": "Position / Swap",
    "language": "Semantic",
    "task": "Task",
    "environment": "Environment",
}


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarize(path: Path) -> dict:
    rows = read_rows(path)
    succ = sum(int(r["success"]) for r in rows)
    episodes = sum(int(r["episodes"]) for r in rows)
    weighted_steps = sum(float(r.get("mean_steps", 0.0)) * int(r["episodes"]) for r in rows)
    return {
        "success": succ,
        "episodes": episodes,
        "success_rate": succ / max(episodes, 1),
        "mean_steps": weighted_steps / max(episodes, 1),
        "tasks": len(rows),
    }


def fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def write_libero_summary(libero_dir: Path, summary_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for suite in BASE_SUITES:
        stats = summarize(libero_dir / f"eval_{suite}.csv")
        rows.append({"suite": suite, **stats})

    out = summary_dir / "libero_summary.csv"
    with open(out, "w", newline="") as f:
        fieldnames = ["suite", "success", "episodes", "success_rate", "mean_steps", "tasks"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return rows


def write_pro_summary(pro_dir: Path, summary_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for suite in BASE_SUITES:
        for pert in PERTS:
            stats = summarize(pro_dir / f"eval_pro_{suite}_{pert}.csv")
            rows.append({"suite": suite, "condition": pert, **stats})

    out = summary_dir / "libero_pro_matrix.csv"
    with open(out, "w", newline="") as f:
        fieldnames = ["suite", "condition", "success", "episodes", "success_rate", "mean_steps", "tasks"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    lines = ["# VTWAM LIBERO-PRO Summary", ""]
    lines.append("| Suite | Object | Position / Swap | Semantic | Task | Environment |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for suite in BASE_SUITES:
        by_cond = {r["condition"]: r for r in rows if r["suite"] == suite}
        cells = [fmt_pct(by_cond[p]["success_rate"]) for p in PERTS]
        lines.append(f"| {suite} | " + " | ".join(cells) + " |")

    lines.append("")
    for pert in PERTS:
        vals = [r["success_rate"] for r in rows if r["condition"] == pert]
        lines.append(f"- **{PERT_LABELS[pert]}**: mean {sum(vals) / len(vals) * 100:.1f}%")
    all_vals = [r["success_rate"] for r in rows]
    lines.append(f"- **Full PRO Avg**: {sum(all_vals) / len(all_vals) * 100:.2f}%")
    (summary_dir / "libero_pro_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="logs/vtwam_eval")
    args = p.parse_args()
    root = Path(args.root)
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    libero_rows = write_libero_summary(root / "libero", summary_dir)
    pro_rows = write_pro_summary(root / "libero_pro", summary_dir)

    avg_libero = sum(r["success_rate"] for r in libero_rows) / len(libero_rows)
    avg_pro = sum(r["success_rate"] for r in pro_rows) / len(pro_rows)
    print(f"[done] wrote {summary_dir / 'libero_summary.csv'}")
    print(f"[done] wrote {summary_dir / 'libero_pro_matrix.csv'}")
    print(f"[done] wrote {summary_dir / 'libero_pro_summary.md'}")
    print(f"[summary] LIBERO avg={avg_libero * 100:.2f}% PRO avg={avg_pro * 100:.2f}%")


if __name__ == "__main__":
    main()
