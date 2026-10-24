"""Build reports/eval_compare_fast_minibatch.md from reports/eval.json."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.compare_runs import load_eval_rollup

try:
    from evaluation.plot_compare import write_compare_figures as _write_compare_figures
except ImportError:
    _write_compare_figures = None  # type: ignore[misc, assignment]


def _fmt_pm(mean: float | None, std: float | None) -> str:
    if mean is None or mean != mean:
        return "—"
    if std is None or std == 0.0:
        return f"{mean:.4f}"
    return f"{mean:.4f}±{std:.4f}"


def _fmt_d(mean: float, std: float | None) -> str:
    if mean is None or mean != mean:
        return "—"
    if std is None or std == 0.0:
        return f"{mean:+.4f}"
    return f"{mean:+.4f}±{std:.4f}"


def _figure_markdown() -> list[str]:
    return [
        "## Figures\n",
        "_Δ = minibatch − fast (paired trials). Error bars when `n_trials` > 1._\n",
        "",
        "### ΔARI by dataset\n",
        "![ΔARI](figures/delta_ari.png)\n",
        "",
        "### ΔNMI by dataset\n",
        "![ΔNMI](figures/delta_nmi.png)\n",
        "",
        "### ARI and NMI vs diagonal\n",
        "![Scatter](figures/scatter_ari_nmi.png)\n",
        "",
        "### Win counts (rollup)\n",
        "![Win counts](figures/win_counts.png)\n",
        "",
    ]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"
    comp, meta = load_eval_rollup(reports)

    figure_lines: list[str] = []
    if _write_compare_figures is not None:
        try:
            _write_compare_figures(reports)
            figure_lines = _figure_markdown()
        except Exception as e:
            print(f"generate_eval_compare_report: figures skipped ({e})", flush=True)

    lines: list[str] = []
    lines.append("# Clustering evaluation: fast_kmeans vs minibatch_kmeans\n")
    lines.append(
        "Comparative report from `python -m evaluation.run` on the default benchmark tasks.\n"
    )
    lines.append("## Run configuration\n")
    lines.append("| Setting | Value |")
    lines.append("|---------|-------|")
    lines.append("| Backends | `fastkmeans`, `minibatchkmeans` |")
    lines.append("| Iterations | `evaluation.heuristics.scaled_niter(n)` (per dataset) |")
    lines.append("| Trials | See JSON `n_trials` / `by_seed` (`--n-trials N`, seeds `seed`..`seed+N-1`) |")
    lines.append("| Data | `--data-path ./clustering-data-v1` |")
    lines.append(
        "| Minibatch batch size | `--batch-size` (default 16384; capped at `n` per dataset) |"
    )
    lines.append("")
    lines.extend(figure_lines)
    lines.append("Raw JSON: `reports/eval.json`.\n")
    lines.append(
        "When multiple seeds are present, **ΔARI / ΔNMI** use the mean (and ± stdev) of "
        "**per-seed** differences (minibatch − fast at the same seed).\n"
    )

    sample_seeds = meta.get("sample_seeds")
    sample_n_trials = meta.get("sample_n_trials")

    rows_path = reports / "eval.json"
    with open(rows_path, encoding="utf-8") as f:
        rows = json.load(f)
    dev = next((r["device"] for r in rows if r.get("device")), "?")
    niter = next((r["niter"] for r in rows if r.get("niter")), "?")
    stats_n = meta["n_tasks"]
    lines.append(f"## Results ({stats_n} datasets × 2 methods)\n")
    lines.append(f"Device **{dev}**, **{niter}** iterations.\n")
    if sample_seeds is not None:
        nt = sample_n_trials if sample_n_trials is not None else len(sample_seeds)
        lines.append(f"`n_trials` = **{nt}**, seeds **{sample_seeds}**.\n")
    w = meta["rollup_wins"]
    lines.append(
        f"**Summary (minibatch − fast):** mean ΔARI **{meta['mean_dari']:+.4f}**, "
        f"mean ΔNMI **{meta['mean_dnmi']:+.4f}**. "
        f"ARI wins — fast **{w['ari_fast']}**, minibatch **{w['ari_mb']}**, tie **{w['ari_tie']}**. "
        f"NMI wins — fast **{w['nmi_fast']}**, minibatch **{w['nmi_mb']}**, tie **{w['nmi_tie']}**.\n"
    )
    lines.append(
        "| Dataset | ARI fast | ARI mb | ΔARI | NMI fast | NMI mb | ΔNMI |"
    )
    lines.append("|---------|----------|--------|------|----------|--------|------|")
    for c in comp:
        af = _fmt_pm(c["ari_fast"], c.get("ari_fast_std"))
        am = _fmt_pm(c["ari_mb"], c.get("ari_mb_std"))
        nf = _fmt_pm(c["nmi_fast"], c.get("nmi_fast_std"))
        nm = _fmt_pm(c["nmi_mb"], c.get("nmi_mb_std"))
        d1 = _fmt_d(c["dari"], c.get("dari_std"))
        d2 = _fmt_d(c["dnmi"], c.get("dnmi_std"))
        lines.append(
            f"| {c['task']} | {af} | {am} | {d1} | {nf} | {nm} | {d2} |"
        )
    lines.append("")

    out = reports / "eval_compare_fast_minibatch.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
