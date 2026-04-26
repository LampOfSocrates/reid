# Copyright (c) EEEM071, University of Surrey
"""Utilities for rendering and comparing per-run training/eval metrics.

Reads ``comparison_summary.json`` files written by ``coursework/main.py`` and
produces inline notebook output (tables + charts) plus a self-contained HTML
report.
"""

from __future__ import annotations

import base64
import html as _html
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display

sns.set_theme(style="whitegrid")


def _latest_summary(save_dir_prefix: str | Path) -> Path:
    save_dir_prefix = Path(save_dir_prefix)
    parent = save_dir_prefix.parent
    stem = save_dir_prefix.name
    candidates = sorted(parent.glob(f"{stem}_*/comparison_summary.json"))
    if not candidates:
        raise FileNotFoundError(f"No runs found under {save_dir_prefix}_*")
    return candidates[-1]


def _load(save_dir_prefix: str | Path) -> dict:
    with open(_latest_summary(save_dir_prefix), "r", encoding="utf-8") as f:
        return json.load(f)


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_run(save_dir_prefix: str | Path) -> None:
    s = _load(save_dir_prefix)
    name = s["experiment"]
    display(Markdown(
        f"### {name}  —  best Rank-1: {s['best_rank1']:.2%} · "
        f"best mAP: {s['best_mAP']:.2%} · {s['total_elapsed_seconds']:.0f}s"
    ))

    train_df = pd.DataFrame(s["epoch_summaries"])
    eval_df = pd.DataFrame(s["eval_history"])
    merged = train_df.merge(eval_df, on="epoch", how="left")
    cols = [
        "epoch", "epoch_time_seconds", "avg_xent", "avg_htri", "avg_acc",
        "rank1", "rank5", "rank10", "rank20", "mAP",
    ]
    display(merged[[c for c in cols if c in merged.columns]].round(4))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(train_df["epoch"], train_df["avg_xent"], marker="o", label="xent")
    axes[0].plot(train_df["epoch"], train_df["avg_htri"], marker="o", label="htri")
    axes[0].set_title("Train losses"); axes[0].set_xlabel("epoch"); axes[0].legend()

    axes[1].plot(train_df["epoch"], train_df["avg_acc"], marker="o", color="green")
    axes[1].set_title("Train accuracy (%)"); axes[1].set_xlabel("epoch")

    if not eval_df.empty:
        axes[2].plot(eval_df["epoch"], eval_df["rank1"], marker="o", label="Rank-1")
        axes[2].plot(eval_df["epoch"], eval_df["mAP"], marker="o", label="mAP")
        axes[2].set_title("Eval"); axes[2].set_xlabel("epoch"); axes[2].legend()

    fig.suptitle(name)
    plt.tight_layout()
    plt.show()


def compare_runs(
    save_dir_prefixes: Iterable[str | Path],
    runtime: str | None = None,
    html: bool = False,
) -> pd.DataFrame:
    save_dir_prefixes = list(save_dir_prefixes)

    rows, train_frames, eval_frames = [], [], []
    for p in save_dir_prefixes:
        s = _load(p)
        name = s["experiment"]
        rows.append({
            "experiment": name,
            "best_rank1": s["best_rank1"], "best_mAP": s["best_mAP"],
            "final_rank1": (s["final_eval"] or {}).get("rank1"),
            "final_mAP": (s["final_eval"] or {}).get("mAP"),
            "avg_epoch_s": s["avg_epoch_time_seconds"],
            "total_s": s["total_elapsed_seconds"],
        })
        t = pd.DataFrame(s["epoch_summaries"]); t["experiment"] = name; train_frames.append(t)
        e = pd.DataFrame(s["eval_history"]); e["experiment"] = name; eval_frames.append(e)

    summary = pd.DataFrame(rows).sort_values("best_rank1", ascending=False)
    display(Markdown("### Summary"))
    display(summary.style.format({
        "best_rank1": "{:.2%}", "best_mAP": "{:.2%}",
        "final_rank1": "{:.2%}", "final_mAP": "{:.2%}",
        "avg_epoch_s": "{:.1f}", "total_s": "{:.0f}",
    }))

    train_all = pd.concat(train_frames, ignore_index=True)
    eval_all = pd.concat(eval_frames, ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    sns.lineplot(data=train_all, x="epoch", y="avg_xent", hue="experiment", marker="o", ax=axes[0, 0]); axes[0, 0].set_title("xent")
    sns.lineplot(data=train_all, x="epoch", y="avg_acc", hue="experiment", marker="o", ax=axes[0, 1]); axes[0, 1].set_title("train acc (%)")
    sns.lineplot(data=eval_all, x="epoch", y="rank1", hue="experiment", marker="o", ax=axes[1, 0]); axes[1, 0].set_title("Rank-1")
    sns.lineplot(data=eval_all, x="epoch", y="mAP", hue="experiment", marker="o", ax=axes[1, 1]); axes[1, 1].set_title("mAP")
    plt.tight_layout()
    plt.show()

    if html:
        if runtime is None:
            raise ValueError("pass runtime=RUNTIME when html=True")
        export_html(save_dir_prefixes, runtime)

    return summary


REPORTS_DIR = Path("coursework") / "reports"


def export_html(
    save_dir_prefixes: Iterable[str | Path],
    runtime: str,
    out_name: str | None = None,
) -> Path:
    save_dir_prefixes = list(save_dir_prefixes)
    if out_name is None:
        out_name = f"comparison_{runtime}.html"
    elif runtime not in out_name:
        stem = Path(out_name).stem
        suffix = Path(out_name).suffix or ".html"
        out_name = f"{stem}_{runtime}{suffix}"
    out_path = REPORTS_DIR / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows, train_frames, eval_frames, sources = [], [], [], []
    per_run_tables = []
    for p in save_dir_prefixes:
        src = _latest_summary(p)
        sources.append(src)
        s = json.loads(src.read_text(encoding="utf-8"))
        name = s["experiment"]
        rows.append({
            "experiment": name,
            "max_epoch": s.get("max_epoch"),
            "data_fraction": s.get("data_fraction"),
            "best_rank1": s["best_rank1"], "best_mAP": s["best_mAP"],
            "final_rank1": (s["final_eval"] or {}).get("rank1"),
            "final_mAP": (s["final_eval"] or {}).get("mAP"),
            "avg_epoch_s": s["avg_epoch_time_seconds"],
            "total_s": s["total_elapsed_seconds"],
        })
        t = pd.DataFrame(s["epoch_summaries"]); t["experiment"] = name; train_frames.append(t)
        e = pd.DataFrame(s["eval_history"]); e["experiment"] = name; eval_frames.append(e)

        merged = t.merge(e, on="epoch", how="left")
        cols = [c for c in [
            "epoch", "epoch_time_seconds", "avg_xent", "avg_htri", "avg_acc",
            "rank1", "rank5", "rank10", "rank20", "mAP",
        ] if c in merged.columns]
        per_run_tables.append(
            (name, src, merged[cols].round(4).to_html(index=False, classes="run-table"))
        )

    summary = pd.DataFrame(rows).sort_values("best_rank1", ascending=False)
    summary_html = (summary.style
        .format({
            "max_epoch": "{:.0f}", "data_fraction": "{:.2%}",
            "best_rank1": "{:.2%}", "best_mAP": "{:.2%}",
            "final_rank1": "{:.2%}", "final_mAP": "{:.2%}",
            "avg_epoch_s": "{:.1f}", "total_s": "{:.0f}",
        })
        .hide(axis="index").to_html())

    unique_max_epochs = sorted({r["max_epoch"] for r in rows if r["max_epoch"] is not None})
    unique_fractions = sorted({r["data_fraction"] for r in rows if r["data_fraction"] is not None})
    config_html = (
        "<p><strong>MAX_EPOCH:</strong> "
        f"{', '.join(str(v) for v in unique_max_epochs) or 'n/a'}"
        " &nbsp;·&nbsp; <strong>DATA_FRACTION:</strong> "
        f"{', '.join(f'{v:.2%}' for v in unique_fractions) or 'n/a'}"
        "</p>"
    )

    train_all = pd.concat(train_frames, ignore_index=True)
    eval_all = pd.concat(eval_frames, ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    sns.lineplot(data=train_all, x="epoch", y="avg_xent", hue="experiment", marker="o", ax=axes[0, 0]); axes[0, 0].set_title("xent")
    sns.lineplot(data=train_all, x="epoch", y="avg_acc", hue="experiment", marker="o", ax=axes[0, 1]); axes[0, 1].set_title("train acc (%)")
    sns.lineplot(data=eval_all, x="epoch", y="rank1", hue="experiment", marker="o", ax=axes[1, 0]); axes[1, 0].set_title("Rank-1")
    sns.lineplot(data=eval_all, x="epoch", y="mAP", hue="experiment", marker="o", ax=axes[1, 1]); axes[1, 1].set_title("mAP")
    plt.tight_layout()
    chart_b64 = _fig_to_b64(fig)

    sources_html = "\n".join(
        f'<li><code>{_html.escape(str(src))}</code></li>' for src in sources
    )
    runs_html = "\n".join(
        f'<h3>{_html.escape(name)}</h3>'
        f'<p class="src">source: <code>{_html.escape(str(src))}</code></p>'
        f'{table}'
        for name, src, table in per_run_tables
    )

    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Comparison – {_html.escape(runtime)}</title>
<style>
 body{{font-family:system-ui,Segoe UI,Arial;margin:24px;color:#222;max-width:1200px}}
 h1,h2,h3{{margin-top:1.4em}}
 table{{border-collapse:collapse;margin:8px 0;font-size:13px}}
 th,td{{border:1px solid #ddd;padding:4px 8px;text-align:right}}
 th{{background:#f5f5f5}}
 .src{{color:#666;font-size:12px;margin:4px 0 8px}}
 img{{max-width:100%;height:auto;border:1px solid #eee}}
 code{{background:#f4f4f4;padding:1px 4px;border-radius:3px}}
</style></head><body>
<h1>Run comparison – {_html.escape(runtime)}</h1>
<p>Generated {datetime.now().isoformat(timespec="seconds")}</p>
{config_html}

<h2>Summary</h2>
{summary_html}

<h2>Charts</h2>
<img src="data:image/png;base64,{chart_b64}" alt="comparison charts"/>

<h2>Per-run epoch metrics</h2>
{runs_html}

<h2>Source files</h2>
<ul>{sources_html}</ul>
</body></html>"""

    out_path.write_text(doc, encoding="utf-8")
    print(f"Wrote {out_path.resolve()}")
    return out_path
