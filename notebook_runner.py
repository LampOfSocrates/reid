from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from experiment_utils import create_run_dir, write_run_config
from plot_metrics import plot_metrics
from train import train


@contextmanager
def _pushd(path: str | Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def run_experiment(
    model_name: str,
    epochs: int,
    batch_size: int,
    data_dir: str | Path,
    output_root: str | Path,
    learning_rate: float = 0.00035,
    run_tag: str | None = None,
    timestamp: str | None = None,
    use_gpu: bool = True,
) -> dict[str, Path | float | None]:
    run_dir = create_run_dir(
        output_root=output_root,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        extra_params={"lr": learning_rate, "tag": run_tag} if run_tag else {"lr": learning_rate},
        timestamp=timestamp,
    )

    config_path = write_run_config(
        run_dir,
        {
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "data_dir": str(data_dir),
            "output_root": str(output_root),
            "learning_rate": learning_rate,
            "run_tag": run_tag,
            "use_gpu": use_gpu,
        },
    )

    repo_root = Path(__file__).resolve().parent
    metrics_name = f"metrics_{model_name}.csv"
    plot_name = f"plot_{model_name}.png"
    repo_metrics = repo_root / metrics_name
    repo_plot = repo_root / plot_name
    run_metrics = run_dir / metrics_name
    run_plot = run_dir / plot_name

    with _pushd(repo_root):
        train(model_name, epochs, batch_size, str(data_dir), use_gpu=use_gpu)

    if repo_metrics.exists():
        shutil.move(str(repo_metrics), str(run_metrics))

    if run_metrics.exists():
        plot_metrics(run_metrics, run_plot)

    if repo_plot.exists():
        try:
            repo_plot.unlink()
        except OSError:
            pass

    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "metrics_csv": str(run_metrics),
                "plot_path": str(run_plot),
                "config_path": str(config_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "run_dir": run_dir,
        "metrics_csv": run_metrics,
        "plot_path": run_plot,
        "config_path": config_path,
        "summary_path": summary_path,
    }
