# Copyright (c) EEEM071, University of Surrey

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from math import ceil

import matplotlib.pyplot as plt
from PIL import Image


def _read_index_line(file_path: str | Path, line_number: int) -> list[int]:
    """
    Read one line from an index file and extract integers.
    Supports space/comma/tab separated formats.
    """
    file_path = Path(file_path)
    lines = file_path.read_text(encoding="utf-8").splitlines()

    if line_number < 0 or line_number >= len(lines):
        raise IndexError(
            f"Index {line_number} out of range for {file_path.name} (has {len(lines)} lines)."
        )

    line = lines[line_number].strip()
    if not line:
        return []

    return [int(value) for value in re.findall(r"-?\d+", line)]


def _resolve_query_and_gallery(root: str | Path):
    """
    Assumes a VeRi-style layout:
      root/
        image_query/
        image_test/
        gt_index_776.txt
        jk_index_776.txt
    """
    root = Path(root)
    query_dir = root / "image_query"
    gallery_dir = root / "image_test"

    if not query_dir.exists():
        raise FileNotFoundError(f"Missing query folder: {query_dir}")
    if not gallery_dir.exists():
        raise FileNotFoundError(f"Missing gallery folder: {gallery_dir}")

    query_files = sorted(query_dir.glob("*.jpg"))
    gallery_files = sorted(gallery_dir.glob("*.jpg"))

    if not query_files:
        raise FileNotFoundError(f"No query images found in {query_dir}")
    if not gallery_files:
        raise FileNotFoundError(f"No gallery images found in {gallery_dir}")

    return query_files, gallery_files


def _format_image_label(prefix: str, img_path: Path, index: int | None = None, width: int = 18) -> str:
    stem_parts = img_path.stem.split("_")
    camera_text = ""
    if len(stem_parts) >= 2 and stem_parts[1].startswith("c"):
        camera_text = f"Camera: {stem_parts[1]}"

    header = prefix if index is None else f"{prefix} {index}"
    lines = [header, f"File: {img_path.name}"]
    if camera_text:
        lines.append(camera_text)

    wrapped_lines = []
    for line in lines:
        wrapped_lines.extend(textwrap.wrap(line, width=width) or [""])
    return "\n".join(wrapped_lines)


def _draw_image_grid(
    fig,
    grid_spec,
    title: str,
    image_paths: list[Path],
    label_prefix: str,
    max_cols: int,
    label_width: int,
    title_fontsize: int,
    label_fontsize: int,
):
    section = grid_spec.subgridspec(2, 1, height_ratios=[0.18, 0.82], hspace=0.12)
    title_ax = fig.add_subplot(section[0])
    title_ax.axis("off")
    title_ax.text(
        0.0,
        0.5,
        title,
        ha="left",
        va="center",
        fontsize=title_fontsize,
        fontweight="bold",
    )

    if not image_paths:
        empty_ax = fig.add_subplot(section[1])
        empty_ax.axis("off")
        empty_ax.text(
            0.5,
            0.5,
            "No images to display",
            ha="center",
            va="center",
            fontsize=label_fontsize,
            fontweight="semibold",
        )
        return

    cols = min(max_cols, max(1, len(image_paths)))
    rows = ceil(len(image_paths) / cols)
    image_grid = section[1].subgridspec(rows, cols, hspace=0.6, wspace=0.35)

    for slot in range(rows * cols):
        cell = image_grid[slot // cols, slot % cols].subgridspec(
            2,
            1,
            height_ratios=[0.8, 0.2],
            hspace=0.08,
        )
        image_ax = fig.add_subplot(cell[0])
        text_ax = fig.add_subplot(cell[1])
        image_ax.axis("off")
        text_ax.axis("off")
        if slot >= len(image_paths):
            continue

        img_path = image_paths[slot]
        image = Image.open(img_path).convert("RGB")
        image_ax.imshow(image)
        text_ax.text(
            0.5,
            0.98,
            _format_image_label(
                label_prefix,
                img_path,
                index=slot + 1 if label_prefix != "QUERY" else None,
                width=label_width,
            ),
            ha="center",
            va="top",
            fontsize=label_fontsize,
            fontweight="semibold",
        )


def show_veri_good_and_junk(
    root: str | Path,
    query_index: int,
    gt_file: str | Path | None = None,
    jk_file: str | Path | None = None,
    index_is_one_based: bool = False,
    gallery_indices_are_one_based: bool = True,
    max_good: int = 10,
    max_junk: int = 10,
    max_cols: int = 4,
    figsize: tuple[int, int] | None = None,
    show_plot: bool = True,
):
    """
    Show the query image together with its good and junk gallery matches.
    """
    root = Path(root)
    gt_file = Path(gt_file) if gt_file else root / "gt_index_776.txt"
    jk_file = Path(jk_file) if jk_file else root / "jk_index_776.txt"

    if not gt_file.exists():
        raise FileNotFoundError(f"Missing gt file: {gt_file}")
    if not jk_file.exists():
        raise FileNotFoundError(f"Missing junk file: {jk_file}")

    query_files, gallery_files = _resolve_query_and_gallery(root)

    q_idx = query_index - 1 if index_is_one_based else query_index
    if q_idx < 0 or q_idx >= len(query_files):
        raise IndexError(
            f"Query index {query_index} out of range. Valid range is "
            f"{1 if index_is_one_based else 0} to "
            f"{len(query_files) if index_is_one_based else len(query_files) - 1}"
        )

    good_idx = _read_index_line(gt_file, q_idx)
    junk_idx = _read_index_line(jk_file, q_idx)

    if gallery_indices_are_one_based:
        good_idx = [index - 1 for index in good_idx if index > 0]
        junk_idx = [index - 1 for index in junk_idx if index > 0]

    good_idx = [index for index in good_idx if 0 <= index < len(gallery_files)]
    junk_idx = [index for index in junk_idx if 0 <= index < len(gallery_files)]

    good_files = [gallery_files[index] for index in good_idx[:max_good]]
    junk_files = [gallery_files[index] for index in junk_idx[:max_junk]]
    query_file = query_files[q_idx]

    good_rows = max(1, ceil(max(1, len(good_files)) / max_cols))
    junk_rows = max(1, ceil(max(1, len(junk_files)) / max_cols))
    total_height = 6 + good_rows * 4 + junk_rows * 4
    total_width = max_cols * 4.5
    if figsize is None:
        figsize = (total_width, total_height)

    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.5, max(2.0, good_rows * 2.6), max(2.0, junk_rows * 2.6)],
        hspace=0.28,
    )
    fig.suptitle(
        "VeRi Query With Good And Junk Matches",
        fontsize=26,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.94,
        f"Query index: {query_index} | Good shown: {len(good_files)} / {len(good_idx)} | "
        f"Bad shown: {len(junk_files)} / {len(junk_idx)}",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="semibold",
    )

    query_section = outer[0].subgridspec(2, 1, height_ratios=[0.2, 0.8], hspace=0.08)
    query_title_ax = fig.add_subplot(query_section[0])
    query_title_ax.axis("off")
    query_title_ax.text(
        0.0,
        0.5,
        "QUERY IMAGE",
        ha="left",
        va="center",
        fontsize=20,
        fontweight="bold",
    )
    query_content = query_section[1].subgridspec(2, 1, height_ratios=[0.8, 0.2], hspace=0.08)
    query_ax = fig.add_subplot(query_content[0])
    query_text_ax = fig.add_subplot(query_content[1])
    query_ax.axis("off")
    query_text_ax.axis("off")
    query_img = Image.open(query_file).convert("RGB")
    query_ax.imshow(query_img)
    query_text_ax.text(
        0.5,
        0.98,
        _format_image_label("QUERY", query_file, width=28),
        ha="center",
        va="top",
        fontsize=15,
        fontweight="semibold",
    )

    _draw_image_grid(
        fig=fig,
        grid_spec=outer[1],
        title="GOOD MATCHES",
        image_paths=good_files,
        label_prefix="GOOD",
        max_cols=max_cols,
        label_width=24,
        title_fontsize=20,
        label_fontsize=13,
    )
    _draw_image_grid(
        fig=fig,
        grid_spec=outer[2],
        title="BAD / JUNK MATCHES",
        image_paths=junk_files,
        label_prefix="JUNK",
        max_cols=max_cols,
        label_width=24,
        title_fontsize=20,
        label_fontsize=13,
    )

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.92])
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "query_file": str(query_file),
        "good_files": [str(path) for path in good_files],
        "junk_files": [str(path) for path in junk_files],
        "num_good_total": len(good_idx),
        "num_junk_total": len(junk_idx),
    }
