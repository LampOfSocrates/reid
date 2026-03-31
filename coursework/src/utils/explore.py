# Copyright (c) EEEM071, University of Surrey

from __future__ import annotations

import re
from pathlib import Path

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


def show_veri_good_and_junk(
    root: str | Path,
    query_index: int,
    gt_file: str | Path | None = None,
    jk_file: str | Path | None = None,
    index_is_one_based: bool = False,
    gallery_indices_are_one_based: bool = True,
    max_good: int = 10,
    max_junk: int = 10,
    figsize=(18, 8),
    show_plot: bool = True,
):
    """
    Show the query image together with its good and junk gallery matches.
    """
    root = Path(root)
    gt_file = Path(gt_file) if gt_file else root / "gt_index.txt"
    jk_file = Path(jk_file) if jk_file else root / "jk_index.txt"

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

    ncols = max(1 + len(good_files), 1 + len(junk_files))
    fig, axes = plt.subplots(2, ncols, figsize=figsize)
    if ncols == 1:
        axes = axes.reshape(2, 1)

    for axis in axes.ravel():
        axis.axis("off")

    query_img = Image.open(query_file).convert("RGB")
    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title(f"QUERY\n{query_file.name}", fontsize=10)

    for offset, img_path in enumerate(good_files, start=1):
        image = Image.open(img_path).convert("RGB")
        axes[0, offset].imshow(image)
        axes[0, offset].set_title(f"GOOD {offset}\n{img_path.name}", fontsize=10)

    axes[1, 0].imshow(query_img)
    axes[1, 0].set_title(f"QUERY\n{query_file.name}", fontsize=10)

    for offset, img_path in enumerate(junk_files, start=1):
        image = Image.open(img_path).convert("RGB")
        axes[1, offset].imshow(image)
        axes[1, offset].set_title(f"JUNK {offset}\n{img_path.name}", fontsize=10)

    plt.tight_layout()
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
