# Copyright (c) EEEM071, University of Surrey

from __future__ import annotations

import base64
import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import HTML, display
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


def _show_single_image(
    image_path: Path,
    figsize: tuple[int, int] | None = None,
    show_plot: bool = True,
):
    if figsize is None:
        figsize = (6, 6)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    image = Image.open(image_path).convert("RGB")
    ax.imshow(image)
    ax.set_box_aspect(image.height / image.width)

    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _image_to_data_uri(image_path: Path, max_size: tuple[int, int] = (220, 220)) -> str:
    image = Image.open(image_path).convert("RGB")
    image.thumbnail(max_size)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _display_match_table(title: str, matches: list[tuple[int, Path]], columns: int = 4):
    if not matches:
        print(f"No {title.lower()} to display.")
        return

    rows = []
    for start in range(0, len(matches), columns):
        chunk = matches[start : start + columns]
        cells = []
        for index, image_path in chunk:
            image_uri = _image_to_data_uri(image_path)
            path_text = str(image_path).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            cells.append(
                "<td style='border:1px solid #ccc;padding:10px;vertical-align:top;text-align:center;'>"
                f"<img src='{image_uri}' style='max-width:220px;max-height:220px;display:block;margin:0 auto 8px auto;'/>"
                f"<div style='font-size:13px;line-height:1.35;text-align:left;word-break:break-word;'>"
                f"<strong>Index:</strong> {index}<br>"
                f"<strong>Path:</strong> {path_text}"
                "</div></td>"
            )
        while len(cells) < columns:
            cells.append("<td style='border:1px solid #ccc;padding:10px;'></td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    html = (
        f"<div style='margin:16px 0;'>"
        f"<div style='font-weight:700;font-size:18px;margin-bottom:8px;'>{title}</div>"
        "<table style='border-collapse:collapse;width:100%;table-layout:fixed;'>"
        + "".join(rows)
        + "</table></div>"
    )
    display(HTML(html))


def show_veri_good_and_junk(
    root: str | Path,
    query_index: int,
    gt_file: str | Path | None = None,
    jk_file: str | Path | None = None,
    index_is_one_based: bool = False,
    gallery_indices_are_one_based: bool = True,
    max_good: int = 10,
    max_bad: int = 10,
    figsize: tuple[int, int] | None = None,
    show_plot: bool = True,
):
    """
    Show good and bad gallery matches for a query index.
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
    bad_idx = _read_index_line(jk_file, q_idx)

    if gallery_indices_are_one_based:
        good_idx = [index - 1 for index in good_idx if index > 0]
        bad_idx = [index - 1 for index in bad_idx if index > 0]

    good_idx = [index for index in good_idx if 0 <= index < len(gallery_files)]
    bad_idx = [index for index in bad_idx if 0 <= index < len(gallery_files)]

    good_files = [gallery_files[index] for index in good_idx[:max_good]]
    bad_files = [gallery_files[index] for index in bad_idx[:max_bad]]

    print(f"Query index: {query_index}")
    print(f"Good indices ({len(good_idx)}): {good_idx}")
    print(f"Bad indices ({len(bad_idx)}): {bad_idx}")

    good_matches = list(zip(good_idx[:max_good], good_files))
    bad_matches = list(zip(bad_idx[:max_bad], bad_files))

    _display_match_table("Good Matches", good_matches)
    _display_match_table("Bad Matches", bad_matches)
