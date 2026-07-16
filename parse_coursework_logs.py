import argparse
import re
import os
from pathlib import Path

import pandas as pd


LOGS_ROOT = Path(__file__).parent / "coursework" / "logs"

ARGS_FIELDS = {
    "lr":            r"lr=([\d.e+-]+)",
    "optim":         r"optim='([^']+)'",
    "train_batch_size": r"train_batch_size=(\d+)",
    "random_erase":  r"random_erase=(True|False)",
    "color_jitter":  r"color_jitter=(True|False)",
    "label_smooth":  r"label_smooth=(True|False)",
    "data_fraction": r"data_fraction=([\d.]+)",
    "max_epoch":     r"max_epoch=(\d+)",
    "arch":          r"arch='([^']+)'",
}


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    result = {"file": str(path.relative_to(LOGS_ROOT)), "size_kb": round(path.stat().st_size / 1024, 1)}

    # --- Args line: use the LAST block (handles resumed runs that append a new header) ---
    all_args = list(re.finditer(r"Args:Namespace\((.+?)\)", text))
    if all_args:
        args_str = all_args[-1].group(1)
        for key, pattern in ARGS_FIELDS.items():
            m = re.search(pattern, args_str)
            result[key] = m.group(1) if m else None
    else:
        for key in ARGS_FIELDS:
            result[key] = None

    # --- Start time: use the FIRST experiment header ---
    t_match = re.search(r"Experiment time:(\S+ \S+)", text)
    result["start_time"] = t_match.group(1) if t_match else None

    # --- Elapsed: sum all legs (original + any resumes) ---
    def hms_to_seconds(s):
        h, m, sec = map(int, s.split(":"))
        return h * 3600 + m * 60 + sec

    elapsed_matches = re.findall(r"Elapsed (\d+:\d+:\d+)", text)
    if elapsed_matches:
        total_sec = sum(hms_to_seconds(e) for e in elapsed_matches)
        h, rem = divmod(total_sec, 3600)
        m, s = divmod(rem, 60)
        result["elapsed"] = f"{h}:{m:02d}:{s:02d}"
    else:
        result["elapsed"] = None

    # --- Last CMC block ---
    # Find all occurrences of the results block and take the last one
    cmc_blocks = list(re.finditer(
        r"Computing CMC and mAP.*?mAP:\s*([\d.]+)%.*?Rank-1\s*:\s*([\d.]+)%.*?Rank-5\s*:\s*([\d.]+)%",
        text, re.DOTALL
    ))
    if cmc_blocks:
        last = cmc_blocks[-1]
        result["mAP"]    = last.group(1) + "%"
        result["rank1"]  = last.group(2) + "%"
        result["rank5"]  = last.group(3) + "%"
    else:
        result["mAP"] = result["rank1"] = result["rank5"] = None

    return result


def find_log_files(root: Path):
    return sorted(root.rglob("log_train.txt"))


COLUMNS = {
    "file":             "File",
    "start_time":       "Start",
    "elapsed":          "Elapsed",
    "optim":            "Optim",
    "lr":               "LR",
    "train_batch_size": "BS",
    "max_epoch":        "Ep",
    "data_fraction":    "Frac",
    "random_erase":     "RE",
    "color_jitter":     "CJ",
    "label_smooth":     "LS",
    "arch":             "Arch",
    "mAP":              "mAP",
    "rank1":            "R@1",
    "rank5":            "R@5",
}

BOOL_COLS = {"random_erase", "color_jitter", "label_smooth"}


def to_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(COLUMNS.keys()))
    for col in BOOL_COLS:
        df[col] = df[col].map({"True": "T", "False": "F", None: ""})
    df = df.rename(columns=COLUMNS)
    df = df.reset_index(drop=True)
    return df


PERCENT_COLS  = {"mAP", "R@1", "R@5"}
NUMERIC_COLS  = {"LR", "BS", "Ep", "Frac"}
DESC_DEFAULTS = PERCENT_COLS  # metrics sort high-to-low by default


def resolve_sort_keys(tokens: list[str]) -> list[tuple[str, bool]]:
    """Map user-supplied tokens (like 'mAP', '-R@1', 'lr') to (display_col, ascending)."""
    label_lookup = {label.lower(): label for label in COLUMNS.values()}
    label_lookup.update({key.lower(): label for key, label in COLUMNS.items()})

    resolved = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        ascending = None
        if tok.startswith("-"):
            ascending, tok = True, tok[1:]
        elif tok.startswith("+"):
            ascending, tok = False, tok[1:]
        col = label_lookup.get(tok.lower())
        if not col:
            valid = sorted(set(COLUMNS.values()))
            raise SystemExit(f"Unknown sort field: {tok!r}. Valid: {', '.join(valid)}")
        if ascending is None:
            ascending = col not in DESC_DEFAULTS
        resolved.append((col, ascending))
    return resolved


def sort_dataframe(df: pd.DataFrame, sort_keys: list[tuple[str, bool]]) -> pd.DataFrame:
    if not sort_keys:
        return df
    df = df.copy()
    sort_cols, ascending = [], []
    for col, asc in sort_keys:
        if col in PERCENT_COLS:
            tmp = f"__{col}_num"
            df[tmp] = pd.to_numeric(df[col].astype(str).str.rstrip("%"), errors="coerce")
            sort_cols.append(tmp)
        elif col in NUMERIC_COLS:
            tmp = f"__{col}_num"
            df[tmp] = pd.to_numeric(df[col], errors="coerce")
            sort_cols.append(tmp)
        else:
            sort_cols.append(col)
        ascending.append(asc)
    df = df.sort_values(by=sort_cols, ascending=ascending, kind="mergesort", na_position="last")
    df = df.drop(columns=[c for c in sort_cols if c.startswith("__")])
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Parse coursework training logs.")
    parser.add_argument(
        "--sortby",
        help="Comma-separated columns to sort by (e.g. 'mAP', 'mAP,R@1', '-Elapsed'). "
             "Prefix with '-' to force ascending, '+' for descending.",
    )
    args = parser.parse_args()
    sort_keys = resolve_sort_keys(args.sortby.split(",")) if args.sortby else []

    log_files = find_log_files(LOGS_ROOT)
    if not log_files:
        print(f"No log_train.txt files found under {LOGS_ROOT}")
        return

    rows = [parse_log(f) for f in log_files]

    complete   = [r for r in rows if r["mAP"] is not None]
    incomplete = [r for r in rows if r["mAP"] is None]

    print(f"Found {len(rows)} log files  ({len(complete)} with results, {len(incomplete)} without)\n")

    if complete:
        df = to_dataframe(complete)
        df = sort_dataframe(df, sort_keys)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 0)
        pd.set_option("display.max_colwidth", 60)
        sort_desc = ", ".join(f"{c} ({'asc' if a else 'desc'})" for c, a in sort_keys) or "default"
        print(f"=== Runs with evaluation results (sorted by: {sort_desc}) ===")
        print(df.to_string(index=True))

    if incomplete:
        print("\n=== Runs without evaluation results ===")
        inc_df = pd.DataFrame(incomplete)[["file", "start_time", "size_kb"]].rename(
            columns={"file": "File", "start_time": "Start", "size_kb": "Size (KB)"}
        )
        print(inc_df.to_string(index=False))


if __name__ == "__main__":
    main()
