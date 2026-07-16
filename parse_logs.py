import os
import re
from datetime import datetime

LOG_DIR = r"C:\Users\soura\Code\2026\reid\logs"

TARGETS = [
    "resnet50_fc512_lr0p0003_bs128-veri_20260427_121129",
    "resnet50_fc512_lr0p0003_bs32-veri_20260427_081034",
    "resnet50_fc512_lr0p0003_bs48-veri_20260427_093630",
    "resnet50_fc512_lr0p0003_bs96-veri_20260427_105620",
    "resnet50_fc512_lr_0p00005-veri_20260427_080428",
    "resnet50_fc512_lr_0p0001-veri_20260427_092220",
    "resnet50_fc512_lr_0p001-veri_20260427_103730",
]

def parse_folder_name(name):
    lr_match = re.search(r'lr_?([\dp]+)-', name) or re.search(r'lr_?([\dp]+)_', name)
    lr_str = lr_match.group(1) if lr_match else "?"
    lr = lr_str.replace("p", ".").replace("0.", "0.") if lr_str != "?" else "?"
    # handle cases like "0p0003" -> "0.0003", "0p00005" -> "0.00005"

    bs_match = re.search(r'_bs(\d+)', name)
    bs = bs_match.group(1) if bs_match else "?"

    ts_match = re.search(r'(\d{8}_\d{6})$', name)
    if ts_match:
        ts = datetime.strptime(ts_match.group(1), "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    else:
        ts = "?"

    return lr, bs, ts

def parse_log(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # Last mAP  (matches "mAP: 59.9%")
    map_vals = re.findall(r'mAP\s*:\s*([0-9]+\.[0-9]+)%', text)
    last_map = map_vals[-1] + "%" if map_vals else "?"

    # Last Rank-1  (matches "Rank-1  : 90.5%")
    r1_vals = re.findall(r'Rank-1\s*:\s*([0-9]+\.[0-9]+)%', text)
    last_r1 = r1_vals[-1] + "%" if r1_vals else "?"

    # Last Rank-5  (matches "Rank-5  : 95.6%")
    r5_vals = re.findall(r'Rank-5\s*:\s*([0-9]+\.[0-9]+)%', text)
    last_r5 = r5_vals[-1] + "%" if r5_vals else "?"

    full_sample = text
    epoch_matches = re.findall(r'[Ee]poch\s*\[?\s*(\d+)\s*/?\s*(\d+)?\]?', full_sample)
    if epoch_matches:
        last_ep = max(int(m[0]) for m in epoch_matches)
        total_ep = epoch_matches[0][1] if epoch_matches[0][1] else "?"
    else:
        last_ep = "?"
        total_ep = "?"

    return last_map, last_r1, last_r5, last_ep, total_ep

rows = []
for folder in TARGETS:
    log_path = os.path.join(LOG_DIR, folder, "log_train.txt")
    lr, bs, ts = parse_folder_name(folder)
    size_bytes = os.path.getsize(log_path) if os.path.exists(log_path) else 0
    size_str = f"{size_bytes/1024:.0f} KB" if size_bytes < 1024*1024 else f"{size_bytes/1024/1024:.1f} MB"

    if os.path.exists(log_path):
        last_map, last_r1, last_r5, last_ep, total_ep = parse_log(log_path)
    else:
        last_map = last_r1 = last_r5 = last_ep = total_ep = "missing"

    short_name = folder.replace("resnet50_fc512_", "").split("-veri_")[0]
    rows.append((short_name, ts, size_str, lr, bs, f"{last_ep}/{total_ep}", last_map, last_r1, last_r5))

# Print table
headers = ["Run", "Time", "Size", "LR", "BS", "Epochs", "mAP", "Rank-1", "Rank-5"]
col_w = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]

def fmt_row(row):
    return " | ".join(str(v).ljust(col_w[i]) for i, v in enumerate(row))

sep = "-+-".join("-" * w for w in col_w)
print(fmt_row(headers))
print(sep)
for row in rows:
    print(fmt_row(row))
