import sys
import os

path = r"C:\Users\soura\Code\2026\reid\logs\resnet50_fc512_lr0p0003_bs32-veri_20260427_081034\log_train.txt"

def tail_lines(path, n=30):
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        buf = bytearray()
        pos = size
        while pos > 0 and buf.count(b"\n") <= n:
            chunk = min(4096, pos)
            pos -= chunk
            f.seek(pos)
            buf = f.read(chunk) + buf
    lines = buf.decode("utf-8", errors="replace").splitlines()
    return lines[-n:]

lines = tail_lines(path, 30)
for i, line in enumerate(lines):
    print(f"{i:3d}: {repr(line)}")
