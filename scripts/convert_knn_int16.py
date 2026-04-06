"""Convert kNN sidecar file from int64 to int16 to reduce disk usage ~4x."""
import sys
import torch

path = sys.argv[1]
data = torch.load(path, weights_only=False)

for i, entry in enumerate(data):
    if entry is not None and "knn_indices" in entry:
        entry["knn_indices"] = entry["knn_indices"].to(torch.int16)

torch.save(data, path)
print(f"Converted {path} — {sum(1 for e in data if e is not None)} entries to int16")
