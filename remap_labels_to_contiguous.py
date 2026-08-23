import sys
import os
import numpy as np

def remap_labels(src_path, out_path=None, overwrite=False):
    labels = np.load(src_path)
    # handle (N,seq,2) format -> extract gesture id column
    if labels.ndim == 3:
        ids = labels[:, 0, 0].astype(int)
    else:
        ids = labels.astype(int).reshape(-1)
    uniq = sorted(np.unique(ids))
    mapping = {int(old): new for new, old in enumerate(uniq)}
    remapped_ids = np.vectorize(lambda x: mapping[int(x)])(ids).astype(np.int64)

    # reconstruct full label array with same shape
    if labels.ndim == 3:
        new_labels = labels.copy()
        new_labels[:, 0, 0] = remapped_ids
    else:
        new_labels = remapped_ids

    if out_path is None:
        out_path = src_path.replace(".npy", "_contig.npy")

    if overwrite:
        np.save(src_path, new_labels)
        print(f"Overwrote {src_path} with remapped labels")
    else:
        np.save(out_path, new_labels)
        print(f"Saved remapped labels -> {out_path}")

    print("Mapping (old->new):", mapping)
    return mapping

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/remap_labels_to_contiguous.py <label_npy> [--overwrite]")
        sys.exit(1)
    path = sys.argv[1]
    overwrite = "--overwrite" in sys.argv
    remap_labels(path, overwrite=overwrite)