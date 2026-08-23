import os
import sys
import argparse
import numpy as np

"""
python scripts/filter_embeddings_by_gestures.py \
  -f limu_v1 --dataset Edery --version 20_120 --remove 3,9
"""

# ensure repo root is on sys.path so local modules (embedding, etc.) can be imported
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from embedding import load_embedding_label

def extract_gesture_info(labels):
    if len(labels.shape) == 3:
        return labels[:, 0, 0].astype(int)
    else:
        return labels.astype(int)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--embedding_model', '-f', required=True, help='existing embedding prefix (e.g. limu_v1)')
    p.add_argument('--dataset', required=True)
    p.add_argument('--version', required=True)
    p.add_argument('--remove', required=True, help='Comma-separated gesture IDs to remove, e.g. 2,4')
    p.add_argument('--out_prefix', default=None, help='Output prefix (default: <input>_filtered)')
    args = p.parse_args()

    remove_ids = [int(x) for x in args.remove.split(',') if x.strip()]
    out_prefix = args.out_prefix or (args.embedding_model + '_filtered')

    print(f"Loading embeddings for {args.embedding_model} / {args.dataset} / {args.version} ...")
    embeddings, labels = load_embedding_label(args.embedding_model, args.dataset, args.version)

    gesture_ids = extract_gesture_info(labels)
    mask_keep = ~np.isin(gesture_ids, remove_ids)
    kept = mask_keep.sum()
    total = len(gesture_ids)
    print(f"Keeping {kept}/{total} samples (removed gestures: {remove_ids})")

    embeddings_f = embeddings[mask_keep]
    labels_f = labels[mask_keep]

    out_dir = 'embed'
    os.makedirs(out_dir, exist_ok=True)

    emb_out = os.path.join(out_dir, f"embed_{out_prefix}_{args.dataset}_{args.version}.npy")
    lab_out = os.path.join(out_dir, f"label_{out_prefix}_{args.dataset}_{args.version}.npy")

    np.save(emb_out, embeddings_f.astype(np.float32))
    np.save(lab_out, labels_f.astype(np.int64))

    print(f"Saved filtered embeddings -> {emb_out}")
    print(f"Saved filtered labels -> {lab_out}")
    print(f"Use embedding prefix: {out_prefix} when running classifier")

if __name__ == "__main__":
    main()