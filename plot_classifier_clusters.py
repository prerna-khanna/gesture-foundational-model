#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cleaned standalone script to extract features from a trained contrastive classifier and plot clusters.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from config import load_dataset_stats, load_dataset_label_names

try:
    from embedding import load_embedding_label
except Exception:
    load_embedding_label = None

from utils import prepare_classifier_dataset
from contrastive.models import ContrastiveTransformerClassifier, ContrastiveGRUClassifier, ContrastiveLSTMClassifier


def load_embeddings_from_args(args):
    """Return (embeddings, labels, label_names)"""
    if args.embeddings_file and args.labels_file:
        if not os.path.exists(args.embeddings_file):
            raise RuntimeError(f"Embeddings file not found: {args.embeddings_file}")
        if not os.path.exists(args.labels_file):
            raise RuntimeError(f"Labels file not found: {args.labels_file}")
        emb = np.load(args.embeddings_file)
        labs = np.load(args.labels_file)
        return emb, labs, None

    if load_embedding_label is not None and args.model_file is not None:
        emb, labs = load_embedding_label(args.model_file, args.dataset, args.dataset_version)
        return emb, labs, None

    raise RuntimeError('No embeddings/labels provided and helper not available. Provide --embeddings_file and --labels_file, or use --checkpoint with dataset files.')


def extract_features_with_model(model, device, embeddings, batch_size=256):
    import torch
    if isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)
    model.eval()
    features = []
    with torch.no_grad():
        for i in range(0, embeddings.shape[0], batch_size):
            batch = torch.from_numpy(embeddings[i:i+batch_size]).float().to(device)
            out = model(batch, True)
            if isinstance(out, tuple) and len(out) >= 2:
                feats = out[1]
                features.append(feats.cpu().numpy())
            else:
                raise RuntimeError('Model did not return features for return_features=True')
    return np.vstack(features)


def plot_2d(points, labels, label_names=None, out_file='clusters_tsne.png', title=None):
    plt.figure(figsize=(8, 6))
    unique = np.unique(labels)
    n = unique.size
    # Choose a colormap that can supply n distinct colors
    try:
        cmap = plt.cm.get_cmap('tab20' if n <= 20 else 'hsv', n)
        colors = [cmap(i) for i in range(n)]
    except Exception:
        cmap = plt.cm.get_cmap('hsv')
        colors = [cmap(i / max(1, n - 1)) for i in range(n)]

    label_to_color = {int(l): colors[i] for i, l in enumerate(unique)}
    for lab in unique:
        lab_i = int(lab)
        idx = labels == lab
        name = None
        if label_names is not None:
            # label_names may be list indexed by class id
            try:
                # try direct index
                name = label_names[lab_i]
            except Exception:
                # fallback to string conversion
                name = str(lab_i)
        else:
            name = str(lab_i)
        plt.scatter(points[idx, 0], points[idx, 1], label=name, s=10, color=label_to_color[lab_i])
    # place legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if title:
        plt.title(title)
    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plot to {out_file}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--embeddings_file', type=str, default=None, help='.npy file with embeddings (N,... )')
    p.add_argument('--labels_file', type=str, default=None, help='.npy file with labels (N, ) or (N,1,?)')
    p.add_argument('--checkpoint', type=str, default=None, help='Path to trained model checkpoint (.pt)')
    p.add_argument('--data_file', type=str, default=None, help='.npy file with full dataset (N, seq, features) - required when using --checkpoint')
    p.add_argument('--label_file', type=str, default=None, help='.npy file with labels - required when using --checkpoint')
    p.add_argument('--training_rate', type=float, default=0.8, help='training split fraction used during training')
    p.add_argument('--label_rate', type=float, default=1.0, help='label rate used during dataset preparation')
    p.add_argument('--model_file', type=str, default=None, help='model identifier used by load_embedding_label helper')
    p.add_argument('--dataset', type=str, default=None)
    p.add_argument('--dataset_version', type=str, default=None)
    p.add_argument('--label_index', type=int, default=0)
    p.add_argument('--method', choices=['tsne', 'pca'], default='tsne')
    p.add_argument('--subsample', type=int, default=2000, help='max points to use for plotting (random sample)')
    p.add_argument('--out', type=str, default='plots/clusters_tsne.png')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--model_arch', type=str, default='transformer', help='model arch when extracting features')
    args = p.parse_args()

    label_names = None

    # Embeddings-only mode
    if not args.checkpoint:
        emb, labs, label_names = load_embeddings_from_args(args)
        labs = np.array(labs)
        if labs.ndim > 1:
            labs_flat = labs.reshape(labs.shape[0], -1)[:, 0]
        else:
            labs_flat = labs

        n = emb.shape[0]
        if args.subsample and n > args.subsample:
            idx = np.random.choice(n, size=args.subsample, replace=False)
            emb_plot = emb[idx]
            labs_plot = labs_flat[idx]
        else:
            emb_plot = emb
            labs_plot = labs_flat

    # Checkpoint mode: prepare dataset and run model on test split
    if args.checkpoint:
        if args.data_file is None or args.label_file is None:
            raise RuntimeError('--data_file and --label_file are required when using --checkpoint')
        ckpt_path = args.checkpoint
        if not os.path.exists(ckpt_path) and os.path.exists(ckpt_path + '.pt'):
            ckpt_path = ckpt_path + '.pt'
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

        data = np.load(args.data_file).astype(np.float32)
        labels = np.load(args.label_file).astype(np.float32)

        # Load checkpoint state dict early so we can compare expected input dim to provided data
        import torch
        print(f'Loading checkpoint: {ckpt_path}')
        state = torch.load(ckpt_path, map_location='cpu')
        if isinstance(state, dict) and ('model_state_dict' in state or 'state_dict' in state):
            if 'model_state_dict' in state:
                state = state['model_state_dict']
            else:
                state = state['state_dict']

        state_keys = list(state.keys()) if isinstance(state, dict) else []

        # Infer architecture from checkpoint keys if possible
        arch = args.model_arch.lower()
        inferred_arch = None
        if any(k.startswith('input_projection') or 'transformer_encoder' in k for k in state_keys):
            inferred_arch = 'transformer'
        elif any('gru' in k for k in state_keys) or any('bigru' in k for k in state_keys):
            inferred_arch = 'gru'
        elif any('lstm' in k for k in state_keys):
            inferred_arch = 'lstm'

        if inferred_arch is not None and inferred_arch != arch:
            print(f'Warning: checkpoint appears to be {inferred_arch}-based but --model_arch={arch}; switching to {inferred_arch} for loading')
            arch = inferred_arch

        # Try to infer input dim from checkpoint parameters
        inferred_input_dim = None
        try:
            if arch == 'transformer' and 'input_projection.weight' in state:
                inferred_input_dim = state['input_projection.weight'].shape[1]
            elif arch == 'gru' and any(k.endswith('weight_ih_l0') for k in state_keys):
                for k in state_keys:
                    if k.endswith('weight_ih_l0'):
                        inferred_input_dim = state[k].shape[1]
                        break
            elif arch == 'lstm' and any(k.endswith('weight_ih_l0') for k in state_keys):
                for k in state_keys:
                    if k.endswith('weight_ih_l0'):
                        inferred_input_dim = state[k].shape[1]
                        break
        except Exception:
            inferred_input_dim = None

        if inferred_input_dim is not None:
            print(f'Inferring input_dim={inferred_input_dim} from checkpoint')
            input_dim_ckpt = int(inferred_input_dim)
        else:
            input_dim_ckpt = None

        # Now decide how to treat the provided data: 2D embeddings, 3D embeddings with matching input_dim, or raw data
        if data.ndim == 2:
            print(f'Using 2D embeddings as model input: original shape {data.shape}')
            data_test = data.reshape(data.shape[0], 1, data.shape[1])
            label_test = labels.reshape(labels.shape[0], -1)[:, 0] if labels.ndim > 1 else labels
            print(f'Test split size: {data_test.shape[0]} (embeddings)')
            unique_labels = np.unique(label_test)
            num_classes = int(unique_labels.size)
        elif data.ndim == 3 and input_dim_ckpt is not None and data.shape[2] == input_dim_ckpt:
            # Provided file already matches the model input dim (N, seq_len, input_dim)
            print(f'Using 3D embeddings as model input: shape {data.shape}')
            data_test = data
            label_test = labels.reshape(labels.shape[0], -1)[:, 0] if labels.ndim > 1 else labels
            print(f'Test split size: {data_test.shape[0]} (3D embeddings)')
            unique_labels = np.unique(label_test)
            num_classes = int(unique_labels.size)
        else:
            data_train, label_train, data_vali, label_vali, data_test, label_test = prepare_classifier_dataset(
                data, labels, label_index=args.label_index, training_rate=args.training_rate, label_rate=args.label_rate,
                merge=0, seed=42, balance=False)
            print(f'Test split size: {data_test.shape[0]}')
            unique_labels = np.unique(labels[..., args.label_index])
            num_classes = int(unique_labels.size)

        # Instantiate model with inferred input dim / arch
        if arch == 'gru':
            model = ContrastiveGRUClassifier(input_dim=input_dim_ckpt, hidden_dim=128, num_classes=num_classes)
        elif arch == 'lstm':
            model = ContrastiveLSTMClassifier(input_dim=input_dim_ckpt, hidden_dim=128, num_classes=num_classes)
        else:
            model = ContrastiveTransformerClassifier(input_dim=input_dim_ckpt, hidden_dim=128, num_classes=num_classes)

        # Load weights into model
        try:
            model.load_state_dict(state)
        except Exception as e:
            try:
                model.load_state_dict(state, strict=False)
                print('Loaded checkpoint with strict=False (some keys missing or unexpected)')
            except Exception:
                raise RuntimeError(f'Failed to load checkpoint into model: {e}')

        # Extract features
        features = extract_features_with_model(model, args.device, data_test)
        labs_plot = label_test.reshape(label_test.shape[0], -1)[:, 0]

    # If not checkpoint mode, use embeddings
    else:
        features = emb_plot.reshape(emb_plot.shape[0], -1)
        labs_plot = labs_plot

    # Dimensionality reduction
    if args.method == 'pca':
        pca = PCA(n_components=2)
        points = pca.fit_transform(features)
    else:
        tsne = TSNE(n_components=2)
        points = tsne.fit_transform(features)

    # Try to load label names from dataset config if not provided
    if label_names is None and args.dataset and args.dataset_version:
        try:
            ds_cfg = load_dataset_stats(args.dataset, args.dataset_version)
            if ds_cfg is not None:
                names, nnum, desc = load_dataset_label_names(ds_cfg, args.label_index)
                if names is not None:
                    label_names = names
        except Exception:
            label_names = None

    # Normalize labels to integer numpy array
    labs_plot = np.array(labs_plot)
    try:
        labs_plot = labs_plot.astype(int)
    except Exception:
        # if labels are not directly castable, leave as is
        pass

    # Auto-correct 1-based indexing when label names are available
    if label_names is not None and isinstance(label_names, (list, tuple)):
        n_labels = len(label_names)
        min_lab = int(labs_plot.min()) if labs_plot.size > 0 else 0
        max_lab = int(labs_plot.max()) if labs_plot.size > 0 else -1
        if min_lab >= 1 and max_lab == n_labels:
            print(f'Info: detected 1-based labels (range 1..{n_labels}), converting to 0-based to match label names')
            labs_plot = (labs_plot - 1).astype(int)

    plot_2d(points, labs_plot, label_names=label_names, out_file=args.out, title=f'{args.method.upper()} of classifier features')


if __name__ == '__main__':
    main()

