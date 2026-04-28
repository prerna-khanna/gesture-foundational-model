import argparse
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATA_PATH = 'dataset/HGAG_DATA/data_20_120.npy'
DEFAULT_LABEL_PATH = 'dataset/HGAG_DATA/label_20_120.npy'
DEFAULT_MAPPING_PATH = 'dataset/HGAG_DATA/activity_mapping.csv'
DEFAULT_OUTPUT_DIR = 'dataset/HGAG_DATA/activity_plots'


def load_activity_names(mapping_path, num_activities):
    if os.path.exists(mapping_path):
        mapping = pd.read_csv(mapping_path)
        if {'activity_label', 'gesture_name'}.issubset(mapping.columns):
            names = mapping.sort_values('activity_label')['gesture_name'].tolist()
            if len(names) >= num_activities:
                return names[:num_activities]
    return [f'Activity_{i}' for i in range(num_activities)]


def detect_nucleus_batch(energies, min_nucleus_width=15, max_nucleus_width=40):
    """
    energies: np.ndarray of shape (n_sequences, seq_len)
    Returns list of [start,end] for each sequence.
    """
    batch_nucleus_points = []
    for sequence_energy in energies:
        seq = sequence_energy.astype(float)
        seq_len = len(seq)
        e_min = np.min(seq)
        e_max = np.max(seq)

        if e_max - e_min < 1e-6:
            mid = seq_len // 2
            batch_nucleus_points.append([
                max(0, mid - min_nucleus_width // 2),
                min(seq_len, mid + min_nucleus_width // 2)
            ])
            continue

        norm_energy = (seq - e_min) / (e_max - e_min)
        gradient = np.gradient(norm_energy)
        abs_gradient = np.abs(gradient)
        thresh = np.percentile(abs_gradient, 90)
        significant_changes = np.where(abs_gradient > thresh)[0]

        if len(significant_changes) < 2:
            active_idx = np.where(norm_energy > 0.5)[0]
            if len(active_idx) > 0:
                start = int(max(0, active_idx[0]))
                end = int(min(seq_len, active_idx[-1] + 1))
                if end - start < min_nucleus_width:
                    mid = (start + end) // 2
                    start = max(0, mid - min_nucleus_width // 2)
                    end = min(seq_len, mid + min_nucleus_width // 2)
                if end - start > max_nucleus_width:
                    mid = (start + end) // 2
                    start = max(0, mid - max_nucleus_width // 2)
                    end = min(seq_len, mid + max_nucleus_width // 2)
                batch_nucleus_points.append([start, end])
            else:
                mid = seq_len // 2
                batch_nucleus_points.append([
                    max(0, mid - min_nucleus_width // 2),
                    min(seq_len, mid + min_nucleus_width // 2)
                ])
        else:
            # group consecutive indices
            transitions = []
            current = [significant_changes[0]]
            for i in range(1, len(significant_changes)):
                if significant_changes[i] - significant_changes[i-1] <= 3:
                    current.append(significant_changes[i])
                else:
                    transitions.append(int(np.mean(current)))
                    current = [significant_changes[i]]
            if current:
                transitions.append(int(np.mean(current)))

            if len(transitions) >= 2:
                start = max(0, transitions[0])
                end = min(seq_len, transitions[-1])
            else:
                tr = transitions[0]
                if np.mean(norm_energy[:tr]) > np.mean(norm_energy[tr:]):
                    start = 0
                    end = min(seq_len, tr + min_nucleus_width // 2)
                else:
                    start = max(0, tr - min_nucleus_width // 2)
                    end = seq_len

            if end - start < min_nucleus_width:
                mid = (start + end) // 2
                start = max(0, mid - min_nucleus_width // 2)
                end = min(seq_len, mid + min_nucleus_width // 2)
            if end - start > max_nucleus_width:
                mid = (start + end) // 2
                start = max(0, mid - max_nucleus_width // 2)
                end = min(seq_len, mid + max_nucleus_width // 2)
            batch_nucleus_points.append([int(start), int(end)])

    return batch_nucleus_points


def detect_nucleus(energy, min_nucleus_width=15, max_nucleus_width=40):
    """
    Fixed-center nucleus detector with the same signature as the adaptive version.
    This implementation ignores the min/max args and returns a middle-50-sample
    window (25 left, 25 right) for each sequence in `energy`.

    Args:
        energy: iterable or array of shape (n_sequences, seq_len) or (seq_len,) for single sequence
        min_nucleus_width, max_nucleus_width: accepted for API compatibility but unused

    Returns:
        list of [start, end] pairs for each sequence
    """
    fixed_width = 50
    half = fixed_width // 2
    batch_nucleus_points = []
    for seq in energy:
        # try to coerce to numpy array if needed
        if hasattr(seq, 'cpu') and hasattr(seq, 'numpy'):
            try:
                seq = seq.cpu().numpy()
            except Exception:
                seq = np.asarray(seq)
        elif not isinstance(seq, np.ndarray):
            seq = np.asarray(seq)

        seq_len = len(seq)
        mid = seq_len // 2
        start = max(0, mid - half)
        end = min(seq_len, mid + half)
        batch_nucleus_points.append([int(start), int(end)])

    return batch_nucleus_points


def plot_activity_mean_std(data, labels, activity_names, output_dir, sample_size=100, nucleus_width=60):
    os.makedirs(output_dir, exist_ok=True)

    activity_ids = np.unique(labels[:, 0, 1].astype(int))
    time_steps = np.arange(data.shape[1])
    axis_names = ['X', 'Y', 'Z']
    sensor_names = ['Accelerometer', 'Gyroscope']

    for activity_id in activity_ids:
        activity_indices = np.where(labels[:, 0, 1].astype(int) == activity_id)[0]
        if activity_indices.size == 0:
            continue

        rng = np.random.default_rng(42 + int(activity_id))
        chosen_size = min(sample_size, activity_indices.size)
        sampled_indices = rng.choice(activity_indices, size=chosen_size, replace=False)
        sampled_data = data[sampled_indices]

        mean_signal = sampled_data.mean(axis=0)
        std_signal = sampled_data.std(axis=0)

        # Use fixed nucleus window centered on the sequence midpoint
        seq_len = data.shape[1]
        half = nucleus_width // 2
        mid = seq_len // 2
        agg_start = max(0, mid - half)
        agg_end = min(seq_len, mid + half)

        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        fig.suptitle(
            f'HGAG Activity: {activity_names[int(activity_id)]} (n={chosen_size})',
            fontsize=16,
            fontweight='bold',
        )

        for row_idx, axis_name in enumerate(axis_names):
            for col_idx, sensor_name in enumerate(sensor_names):
                channel_idx = row_idx if col_idx == 0 else row_idx + 3
                ax = axes[row_idx, col_idx]
                mean_values = mean_signal[:, channel_idx]
                std_values = std_signal[:, channel_idx]

                ax.plot(time_steps, mean_values, color='black', linewidth=2)
                # ax.fill_between(
                #     time_steps,
                #     mean_values - std_values,
                #     mean_values + std_values,
                #     color='gray',
                #     alpha=0.25,
                # )
                ax.set_title(f'{sensor_name} - {axis_name}', fontsize=11)
                ax.grid(True, alpha=0.25)
                if row_idx == 2:
                    ax.set_xlabel('Time step')
                if col_idx == 0:
                    ax.set_ylabel(f'{axis_name} axis')

        # Overlay aggregated nucleus region on all subplots
        for ax_row in axes:
            for ax in ax_row:
                ax.axvspan(agg_start, agg_end, color='red', alpha=0.12)
                ax.axvline(agg_start, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
                ax.axvline(agg_end, color='red', linestyle='--', linewidth=0.8, alpha=0.6)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        safe_name = str(activity_names[int(activity_id)]).replace(' ', '_').replace('/', '_')
        output_path = os.path.join(output_dir, f'{int(activity_id):02d}_{safe_name}.png')
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved plot: {output_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Plot HGAG activity mean and standard deviation figures.')
    parser.add_argument('--data-path', default=DEFAULT_DATA_PATH)
    parser.add_argument('--label-path', default=DEFAULT_LABEL_PATH)
    parser.add_argument('--mapping-path', default=DEFAULT_MAPPING_PATH)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--sample-size', type=int, default=100)
    parser.add_argument('--nucleus-width', type=int, default=60,
                        help='Width (in samples) of the fixed nucleus window centered on midpoint')
    return parser.parse_args()


def main():
    args = parse_args()
    data = np.load(args.data_path)
    labels = np.load(args.label_path)
    activity_names = load_activity_names(args.mapping_path, int(labels[:, 0, 1].max()) + 1)
    plot_activity_mean_std(data, labels, activity_names, args.output_dir,
                           sample_size=args.sample_size, nucleus_width=args.nucleus_width)


if __name__ == '__main__':
    main()