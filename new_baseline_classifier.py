#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Master script for tuning contrastive classifier across multiple datasets.

This script runs tune_classifier_with_contrastive for UTD_MHAD, INF2018, 
HGAG_DATA, and umahand datasets with proper error handling and time tracking.
"""

import sys
import os
import time
import json
from types import SimpleNamespace
from datetime import datetime, timedelta

from config import create_io_config, load_model_config, load_dataset_stats
from embedding import load_embedding_label
from classifier_with_contrastive import run_grid_search


# Configuration
MODEL_VERSION = "v2"
DATASET_VERSION = "20_120"
MODEL_FILE = "limu_v1"
TARGET = "classifier_contrastive_gru"
PREFIX = "gru"
LABEL_INDEX = 1

TRAINING_RATE = 0.8
LABEL_RATE = 0.2
BALANCE = True

# List of datasets to process
DATASETS = [
    "UTD_MHAD",
    "HGAG_DATA",
    "INF2018",
    "umahand"
]

# Keep this grid small enough to run locally, but focused on the knobs that
# matter most for the overfitting pattern we saw.
PARAM_GRID = {
    "lr": [1e-4, 3e-4],
    "weight_decay": [1e-4, 5e-4],
    "lambda2": [0.001, 0.005],
    "grad_clip_norm": [0.5, 1.0],
}

# DRY RUN: Set to True for small test run (1 epoch, small subset)
DRY_RUN = False
DRY_RUN_EPOCHS = 1
DRY_RUN_TRAIN_RATE = 0.05  # Use only 5% of training data
DRY_RUN_LABEL_RATE = 0.5   # Use only 50% of labeled data

# Results tracking
RESULTS_DIR = "results/baseline_classifier_runs"


class DatasetRunner:
    """Runner for processing individual datasets with timing and error handling."""
    
    def __init__(self, dataset_name, model_version, dataset_version, model_file, target, prefix, label_index):
        self.dataset_name = dataset_name
        self.model_version = model_version
        self.dataset_version = dataset_version
        self.model_file = model_file
        self.target = target
        self.prefix = prefix
        self.label_index = label_index
        self.start_time = None
        self.end_time = None
        self.status = "pending"
        self.error_message = None
        self.best_params = None
        self.best_acc = None
        self.best_f1 = None
        
    def run(self, training_rate, label_rate, balance, param_grid, dry_run=False):
        """Run the classifier tuning for this dataset."""
        print(f"\n{'='*80}")
        print(f"Starting dataset: {self.dataset_name}")
        print(f"{'='*80}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.start_time = time.time()
        
        try:
            # Load model config
            model_cfg = load_model_config(self.target, self.prefix, self.model_version)
            if model_cfg is None:
                raise SystemExit(f"Unable to find corresponding model config for {self.target}!")

            # Load dataset config
            dataset_cfg = load_dataset_stats(self.dataset_name, self.dataset_version)
            if dataset_cfg is None:
                raise SystemExit(f"Unable to find corresponding dataset config for {self.dataset_name}!")

            # Create arguments
            save_suffix = "_dry_run" if dry_run else ""
            args = SimpleNamespace(
                model_version=self.model_version,
                dataset=self.dataset_name,
                dataset_version=self.dataset_version,
                gpu=None,
                model_file=self.model_file,
                train_cfg="./config/train.json",
                mask_cfg="./config/mask.json",
                label_index=self.label_index,
                save_model=f"tune_{self.dataset_name.lower()}_{self.dataset_version}{save_suffix}",
                model_cfg=model_cfg,
                dataset_cfg=dataset_cfg,
            )
            args = create_io_config(args, self.dataset_name, self.dataset_version, 
                                   pretrain_model=args.model_file, target=self.target)

            # Load embeddings
            print(f"Loading embeddings for {self.dataset_name}...")
            embedding, labels = load_embedding_label(args.model_file, args.dataset, args.dataset_version)
            print(f"Loaded embeddings with shape: {embedding.shape}")

            # Use adjusted rates for dry run
            actual_training_rate = DRY_RUN_TRAIN_RATE if dry_run else training_rate
            actual_label_rate = DRY_RUN_LABEL_RATE if dry_run else label_rate
            
            print(f"Training rate: {actual_training_rate}, Label rate: {actual_label_rate}")
            
            # Run grid search
            best_params, best_acc, best_f1, results = run_grid_search(
                args,
                embedding,
                labels,
                self.label_index,
                actual_training_rate,
                actual_label_rate,
                balance=balance,
                method=self.prefix,
                param_grid=param_grid,
                dry_run=dry_run,
                dry_run_epochs=DRY_RUN_EPOCHS if dry_run else None,
            )

            self.end_time = time.time()
            self.status = "completed"
            self.best_params = best_params
            self.best_acc = best_acc
            self.best_f1 = best_f1

            print(f"\n{'='*80}")
            print(f"Completed dataset: {self.dataset_name}")
            print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Best params: {best_params}")
            print(f"Best accuracy: {best_acc:.4f}")
            print(f"Best F1: {best_f1:.4f}")
            elapsed = self.end_time - self.start_time
            print(f"Elapsed time: {self._format_time(elapsed)}")
            print(f"{'='*80}\n")

        except Exception as e:
            self.end_time = time.time()
            self.status = "failed"
            self.error_message = str(e)
            print(f"\n{'='*80}")
            print(f"ERROR in dataset: {self.dataset_name}")
            print(f"Error message: {self.error_message}")
            elapsed = self.end_time - self.start_time
            print(f"Elapsed time: {self._format_time(elapsed)}")
            print(f"{'='*80}\n")
            return False

        return True

    def get_elapsed_time(self):
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    @staticmethod
    def _format_time(seconds):
        """Format seconds to human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"

    def get_summary(self):
        """Get summary of this run."""
        return {
            "dataset": self.dataset_name,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_time": self.get_elapsed_time(),
            "best_params": self.best_params,
            "best_accuracy": float(self.best_acc) if self.best_acc else None,
            "best_f1": float(self.best_f1) if self.best_f1 else None,
            "error_message": self.error_message
        }


def estimate_total_time(dataset_runners, avg_time_per_dataset):
    """Estimate total time for all datasets."""
    completed = sum(1 for r in dataset_runners if r.status == "completed")
    pending = len(dataset_runners) - completed
    estimated_remaining = pending * avg_time_per_dataset
    return estimated_remaining


def main():
    """Main function to run tuning on all datasets."""
    
    print(f"\n{'#'*80}")
    print(f"# BASELINE CLASSIFIER TUNING - MASTER SCRIPT")
    print(f"# Datasets: {', '.join(DATASETS)}")
    print(f"# Model: {MODEL_FILE} (v{MODEL_VERSION})")
    print(f"# Target: {TARGET}")
    print(f"# Dry Run: {DRY_RUN}")
    print(f"{'#'*80}\n")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initialize runners for each dataset
    runners = []
    for dataset in DATASETS:
        runner = DatasetRunner(
            dataset_name=dataset,
            model_version=MODEL_VERSION,
            dataset_version=DATASET_VERSION,
            model_file=MODEL_FILE,
            target=TARGET,
            prefix=PREFIX,
            label_index=LABEL_INDEX
        )
        runners.append(runner)
    
    # Overall timing
    overall_start = time.time()
    
    # Process each dataset
    successful = 0
    failed = 0
    failed_datasets = []
    
    for i, runner in enumerate(runners):
        print(f"\nProcessing dataset {i+1}/{len(runners)}: {runner.dataset_name}")
        
        success = runner.run(
            training_rate=TRAINING_RATE,
            label_rate=LABEL_RATE,
            balance=BALANCE,
            param_grid=PARAM_GRID,
            dry_run=DRY_RUN
        )
        
        if success:
            successful += 1
        else:
            failed += 1
            failed_datasets.append(runner.dataset_name)
        
        # Estimate time remaining after each dataset
        if i < len(runners) - 1:
            avg_time = (time.time() - overall_start) / (i + 1)
            estimated_remaining = estimate_total_time(runners, avg_time)
            print(f"\n[TIME ESTIMATE]")
            print(f"Average time per dataset: {DatasetRunner._format_time(avg_time)}")
            print(f"Estimated time for remaining datasets: {DatasetRunner._format_time(estimated_remaining)}")
            print(f"Estimated total completion time: {DatasetRunner._format_time((time.time() - overall_start) + estimated_remaining)}\n")
    
    overall_end = time.time()
    overall_elapsed = overall_end - overall_start
    
    # Print final summary
    print(f"\n{'#'*80}")
    print(f"# FINAL SUMMARY")
    print(f"{'#'*80}")
    print(f"Total datasets processed: {len(runners)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"Failed datasets: {', '.join(failed_datasets)}")
    
    print(f"\nTotal elapsed time: {DatasetRunner._format_time(overall_elapsed)}")
    print(f"Average time per dataset: {DatasetRunner._format_time(overall_elapsed / len(runners))}")
    print(f"\nDetailed Results:")
    print(f"{'-'*80}")
    
    # Print detailed results
    for runner in runners:
        summary = runner.get_summary()
        print(f"\nDataset: {summary['dataset']}")
        print(f"  Status: {summary['status']}")
        print(f"  Elapsed time: {DatasetRunner._format_time(summary['elapsed_time'])}")
        
        if summary['status'] == 'completed':
            print(f"  Best Accuracy: {summary['best_accuracy']:.4f}")
            print(f"  Best F1: {summary['best_f1']:.4f}")
            print(f"  Best Parameters: {summary['best_params']}")
        elif summary['status'] == 'failed':
            print(f"  Error: {summary['error_message']}")
    
    # Save summary to JSON
    summary_file = os.path.join(RESULTS_DIR, f"baseline_classifier_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    summary_data = {
        "run_timestamp": datetime.now().isoformat(),
        "total_elapsed_time": overall_elapsed,
        "dry_run": DRY_RUN,
        "successful_datasets": successful,
        "failed_datasets": failed,
        "datasets": [runner.get_summary() for runner in runners]
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    print(f"\n{'-'*80}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'#'*80}\n")
    
    # Return exit code based on failures
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
