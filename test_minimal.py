#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
print("Script started")
sys.stdout.flush()

try:
    print("Importing torch...")
    sys.stdout.flush()
    import torch
    print("✓ torch imported")
    sys.stdout.flush()
    
    print("Importing transformers...")
    sys.stdout.flush()
    from transformers import AutoTokenizer, AutoModel
    print("✓ transformers imported")
    sys.stdout.flush()
    
    print("Loading embedding...")
    sys.stdout.flush()
    from embedding import load_embedding_label
    print("✓ embedding module imported")
    sys.stdout.flush()
    
    print("Loading data...")
    sys.stdout.flush()
    embedding, labels = load_embedding_label("limu_v1_ablation_nucleus_only", "hhar", "20_120")
    print(f"✓ Data loaded: {embedding.shape}, labels: {labels.shape}")
    sys.stdout.flush()
    
    print("SUCCESS!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
