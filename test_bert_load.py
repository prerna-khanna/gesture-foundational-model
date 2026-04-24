#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import sys
import os

# Test if BERT loading works
print("Testing BERT model loading...")
try:
    from transformers import AutoTokenizer, AutoModel
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print("✓ Tokenizer loaded")
    
    print("Loading BERT model...")
    model = AutoModel.from_pretrained('bert-base-uncased')
    print("✓ BERT model loaded")
    
    # Test a simple forward pass
    print("Testing forward pass...")
    test_desc = ["a biking activity", "a walking activity"]
    inputs = tokenizer(test_desc, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    print(f"✓ Forward pass works, output shape: {outputs.last_hidden_state.shape}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
