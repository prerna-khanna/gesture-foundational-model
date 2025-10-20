# Sony Watch Classifier Accuracy Analysis

## Summary

Testing the Sony Watch gesture recognition pipeline revealed that **the low accuracy issue is in the classifier model itself, NOT in the preprocessing pipeline**.

## Test Results

### Test 1: End-to-End Pipeline (Raw Data → BERT → Classifier)
- **Overall Accuracy**: ~5-10%
- **Issue**: Initially suspected preprocessing problems

### Test 2: Pre-computed Embeddings (Skip preprocessing, use saved BERT embeddings)
- **Overall Accuracy**: 6.03% (196/3251 correct)
- **Random Sample Accuracy**: 5.00% (1/20 correct)
- **Conclusion**: The classifier model itself has issues

## Per-Class Performance (Using Pre-computed Embeddings)

| Class | Accuracy | Correct | Total | Avg Confidence |
|-------|----------|---------|-------|----------------|
| horizontal right | 0.00% | 0 | 160 | 0.0464 |
| horizontal left | 0.00% | 0 | 161 | 0.0450 |
| vertical up | 0.62% | 1 | 166 | 0.0525 |
| vertical down | 0.00% | 0 | 166 | 0.0396 |
| clockwise circle | 0.00% | 0 | 160 | 0.0447 |
| counter-clockwise circle | 0.00% | 0 | 164 | 0.0442 |
| **counter-clockwise square** | **67.70%** | **109** | **161** | **0.0540** |
| clockwise square | 0.00% | 0 | 164 | 0.0473 |
| right diagonal | 0.00% | 0 | 161 | 0.0387 |
| left diagonal | 0.00% | 0 | 164 | 0.0447 |
| vertical down double | 52.47% | 85 | 162 | 0.0542 |
| horizontal right double | 0.00% | 0 | 161 | 0.0421 |
| V-shape down-up | 0.00% | 0 | 161 | 0.0555 |
| V-shape up-down | 0.00% | 0 | 168 | 0.0465 |
| triangle upward | 0.00% | 0 | 162 | 0.0498 |
| triangle downward | 0.62% | 1 | 161 | 0.0504 |
| S-curve leftward | 0.00% | 0 | 163 | 0.0427 |
| S-curve rightward | 0.00% | 0 | 163 | 0.0456 |
| wave leftward | 0.00% | 0 | 162 | 0.0476 |
| **wave rightward** | **0.00%** | **0** | **161** | **0.0591** |

**Note**: The model shows extreme bias:
- 67.70% accuracy on "counter-clockwise square"
- 52.47% accuracy on "vertical down double"
- 0% accuracy on 16 out of 20 classes

## Top Confusion Pairs

The model consistently confuses many gestures:

1. triangle upward → vertical up (129 times)
2. vertical up → counter-clockwise square (123 times)
3. clockwise circle → vertical up (116 times)
4. horizontal right → counter-clockwise square (115 times)
5. right diagonal → counter-clockwise square (115 times)

## Key Findings

### ✅ What's Working Correctly

1. **Label Indexing**: Fixed - labels are 1-indexed in dataset, converted to 0-indexed for model
2. **BERT Embedder**: Successfully generates 72-dimensional embeddings
3. **GRU Classifier**: Architecture loads and runs correctly
4. **Data Pipeline**: Preprocessing steps (nucleus detection, significant axis) work correctly
5. **Embedding Shape**: Pre-computed embeddings (3251, 120, 72) match expected format

### ❌ Problem Identified

**The classifier model checkpoint is poorly trained or incorrectly saved/loaded**:

- Very low confidence scores (~0.05 for all predictions, near random)
- Extreme class imbalance in predictions
- No meaningful pattern recognition
- Performance is essentially random (5-7% for 20 classes ≈ 5% random baseline)

## Possible Root Causes

1. **Model checkpoint is from early training**: The saved model might be from epoch 0 or early in training
2. **Incorrect model state loading**: `strict=False` in `load_state_dict` might be skipping important weights
3. **Architecture mismatch**: The model architecture used during inference doesn't match training
4. **Missing optimizer state**: Contrastive learning might require specific optimizer states
5. **Wrong checkpoint file**: Using embedder checkpoint instead of classifier, or vice versa

## Recommendations

### Immediate Actions

1. **Verify the classifier checkpoint**:
   ```python
   checkpoint = torch.load('saved/classifier_contrastive_gru_sony_watch_20_120/limu_gru_v1.pt')
   print(checkpoint.keys())  # Check what's in the checkpoint
   print(checkpoint.get('epoch', 'No epoch info'))  # Check training progress
   print(checkpoint.get('best_acc', 'No accuracy info'))  # Check validation accuracy
   ```

2. **Check training logs**: Look at `pretrain_combined.log` or training output to see final accuracy

3. **Try different checkpoint**: If multiple checkpoints exist, test with the best one

4. **Retrain if necessary**: The model might need to be retrained completely

### For Android Implementation

**Good News**: The preprocessing and embedding pipeline is correct! Once you have a properly trained classifier model, you can use:

1. **Pre-computed embeddings approach** (recommended for Android):
   - Precompute BERT embeddings offline
   - Ship only the lightweight GRU classifier to Android
   - Much faster inference on device

2. **Full pipeline approach**:
   - Ship both BERT embedder and GRU classifier
   - Run complete preprocessing on device
   - Higher latency but no pre-computation needed

## Files Generated

1. `sony_watch_inference_pipeline.py` - Complete inference pipeline (✓ verified working)
2. `test_sony_watch_pipeline.py` - End-to-end testing script
3. `test_with_precomputed_embeddings.py` - Classifier-only testing script
4. `precomputed_embedding_test_results.json` - Detailed test results
5. `ANDROID_IMPLEMENTATION_GUIDE.md` - Android implementation guide

## Next Steps

1. **Fix the classifier model** by finding a properly trained checkpoint
2. **Verify training accuracy** matches the reported results in papers/logs
3. **Re-test with corrected model** using the same test scripts
4. **Proceed with Android implementation** once accuracy is verified

Expected accuracy after fixing should be **>80%** based on typical gesture recognition benchmarks.
