# Gesture Validation System - Quick Reference

## What This System Does

**gesture_validator.py** + **evaluate_gesture_quality_real.py** = Complete gesture validation system

---

## Quick Start

### 1. Validate a NEW Gesture (Before Adding to Dataset)
```bash
python gesture_validator.py \
    --candidate_data my_gesture_samples.npy \
    --dataset Alexandra \
    --version 20_120
```

**Output**: 
- ✅ `ACCEPT` - Gesture is unique and recognizable → Safe to add
- ❌ `REJECT` - Too similar or hard to recognize → Redesign needed

### 2. Check Quality of EXISTING Gestures
```bash
python evaluate_gesture_quality_real.py --dataset Alexandra --version 20_120
python evaluate_gesture_quality_real.py --dataset Turiya --version 20_120
python evaluate_gesture_quality_real.py --dataset Julius --version 20_120
```

**Output**: 
- Per-gesture quality metrics (Precision, Recall, F1)
- Identifies problematic gestures
- Saves JSON + CSV reports

---

## Current Results (10 Epochs)

| User | Accuracy | Good | Bad |
|------|----------|------|-----|
| Alexandra | 46.7% | 6/15 | **9/15** |
| Turiya | 66.7% | 7/15 | 8/15 |
| Julius | 63.3% | 8/15 | 7/15 |

### Problematic on ALL Users ⚠️
- **Gesture 2** (0% recall)
- **Gesture 10** (low recall)
- **Gesture 15** (confused with 14)

### Good on ALL Users ✅
- **Gesture 5** (always recognized)
- **Gesture 14** (consistent performance)

---

## System Details

### Gesture Validator Checks
1. **Uniqueness**: Is gesture distinct from existing? (< 75% similarity)
2. **Recognizability**: Can model recognize it? (> 70% confidence)

### Quality Evaluator Metrics
- **Precision**: Of predicted samples, how many correct?
- **Recall**: Of actual samples, how many detected?
- **F1-Score**: Harmonic mean of precision and recall

### Training Setup
- **Split**: 70% train / 10% val / 20% test
- **Semi-supervised**: 50% labeled, 50% unlabeled data
- **Model**: ContrastiveTransformerClassifier
- **Epochs**: 10 (quick) or 30 (thorough)

---

## Results Location

```
gesture_quality/
├── Alexandra_quality_report_real.json   # Detailed results
├── Alexandra_quality_report_real.csv    # Quick view
├── Turiya_quality_report_real.json
├── Turiya_quality_report_real.csv
├── Julius_quality_report_real.json
└── Julius_quality_report_real.csv
```

---

## Decision Guide

### For NEW Gestures

**ACCEPT ✅** when:
- Similarity to most similar gesture: < 75%
- Model confidence: > 70%

**REJECT ❌** when:
- Similarity: > 75% (too similar to existing)
- Confidence: < 70% (hard to recognize)

### For EXISTING Gestures

**GOOD ✓** when:
- Recall: > 70%
- No major confusion (< 30% with one gesture)

**PROBLEMATIC ❌** when:
- Recall: < 70% (misses many samples)
- High confusion: > 30% with one gesture

---

## Next Steps

1. **Immediate**: Use gesture_validator.py for new submissions
2. **Short-term**: Address Gestures 2, 10, 15
3. **Medium-term**: Redesign problematic gestures or collect better data
4. **Long-term**: Retrain with cleaned dataset (expect 55-60% accuracy)

---

## File Locations

```
/home/prerna/LIMU-BERT-blind-users/
├── gesture_validator.py                    # Validates NEW gestures
├── evaluate_gesture_quality_real.py        # Evaluates EXISTING gestures
├── COMPLETE_VALIDATION_REPORT.md          # Full details
├── GESTURE_VALIDATION_SUMMARY.md          # System overview
├── PARAMETER_VERIFICATION.md              # Parameter matching
├── QUICK_REFERENCE.md                     # This file
└── gesture_quality/                        # Results directory
```

---

## Troubleshooting

### gesture_validator.py errors
- **Missing embedding file**: Check embed/ folder has candidate embeddings
- **File not found**: Verify paths are correct

### evaluate_gesture_quality_real.py errors
- **Out of memory**: Reduce batch size or use CPU (slower)
- **Module not found**: Install required packages (torch, sklearn, etc)

---

## Key Metrics Explained

**Precision** = TP/(TP+FP) = "Of predictions, how many correct?"
**Recall** = TP/(TP+FN) = "Of actual samples, how many detected?"
**F1** = 2(P*R)/(P+R) = "Overall performance"

**Example**:
- Gesture appears 10 times in test set
- Model detects 7 and correctly classifies 6
- Recall = 6/10 = 60% (misses 40%)
- Precision = 6/7 = 86% (falsely predicted 1)
- F1 = 2(0.86*0.60)/(0.86+0.60) = 0.71

---

## Performance Tips

- Use 10 epochs for quick testing (~5 min per dataset)
- Use 30 epochs for thorough evaluation (~15 min per dataset)
- GPU required for reasonable speed (CUDA available)
- Batch size 16 works for most systems

---

## Contact & Support

**Status**: ✅ System fully operational  
**Last Updated**: April 6, 2026  
**Verified**: All parameters match production classifier
