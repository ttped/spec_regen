# ML Section Classifier

This system uses XGBoost to classify document sections as valid or false positives, replacing the manual rule-based approach with a learned model.

## Overview

The ML classifier integrates into your existing pipeline:

```
Raw OCR → classify_agent → section_processor → [ML FILTER] → docx_writer
                                                    ↑
                                              Removes false
                                              positive sections
```

## Quick Start

### 1. Train a Model

```bash
# First, make sure you have labeled data in your training_features.csv
# Labels: 1 = valid section, 0 = false positive

python section_classifier.py train \
    -f results_simple/training_features.csv \
    -o section_model.joblib
```

### 2. Evaluate Performance

```bash
# See train/test split metrics and cross-validation
python section_classifier.py evaluate \
    -f results_simple/training_features.csv
```

### 3. Filter Documents

```bash
# Filter a single document
python section_classifier.py filter \
    -i results_simple/DOC_repaired.json \
    -o results_simple/DOC_ml_filtered.json \
    -m section_model.joblib

# Or batch process all documents
python pipeline_ml_integration.py \
    --results-dir results_simple \
    --model section_model.joblib \
    --threshold 0.5
```

### 4. Generate DOCX

```bash
# The filtered JSON can be passed directly to docx_writer
python docx_writer.py results_simple/DOC_ml_filtered.json output.docx ...
```

## Files

| File | Purpose |
|------|---------|
| `section_classifier.py` | Core ML training and prediction |
| `pipeline_ml_integration.py` | Batch processing and pipeline integration |
| `feature_extractor.py` | Extract features from documents (updated with new features) |

## Training Data Requirements

The training CSV (`training_features.csv`) needs:

1. **Labeled samples**: `_label` column with `1` (valid) or `0` (false positive)
2. **Enough samples**: At least 20 labeled, ideally 100+
3. **Balance**: Try to have both positive and negative examples

### Labeling Strategy

To efficiently label data:

```python
from feature_extractor import load_training_data

df = load_training_data('training_features.csv')

# Sort by combined_confidence to label high-confidence sections first
df_sorted = df.sort_values('combined_confidence', ascending=False)

# High confidence (likely valid) - quick to verify
high_conf = df_sorted[df_sorted['combined_confidence'] > 0.5]

# Low confidence (likely false positives) - quick to verify
low_conf = df_sorted[df_sorted['combined_confidence'] < 0.2]
```

## Model Parameters

### Training Parameters

```python
train_classifier(
    df,
    test_size=0.2,      # 20% for testing
    random_state=42,    # Reproducibility
    n_estimators=100,   # Number of trees
    max_depth=6,        # Tree depth
    learning_rate=0.1,  # Learning rate
)
```

### Filtering Threshold

The `threshold` parameter controls the precision/recall tradeoff:

| Threshold | Effect |
|-----------|--------|
| 0.3 | Keep more sections (higher recall, lower precision) |
| 0.5 | Balanced (default) |
| 0.7 | Remove more aggressively (higher precision, lower recall) |

```bash
# More conservative - keep more sections
python section_classifier.py filter -m model.joblib --threshold 0.3 ...

# More aggressive - remove more false positives  
python section_classifier.py filter -m model.joblib --threshold 0.7 ...
```

## Feature Importance

After training, the model reports feature importance. Key features typically include:

| Feature | Description | Correlation |
|---------|-------------|-------------|
| `is_logical_next` | Follows previous section logically | Positive |
| `in_toc_exact` | Appears in Table of Contents | Positive |
| `parent_exists` | Parent section exists | Positive |
| `looks_like_date` | Section number + title look like a date | Negative |
| `subsection_looks_like_decimal` | Subsection ≥50 (like 4.88) | Negative |
| `had_text_before_number` | Text appears before section number | Negative |

## Integration with simple_pipeline.py

Add ML filtering as a new step between repair and assets:

```python
# Add to argparse
parser.add_argument("--ml-model", type=str, default=None,
    help="Path to trained ML model for section filtering")
parser.add_argument("--ml-threshold", type=float, default=0.5,
    help="Classification threshold (default: 0.5)")

# Add step 4.5 after repair
if args.ml_model and os.path.exists(args.ml_model):
    from section_classifier import run_ml_filtering_on_file
    
    ml_filtered_out = os.path.join(args.results_dir, f"{stem}_ml_filtered.json")
    run_ml_filtering_on_file(
        repaired_out,
        ml_filtered_out,
        args.ml_model,
        threshold=args.ml_threshold
    )
    # Use filtered output for subsequent steps
    repaired_out = ml_filtered_out
```

## Analyzing Results

### Feature Correlations

```bash
python section_classifier.py analyze -f training_features.csv
```

### Model Summary

```bash
python section_classifier.py summary -m section_model.joblib
```

### Compare Before/After

```python
from pipeline_ml_integration import compare_before_after

compare_before_after(
    'results_simple/DOC_repaired.json',
    'results_simple/DOC_ml_filtered.json'
)
```

## Troubleshooting

### "Not enough labeled samples"

You need at least 20 labeled rows. Check your CSV:

```python
import pandas as pd
df = pd.read_csv('training_features.csv')
print(f"Labeled: {df['_label'].notna().sum()}")
print(f"Valid (1): {(df['_label'] == 1).sum()}")
print(f"False Positive (0): {(df['_label'] == 0).sum()}")
```

### Model performs poorly

1. Check class balance - need both positives and negatives
2. Add more labeled data
3. Run feature correlation analysis to find useful features
4. Try adjusting threshold

### Missing features during prediction

The classifier handles missing features by defaulting to 0. This is normal when the feature extractor version differs from training.

## Requirements

```bash
pip install pandas numpy xgboost scikit-learn joblib
```

## Example Workflow

```bash
# 1. Extract features from all documents
python feature_extractor.py --results_dir results_simple

# 2. Open training_features.csv and label some rows
#    _label = 1 for valid, _label = 0 for false positive

# 3. Train model
python section_classifier.py train \
    -f results_simple/training_features.csv \
    -o section_model.joblib

# 4. Check performance
python section_classifier.py evaluate \
    -f results_simple/training_features.csv

# 5. Apply to all documents
python pipeline_ml_integration.py \
    --results-dir results_simple \
    --model section_model.joblib

# 6. Generate DOCX from filtered files
# (use _ml_filtered.json instead of _repaired.json)
```
