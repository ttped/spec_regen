"""
section_classifier.py - ML-based section classification using XGBoost.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass


# =============================================================================
# CURATED FEATURE LIST
# =============================================================================

SELECTED_FEATURES = [
    'combined_confidence',
    'parent_exists',
    'extended_sequence_score',
    'toc_fuzzy_score',
    'section_depth',
    'sequence_fit_score',
    'x_near_left_margin',
    'title_starts_capital',
    'x_matches_depth_indent',
    'bbox_width_pct',
    'title_quality_score',
    'bbox_right_pct',
    'format_consistency_score',
    'content_length_chars',
    'has_large_vertical_gap',
    'content_is_empty',
    'bbox_bottom_pct',
    'bbox_top_pct',
    'is_in_footer_zone',
    'has_small_vertical_gap',
    'title_is_empty',
    'section_num_ratio_suspicious',
    'bbox_left_pct',
    'looks_like_date',
    'is_simple_number',
    'relative_height',
]


@dataclass
class TrainingResult:
    model: XGBClassifier
    feature_columns: List[str]
    precision: float
    recall: float
    f1: float


def train_and_predict(
    csv_path: str, 
    threshold: float = 0.5
) -> Dict[str, Set[Tuple[str, int]]]:
    """
    Train on labeled data, predict on ALL data.
    
    Returns dict: {doc_name: set of (section_number, page) tuples to KEEP}
    """
    print(f"\n  Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Total rows: {len(df)}")
    
    # Check for required columns
    required = ['_label', '_doc_name', '_section_number_raw', '_page']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    
    # Get labeled data
    labeled_df = df[df['_label'].notna()].copy()
    labeled_df['_label'] = pd.to_numeric(labeled_df['_label'], errors='coerce')
    labeled_df = labeled_df[labeled_df['_label'] >= 0]
    
    pos_count = (labeled_df['_label'] == 1).sum()
    neg_count = (labeled_df['_label'] == 0).sum()
    print(f"  Labeled: {len(labeled_df)} ({pos_count} valid, {neg_count} false positive)")
    
    if len(labeled_df) < 20:
        raise ValueError(f"Need >= 20 labeled rows, have {len(labeled_df)}")
    
    # Prepare features
    available = [f for f in SELECTED_FEATURES if f in df.columns]
    missing_feat = set(SELECTED_FEATURES) - set(available)
    print(f"  Features: {len(available)}/{len(SELECTED_FEATURES)}")
    if missing_feat:
        print(f"  Missing: {missing_feat}")
    
    # Train
    X_labeled = labeled_df[available].fillna(0).replace([np.inf, -np.inf], 0)
    y_labeled = labeled_df['_label'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_labeled, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled
    )

    scale_weight = (neg_count / pos_count) * 1.5 if pos_count > 0 else 1
    
    model = XGBClassifier(
        n_estimators=150, 
        max_depth=3,                 # Reduced from 4 -> 3 to prevent overfitting
        learning_rate=0.05,          # Slower learning for better generalization
        min_child_weight=3,          # Conservative: needs 3+ samples to make a rule
        gamma=0.1,                   # Pruning parameter
        subsample=0.8, 
        colsample_bytree=0.8, 
        random_state=42,
        reg_alpha=0.1,               # L1 Regularization (noise removal)
        reg_lambda=1.0,              # L2 Regularization
        scale_pos_weight=scale_weight
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    test_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, test_pred, average='binary', pos_label=1
    )
    conf = confusion_matrix(y_test, test_pred)
    
    print(f"\n  === MODEL RESULTS ===")
    print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    print(f"  Confusion: TN={conf[0,0]}, FP={conf[0,1]}, FN={conf[1,0]}, TP={conf[1,1]}")
    
    # Predict on ALL
    X_all = df[available].fillna(0).replace([np.inf, -np.inf], 0)
    probs = model.predict_proba(X_all)[:, 1]
    
    print(f"\n  Predictions on {len(df)} rows:")
    print(f"  Prob range: [{probs.min():.3f}, {probs.max():.3f}], mean: {probs.mean():.3f}")
    
    df['_keep'] = probs >= threshold
    print(f"  Keep: {df['_keep'].sum()}, Remove: {(~df['_keep']).sum()}")
    
    # Build result
    result = {}
    for doc_name in df['_doc_name'].unique():
        doc_df = df[(df['_doc_name'] == doc_name) & (df['_keep'])]
        sections = set()
        for _, row in doc_df.iterrows():
            sec_num = row['_section_number_raw']
            page = int(row['_page']) if pd.notna(row['_page']) else 0
            sections.add((sec_num, page))
        result[doc_name] = sections
    
    print(f"  Documents: {len(result)}")
    return result


if __name__ == '__main__':
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results_simple/training_features.csv"
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    train_and_predict(csv_path, threshold)