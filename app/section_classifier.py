"""
section_classifier.py - ML-based section classification using XGBoost.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from typing import List, Dict, Tuple, Set, Any
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

def train_and_predict(
    csv_path: str, 
    threshold: float = 0.5
) -> Tuple[Dict[str, Set[Tuple[str, int]]], Dict[str, Any]]:
    """
    Train on labeled data, predict on ALL data.
    
    Performs two passes:
    1. Validation Pass (80/20 split): To measure generalization performance.
    2. Production Pass (100% labeled): Retrains on ALL labeled data for final predictions.
    
    Returns:
        tuple: (result_dict, metrics_dict)
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
    labeled_df = labeled_df[labeled_df['_label'].isin([0, 1])]
    
    pos_count = (labeled_df['_label'] == 1).sum()
    neg_count = (labeled_df['_label'] == 0).sum()
    print(f"  Labeled: {len(labeled_df)} ({pos_count} valid, {neg_count} false positive)")
    
    if len(labeled_df) < 20:
        raise ValueError(f"Need >= 20 labeled rows, have {len(labeled_df)}")
    
    # Prepare features
    available = [f for f in SELECTED_FEATURES if f in df.columns]
    missing_feat = set(SELECTED_FEATURES) - set(available)
    print(f"  Features: {len(available)}/{len(SELECTED_FEATURES)}")
    
    X_labeled = labeled_df[available].fillna(0).replace([np.inf, -np.inf], 0)
    y_labeled = labeled_df['_label'].astype(int)
    
    scale_weight = (neg_count / pos_count) * 1.5 if pos_count > 0 else 1

    # Hyperparameters
    params = {
        'n_estimators': 150,
        'max_depth': 3,
        'learning_rate': 0.05,
        'min_child_weight': 3,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': scale_weight
    }

    metrics = {}

    # =========================================================================
    # PASS 1: VALIDATION (80/20 Split) - How well does it generalize?
    # =========================================================================
    print(f"\n  --- Phase 1: Validation (80/20 Split) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X_labeled, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled
    )

    val_model = XGBClassifier(**params)
    val_model.fit(X_train, y_train)
    
    val_pred = val_model.predict(X_test)
    v_prec, v_rec, v_f1, _ = precision_recall_fscore_support(y_test, val_pred, average='binary', pos_label=1)
    v_acc = accuracy_score(y_test, val_pred)
    v_conf = confusion_matrix(y_test, val_pred)
    
    metrics['val'] = {
        'precision': v_prec, 'recall': v_rec, 'f1': v_f1, 'accuracy': v_acc, 'confusion': v_conf
    }
    
    print(f"  Precision: {v_prec:.3f}, Recall: {v_rec:.3f}, F1: {v_f1:.3f}")
    print(f"  Confusion (Test Set): TN={v_conf[0,0]}, FP={v_conf[0,1]}, FN={v_conf[1,0]}, TP={v_conf[1,1]}")

    # =========================================================================
    # PASS 2: PRODUCTION (100% Data) - Maximize knowledge
    # =========================================================================
    print(f"\n  --- Phase 2: Full Training (100% Labeled Data) ---")
    final_model = XGBClassifier(**params)
    final_model.fit(X_labeled, y_labeled)
    
    # Check training error (sanity check)
    train_pred = final_model.predict(X_labeled)
    t_prec, t_rec, t_f1, _ = precision_recall_fscore_support(y_labeled, train_pred, average='binary', pos_label=1)
    t_acc = accuracy_score(y_labeled, train_pred)
    t_conf = confusion_matrix(y_labeled, train_pred)

    metrics['train'] = {
        'precision': t_prec, 'recall': t_rec, 'f1': t_f1, 'accuracy': t_acc, 'confusion': t_conf,
        'total_samples': len(labeled_df)
    }

    print(f"  Training Fit: Precision: {t_prec:.3f}, Recall: {t_rec:.3f}, F1: {t_f1:.3f}")
    
    # =========================================================================
    # PREDICT ON EVERYTHING
    # =========================================================================
    X_all = df[available].fillna(0).replace([np.inf, -np.inf], 0)
    probs = final_model.predict_proba(X_all)[:, 1]
    
    print(f"\n  Predictions on {len(df)} rows:")
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
    
    return result, metrics

if __name__ == '__main__':
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results_simple/training_features.csv"
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    train_and_predict(csv_path, threshold)