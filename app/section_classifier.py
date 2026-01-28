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
    print(f"  Columns: {list(df.columns)[:10]}... ({len(df.columns)} total)")
    
    # Check for required columns
    if '_label' not in df.columns:
        raise ValueError("CSV missing '_label' column")
    if '_doc_name' not in df.columns:
        raise ValueError("CSV missing '_doc_name' column")
    if '_section_number_raw' not in df.columns:
        raise ValueError("CSV missing '_section_number_raw' column")
    
    # Get labeled data
    labeled_df = df[df['_label'].notna()].copy()
    labeled_df['_label'] = pd.to_numeric(labeled_df['_label'], errors='coerce')
    labeled_df = labeled_df[labeled_df['_label'] >= 0]  # Filter out negative labels
    
    print(f"\n  Labeled rows: {len(labeled_df)}")
    pos_count = (labeled_df['_label'] == 1).sum()
    neg_count = (labeled_df['_label'] == 0).sum()
    print(f"  Valid (1): {pos_count}")
    print(f"  False positive (0): {neg_count}")
    
    if len(labeled_df) < 20:
        raise ValueError(f"Need at least 20 labeled rows, have {len(labeled_df)}")
    
    # Prepare features
    available_features = [f for f in SELECTED_FEATURES if f in df.columns]
    missing_features = set(SELECTED_FEATURES) - set(available_features)
    
    print(f"\n  Features requested: {len(SELECTED_FEATURES)}")
    print(f"  Features available: {len(available_features)}")
    if missing_features:
        print(f"  Missing features: {missing_features}")
    
    if len(available_features) < 5:
        raise ValueError(f"Not enough features available: {available_features}")
    
    # Training data
    X_train_full = labeled_df[available_features].fillna(0).replace([np.inf, -np.inf], 0)
    y_train_full = labeled_df['_label'].astype(int)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    print(f"\n  Train set: {len(X_train)}")
    print(f"  Test set: {len(X_test)}")
    
    # Train model
    print(f"\n  Training XGBoost...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=neg_count / pos_count if pos_count > 0 else 1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    test_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, test_pred, average='binary', pos_label=1
    )
    conf = confusion_matrix(y_test, test_pred)
    
    print(f"\n  === MODEL PERFORMANCE ===")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  Confusion matrix:")
    print(f"              Predicted")
    print(f"              0     1")
    print(f"  Actual 0   {conf[0,0]:4d}  {conf[0,1]:4d}")
    print(f"  Actual 1   {conf[1,0]:4d}  {conf[1,1]:4d}")
    
    # Feature importance
    importance = sorted(zip(available_features, model.feature_importances_), 
                       key=lambda x: x[1], reverse=True)
    print(f"\n  Top 5 features:")
    for feat, imp in importance[:5]:
        print(f"    {feat}: {imp:.4f}")
    
    # Predict on ALL data
    print(f"\n  === PREDICTING ON ALL {len(df)} ROWS ===")
    X_all = df[available_features].fillna(0).replace([np.inf, -np.inf], 0)
    probabilities = model.predict_proba(X_all)[:, 1]
    
    print(f"  Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
    print(f"  Mean probability: {probabilities.mean():.3f}")
    print(f"  Threshold: {threshold}")
    
    df['_prob'] = probabilities
    df['_keep'] = probabilities >= threshold
    
    keep_count = df['_keep'].sum()
    remove_count = (~df['_keep']).sum()
    print(f"  Sections to KEEP: {keep_count}")
    print(f"  Sections to REMOVE: {remove_count}")
    
    # Build result dict
    print(f"\n  === BUILDING PREDICTIONS BY DOCUMENT ===")
    sections_to_keep = {}
    
    for doc_name in df['_doc_name'].unique():
        doc_df = df[df['_doc_name'] == doc_name]
        keep_df = doc_df[doc_df['_keep'] == True]
        
        sections = set()
        for _, row in keep_df.iterrows():
            sec_num = row['_section_number_raw']
            page = int(row['_page']) if pd.notna(row['_page']) else 0
            sections.add((sec_num, page))
        
        sections_to_keep[doc_name] = sections
    
    print(f"  Documents in predictions: {len(sections_to_keep)}")
    for doc_name, secs in list(sections_to_keep.items())[:3]:
        print(f"    {doc_name}: {len(secs)} sections to keep")
    if len(sections_to_keep) > 3:
        print(f"    ... and {len(sections_to_keep) - 3} more documents")
    
    return sections_to_keep


if __name__ == '__main__':
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results_simple/training_features.csv"
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    result = train_and_predict(csv_path, threshold)
    print(f"\nDone! Predictions for {len(result)} documents.")