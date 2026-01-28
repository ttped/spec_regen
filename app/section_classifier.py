"""
section_classifier.py - ML-based section classification using XGBoost.

This module trains on features from the CSV (computed by feature_extractor.py)
and predicts on the same CSV, marking which sections to keep/remove.

The pipeline flow is:
1. feature_extractor.py creates training_features.csv with ALL sections
2. User labels some rows (_label = 1 valid, 0 false positive)
3. This classifier trains on labeled rows, predicts on ALL rows
4. simple_pipeline uses predictions to filter sections
"""

import os
import warnings
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

warnings.filterwarnings('ignore', category=UserWarning)

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[Error] pandas and numpy required: pip install pandas numpy")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[Warning] XGBoost not installed: pip install xgboost")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        confusion_matrix, 
        precision_recall_fscore_support, 
        roc_auc_score
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[Warning] scikit-learn not installed: pip install scikit-learn")


# =============================================================================
# CURATED FEATURE LIST - Only these features are used for training
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


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_training_data(csv_path: str, verbose: bool = True) -> pd.DataFrame:
    """Load training data from CSV."""
    df = pd.read_csv(csv_path)
    if verbose:
        print(f"Loaded {len(df)} rows from {csv_path}")
    return df


def get_labeled_samples(df: pd.DataFrame, min_label: int = 0) -> pd.DataFrame:
    """Filter to labeled samples only (label >= min_label)."""
    labeled = df[df['_label'].notna()].copy()
    labeled['_label'] = pd.to_numeric(labeled['_label'], errors='coerce')
    labeled = labeled[labeled['_label'] >= min_label]
    return labeled


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare feature matrix using ONLY the curated feature list.
    """
    # Use only selected features that exist in the dataframe
    available_features = [f for f in SELECTED_FEATURES if f in df.columns]
    
    missing = set(SELECTED_FEATURES) - set(available_features)
    if missing:
        print(f"  [Note] Missing features (will use 0): {missing}")
    
    X = df[available_features].copy()
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    return X, available_features


# =============================================================================
# MODEL TRAINING
# =============================================================================

@dataclass
class TrainingResult:
    """Results from model training."""
    model: Any
    feature_columns: List[str]
    train_accuracy: float
    test_accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: Optional[float]
    feature_importance: Dict[str, float]
    confusion_matrix: np.ndarray


def train_classifier(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True
) -> TrainingResult:
    """
    Train XGBoost classifier on labeled data.
    """
    if not HAS_XGBOOST or not HAS_SKLEARN:
        raise ImportError("XGBoost and scikit-learn are required")
    
    # Get labeled samples
    labeled_df = get_labeled_samples(df, min_label=0)
    
    pos_count = (labeled_df['_label'] == 1).sum()
    neg_count = (labeled_df['_label'] == 0).sum()
    
    if verbose:
        print(f"\nTraining data:")
        print(f"  Labeled samples: {len(labeled_df)}")
        print(f"  Valid sections (1): {pos_count}")
        print(f"  False positives (0): {neg_count}")
    
    if len(labeled_df) < 20:
        raise ValueError(f"Need at least 20 labeled samples, have {len(labeled_df)}")
    
    # Prepare features - uses ONLY curated list
    X, feature_cols = prepare_features(labeled_df)
    y = labeled_df['_label'].astype(int)
    
    if verbose:
        print(f"  Features used: {len(feature_cols)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    if verbose:
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # XGBoost parameters
    params = {
        'n_estimators': 100,
        'max_depth': 4,  # Reduced to prevent overfitting
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': random_state,
        'eval_metric': 'logloss',
    }
    
    # Handle class imbalance
    if pos_count > 0 and neg_count > 0:
        params['scale_pos_weight'] = neg_count / pos_count
    
    # Train
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]
    
    train_accuracy = (train_pred == y_train).mean()
    test_accuracy = (test_pred == y_test).mean()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, test_pred, average='binary', pos_label=1
    )
    
    try:
        auc_roc = roc_auc_score(y_test, test_proba)
    except:
        auc_roc = None
    
    conf_matrix = confusion_matrix(y_test, test_pred)
    
    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    if verbose:
        print(f"\n{'='*50}")
        print("MODEL PERFORMANCE")
        print(f"{'='*50}")
        print(f"Train Accuracy: {train_accuracy:.3f}")
        print(f"Test Accuracy:  {test_accuracy:.3f}")
        print(f"Precision:      {precision:.3f}")
        print(f"Recall:         {recall:.3f}")
        print(f"F1 Score:       {f1:.3f}")
        if auc_roc:
            print(f"AUC-ROC:        {auc_roc:.3f}")
        
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              FP    Valid")
        print(f"  Actual FP  {conf_matrix[0,0]:4d}  {conf_matrix[0,1]:4d}")
        print(f"  Actual Val {conf_matrix[1,0]:4d}  {conf_matrix[1,1]:4d}")
        
        print(f"\nTop Features:")
        for i, (feat, imp) in enumerate(list(importance.items())[:10]):
            print(f"  {i+1:2d}. {feat:<30s} {imp:.4f}")
    
    return TrainingResult(
        model=model,
        feature_columns=feature_cols,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc_roc=auc_roc,
        feature_importance=importance,
        confusion_matrix=conf_matrix
    )


# =============================================================================
# PREDICTION ON FULL DATASET
# =============================================================================

def predict_all(
    df: pd.DataFrame, 
    model, 
    feature_columns: List[str],
    threshold: float = 0.5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run predictions on ALL rows in the dataframe (not just labeled ones).
    
    Adds columns:
        _predicted_prob: probability of being a valid section
        _predicted_label: 1 if prob >= threshold, else 0
        _keep: True if should keep this section
    
    Returns the dataframe with predictions added.
    """
    df = df.copy()
    
    # Prepare features for ALL rows
    X, _ = prepare_features(df)
    
    # Ensure columns match what model expects
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]
    
    # Predict
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    df['_predicted_prob'] = probabilities
    df['_predicted_label'] = predictions
    df['_keep'] = predictions == 1
    
    if verbose:
        print(f"\nPredictions on {len(df)} rows:")
        print(f"  Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
        print(f"  Mean probability: {probabilities.mean():.3f}")
        print(f"  Threshold: {threshold}")
        print(f"  Sections to keep: {predictions.sum()}")
        print(f"  Sections to remove: {(predictions == 0).sum()}")
    
    return df


def get_sections_to_remove(
    df: pd.DataFrame,
    doc_name: str = None
) -> List[Tuple[str, int, str]]:
    """
    Get list of sections that should be removed for a specific document.
    
    Returns list of (section_number, page, title) tuples.
    """
    if doc_name:
        doc_df = df[df['_doc_name'] == doc_name]
    else:
        doc_df = df
    
    remove_df = doc_df[doc_df['_keep'] == False]
    
    result = []
    for _, row in remove_df.iterrows():
        result.append((
            row.get('_section_number_raw', ''),
            row.get('_page', 0),
            row.get('_title', '')[:50]
        ))
    
    return result


def get_sections_to_keep(
    df: pd.DataFrame,
    doc_name: str = None
) -> List[Tuple[str, int]]:
    """
    Get list of sections that should be kept for a specific document.
    
    Returns list of (section_number, page) tuples.
    """
    if doc_name:
        doc_df = df[df['_doc_name'] == doc_name]
    else:
        doc_df = df
    
    keep_df = doc_df[doc_df['_keep'] == True]
    
    result = []
    for _, row in keep_df.iterrows():
        result.append((
            row.get('_section_number_raw', ''),
            row.get('_page', 0)
        ))
    
    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and evaluate section classifier")
    parser.add_argument('-f', '--features', required=True, help='Path to training features CSV')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    
    args = parser.parse_args()
    
    print("Loading data...")
    df = load_training_data(args.features)
    
    print(f"\nTraining classifier with {len(SELECTED_FEATURES)} curated features...")
    result = train_classifier(df, test_size=args.test_size)
    
    print("\nRunning predictions on all data...")
    df_with_preds = predict_all(df, result.model, result.feature_columns, threshold=args.threshold)
    
    # Show some examples of what would be removed
    remove_list = get_sections_to_remove(df_with_preds)
    if remove_list:
        print(f"\nSections to remove ({len(remove_list)} total):")
        for sec_num, page, title in remove_list[:10]:
            print(f"  {sec_num:12s} p{page:3d}  {title}")
        if len(remove_list) > 10:
            print(f"  ... and {len(remove_list) - 10} more")
    
    print("\nDone!")