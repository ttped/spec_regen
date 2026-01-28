"""
section_classifier.py - ML-based section classification using XGBoost.

This module:
1. Trains a classifier on labeled feature data
2. Evaluates with train/test split
3. Filters sections before passing to docx_writer
4. Validates against TOC

Usage:
    # Train a new model
    python section_classifier.py train --features training_features.csv --output model.joblib
    
    # Evaluate model performance  
    python section_classifier.py evaluate --features training_features.csv --model model.joblib
    
    # Filter a document's sections using trained model
    python section_classifier.py filter --input organized.json --output filtered.json --model model.joblib
    
    # Run full pipeline with ML filtering
    python section_classifier.py pipeline --results-dir results_simple --model model.joblib

Integration with simple_pipeline.py:
    The filter_sections_with_model() function can be called after section_processor
    and before docx_writer to remove false positives.
"""

import os
import json
import pickle
import warnings
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
import re

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Check for required libraries
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[Error] pandas and numpy are required. Install with: pip install pandas numpy")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[Warning] XGBoost not installed. Install with: pip install xgboost")

try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        classification_report, confusion_matrix, 
        precision_recall_fscore_support, roc_auc_score
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[Warning] scikit-learn not installed. Install with: pip install scikit-learn")

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    print("[Warning] joblib not installed. Install with: pip install joblib")


# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

# Features to exclude from training (metadata columns start with _)
METADATA_COLUMNS = ['_doc_name', '_index', '_section_number_raw', '_title', 
                    '_title_full_length', '_page', '_label', '_section_number_normalized']

# Features that might cause issues (remove if they don't help)
POTENTIALLY_PROBLEMATIC_FEATURES = [
    'is_sandwiched',           # Negative correlation
    'sandwich_same_neighbors', # Negative correlation
]

# High-value features based on domain knowledge
HIGH_VALUE_FEATURES = [
    'is_logical_next',
    'in_toc_exact',
    'parent_exists',
    'title_has_section_keyword',
    'looks_like_date',
    'had_text_before_number',
    'subsection_looks_like_decimal',
]


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_training_data(csv_path: str, verbose: bool = True) -> pd.DataFrame:
    """Load and prepare training data from CSV."""
    if not HAS_PANDAS:
        raise ImportError("pandas is required")
    
    df = pd.read_csv(csv_path)
    
    if verbose:
        print(f"Loaded {len(df)} rows from {csv_path}")
        print(f"  Columns: {len(df.columns)}")
    
    return df


def prepare_features(df: pd.DataFrame, exclude_problematic: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare feature matrix for training.
    
    Args:
        df: DataFrame with features and labels
        exclude_problematic: Whether to exclude features with known negative correlation
        
    Returns:
        Tuple of (feature DataFrame, list of feature column names)
    """
    # Get feature columns (exclude metadata)
    feature_cols = [c for c in df.columns if not c.startswith('_')]
    
    # Optionally exclude problematic features
    if exclude_problematic:
        feature_cols = [c for c in feature_cols if c not in POTENTIALLY_PROBLEMATIC_FEATURES]
    
    # Get feature matrix
    X = df[feature_cols].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    return X, feature_cols


def get_labeled_samples(df: pd.DataFrame, min_label: int = 0) -> pd.DataFrame:
    """
    Filter to labeled samples only.
    
    Args:
        df: Full DataFrame
        min_label: Minimum label value to include (filters out special negative labels)
        
    Returns:
        DataFrame with only valid labeled rows
    """
    # Filter to rows with labels
    labeled = df[df['_label'].notna()].copy()
    
    # Convert to numeric
    labeled['_label'] = pd.to_numeric(labeled['_label'], errors='coerce')
    
    # Filter out special labels (less than min_label)
    labeled = labeled[labeled['_label'] >= min_label]
    
    return labeled


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
    classification_report: str


def train_classifier(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
    **xgb_params
) -> TrainingResult:
    """
    Train XGBoost classifier on labeled data.
    
    Args:
        df: DataFrame with features and _label column
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        verbose: Print progress
        **xgb_params: Additional parameters for XGBClassifier
        
    Returns:
        TrainingResult with model and metrics
    """
    if not HAS_XGBOOST or not HAS_SKLEARN:
        raise ImportError("XGBoost and scikit-learn are required for training")
    
    # Get labeled samples
    labeled_df = get_labeled_samples(df, min_label=0)
    
    if verbose:
        print(f"\nTraining data:")
        print(f"  Total labeled samples: {len(labeled_df)}")
        print(f"  Positive (valid sections): {(labeled_df['_label'] == 1).sum()}")
        print(f"  Negative (false positives): {(labeled_df['_label'] == 0).sum()}")
    
    if len(labeled_df) < 20:
        raise ValueError(f"Need at least 20 labeled samples, have {len(labeled_df)}")
    
    # Prepare features
    X, feature_cols = prepare_features(labeled_df)
    y = labeled_df['_label'].astype(int)
    
    if verbose:
        print(f"  Features used: {len(feature_cols)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    if verbose:
        print(f"\nSplit:")
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")
    
    # Default XGBoost parameters (tuned for this problem)
    default_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': random_state,
        'eval_metric': 'logloss',
        'use_label_encoder': False,
    }
    default_params.update(xgb_params)
    
    # Handle class imbalance
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    if pos_count > 0 and neg_count > 0:
        default_params['scale_pos_weight'] = neg_count / pos_count
    
    # Train model
    if verbose:
        print("\nTraining XGBoost classifier...")
    
    model = XGBClassifier(**default_params)
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
    class_report = classification_report(y_test, test_pred, target_names=['False Positive', 'Valid Section'])
    
    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    if verbose:
        print(f"\n{'='*60}")
        print("TRAINING RESULTS")
        print(f"{'='*60}")
        print(f"Train Accuracy: {train_accuracy:.3f}")
        print(f"Test Accuracy:  {test_accuracy:.3f}")
        print(f"Precision:      {precision:.3f}")
        print(f"Recall:         {recall:.3f}")
        print(f"F1 Score:       {f1:.3f}")
        if auc_roc:
            print(f"AUC-ROC:        {auc_roc:.3f}")
        
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 FP    Valid")
        print(f"  Actual FP     {conf_matrix[0,0]:4d}  {conf_matrix[0,1]:4d}")
        print(f"  Actual Valid  {conf_matrix[1,0]:4d}  {conf_matrix[1,1]:4d}")
        
        print(f"\nTop 10 Most Important Features:")
        for i, (feat, imp) in enumerate(list(importance.items())[:10]):
            print(f"  {i+1:2d}. {feat:<40s} {imp:.4f}")
    
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
        confusion_matrix=conf_matrix,
        classification_report=class_report
    )


def cross_validate_model(
    df: pd.DataFrame,
    n_folds: int = 5,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Perform cross-validation to get more robust performance estimates.
    """
    if not HAS_XGBOOST or not HAS_SKLEARN:
        raise ImportError("XGBoost and scikit-learn are required")
    
    labeled_df = get_labeled_samples(df, min_label=0)
    X, feature_cols = prepare_features(labeled_df)
    y = labeled_df['_label'].astype(int)
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
    )
    
    if verbose:
        print(f"\nRunning {n_folds}-fold cross-validation...")
    
    scores = cross_val_score(model, X, y, cv=n_folds, scoring='f1')
    
    results = {
        'mean_f1': scores.mean(),
        'std_f1': scores.std(),
        'min_f1': scores.min(),
        'max_f1': scores.max(),
    }
    
    if verbose:
        print(f"  F1 Score: {results['mean_f1']:.3f} (+/- {results['std_f1']:.3f})")
        print(f"  Range: [{results['min_f1']:.3f}, {results['max_f1']:.3f}]")
    
    return results


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_model(result: TrainingResult, path: str):
    """Save trained model and metadata."""
    if not HAS_JOBLIB:
        raise ImportError("joblib is required. Install with: pip install joblib")
    
    save_data = {
        'model': result.model,
        'feature_columns': result.feature_columns,
        'metrics': {
            'train_accuracy': result.train_accuracy,
            'test_accuracy': result.test_accuracy,
            'precision': result.precision,
            'recall': result.recall,
            'f1': result.f1,
            'auc_roc': result.auc_roc,
        },
        'feature_importance': result.feature_importance,
    }
    
    joblib.dump(save_data, path)
    print(f"Model saved to: {path}")


def load_model(path: str) -> Tuple[Any, List[str], Dict]:
    """Load trained model and metadata."""
    if not HAS_JOBLIB:
        raise ImportError("joblib is required. Install with: pip install joblib")
    
    data = joblib.load(path)
    return data['model'], data['feature_columns'], data.get('metrics', {})


# =============================================================================
# SECTION FILTERING (INTEGRATION WITH PIPELINE)
# =============================================================================

def extract_features_for_element(
    element: Dict,
    idx: int,
    all_elements: List[Dict],
    doc_stats: Dict = None
) -> Dict[str, Any]:
    """
    Extract features for a single element for prediction.
    
    This is a simplified version of the full feature extraction that can
    run on elements already in the pipeline (after section_processor).
    """
    features = {}
    
    section_num = element.get('section_number', '')
    title = element.get('topic', '') or ''
    bbox = element.get('bbox', {})
    detection_ctx = element.get('detection_context', {})
    
    # Parse section number
    parts = section_num.replace(',', '.').replace('-', '.').split('.')
    parts = [p.strip() for p in parts if p.strip()]
    depth = len(parts)
    
    # Get major section number
    try:
        major = int(parts[0]) if parts else -1
    except ValueError:
        major = -1
    
    # Section number features
    features['section_depth'] = depth
    features['section_major'] = major
    features['is_simple_number'] = 1 if depth == 1 else 0
    
    # Section number string analysis
    features['section_num_alpha_count'] = sum(1 for c in section_num if c.isalpha())
    features['section_num_is_pure_digits'] = 1 if section_num.replace(' ', '').replace('.', '').isdigit() else 0
    features['section_num_dot_count'] = section_num.count('.')
    features['section_num_has_hyphen'] = 1 if '-' in section_num else 0
    features['section_num_has_comma'] = 1 if ',' in section_num else 0
    features['section_num_raw_length'] = len(section_num.replace(' ', ''))
    
    # Check for decimal-looking subsections
    max_subsection = 0
    for p in parts[1:]:
        try:
            max_subsection = max(max_subsection, int(p))
        except ValueError:
            pass
    features['max_subsection_value'] = max_subsection
    features['subsection_looks_like_decimal'] = 1 if max_subsection >= 50 else 0
    
    # Detection context features
    features['had_leading_whitespace'] = 1 if detection_ctx.get('had_leading_whitespace') else 0
    features['had_text_before_number'] = 1 if detection_ctx.get('had_text_before_number') else 0
    features['text_before_number_len'] = len(detection_ctx.get('text_before_number', ''))
    features['original_line_length'] = detection_ctx.get('original_line_length', 0)
    
    # Bounding box features
    if bbox:
        features['bbox_left_px'] = bbox.get('left', 0)
        features['bbox_top_px'] = bbox.get('top', 0)
        features['bbox_width_px'] = bbox.get('width', 0)
        features['bbox_height_px'] = bbox.get('height', 0)
    else:
        features['bbox_left_px'] = 0
        features['bbox_top_px'] = 0
        features['bbox_width_px'] = 0
        features['bbox_height_px'] = 0
    
    # Title features
    features['title_length_chars'] = len(title)
    features['title_length_words'] = len(title.split()) if title else 0
    features['title_is_empty'] = 1 if not title.strip() else 0
    features['title_starts_capital'] = 1 if title and title[0].isupper() else 0
    features['title_is_all_caps'] = 1 if title and title.isupper() else 0
    
    # Date detection
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
              'january', 'february', 'march', 'april', 'june', 'july', 'august', 'september',
              'october', 'november', 'december']
    first_word = title.lower().split()[0] if title.split() else ''
    features['title_starts_with_month'] = 1 if first_word.rstrip('-.') in months else 0
    
    is_possible_day = False
    try:
        num_val = int(section_num.replace('.', '').replace('-', ''))
        is_possible_day = 1 <= num_val <= 31
    except ValueError:
        pass
    features['section_num_is_possible_day'] = 1 if is_possible_day else 0
    features['looks_like_date'] = 1 if is_possible_day and features['title_starts_with_month'] else 0
    
    # Section keywords
    keywords = ['introduction', 'scope', 'requirements', 'overview', 'summary',
               'description', 'specification', 'interface', 'design', 'test',
               'general', 'system', 'software', 'definitions', 'references', 'appendix']
    features['title_has_section_keyword'] = 1 if any(kw in title.lower() for kw in keywords) else 0
    
    # Sequence features
    features['is_logical_next'] = 0
    features['major_gap_from_prev'] = 0
    features['depth_change_from_prev'] = 0
    
    if idx > 0:
        prev = all_elements[idx - 1]
        if prev.get('type') == 'section':
            prev_num = prev.get('section_number', '')
            prev_parts = prev_num.replace(',', '.').replace('-', '.').split('.')
            prev_parts = [p.strip() for p in prev_parts if p.strip()]
            
            try:
                prev_major = int(prev_parts[0]) if prev_parts else -1
            except ValueError:
                prev_major = -1
            
            if major >= 0 and prev_major >= 0:
                features['major_gap_from_prev'] = major - prev_major
            
            features['depth_change_from_prev'] = depth - len(prev_parts)
            
            # Check logical next
            if len(parts) == len(prev_parts) and len(parts) > 0:
                try:
                    if parts[:-1] == prev_parts[:-1]:
                        if int(parts[-1]) == int(prev_parts[-1]) + 1:
                            features['is_logical_next'] = 1
                except ValueError:
                    pass
    
    # Parent exists check
    features['parent_exists'] = 0
    if depth > 1:
        parent = '.'.join(parts[:-1])
        for other in all_elements:
            if other.get('type') == 'section':
                other_num = other.get('section_number', '').replace(',', '.').replace('-', '.')
                if other_num.rstrip('.') == parent:
                    features['parent_exists'] = 1
                    break
    
    # Page and sequence position
    features['page_number'] = element.get('page_number', 0)
    features['sequence_position'] = idx
    
    return features


def filter_sections_with_model(
    elements: List[Dict],
    model,
    feature_columns: List[str],
    threshold: float = 0.5,
    verbose: bool = True
) -> List[Dict]:
    """
    Filter sections using trained ML model.
    
    Args:
        elements: List of document elements (sections, text blocks, figures, etc.)
        model: Trained XGBoost model
        feature_columns: List of feature column names the model expects
        threshold: Probability threshold for classification (default 0.5)
        verbose: Print filtering statistics
        
    Returns:
        Filtered list of elements with false positives removed
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required")
    
    # Separate sections from other elements
    sections = [(i, e) for i, e in enumerate(elements) if e.get('type') == 'section']
    
    if not sections:
        return elements
    
    # Extract features for all sections
    feature_rows = []
    for idx, (orig_idx, section) in enumerate(sections):
        features = extract_features_for_element(section, idx, [s for _, s in sections])
        feature_rows.append(features)
    
    # Create DataFrame and align with model's expected columns
    df = pd.DataFrame(feature_rows)
    
    # Add missing columns with default value 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Select only the columns the model expects, in the right order
    X = df[feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Get predictions
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    # Mark sections to remove
    sections_to_remove = set()
    for (orig_idx, section), pred, prob in zip(sections, predictions, probabilities):
        if pred == 0:  # Predicted as false positive
            sections_to_remove.add(orig_idx)
            if verbose:
                print(f"    Removing: {section.get('section_number', '?')} - {section.get('topic', '')[:40]}... (prob={prob:.3f})")
    
    # Filter elements
    filtered = [e for i, e in enumerate(elements) if i not in sections_to_remove]
    
    if verbose:
        removed_count = len(sections_to_remove)
        kept_count = len(sections) - removed_count
        print(f"\n  ML Filter Results:")
        print(f"    Sections analyzed: {len(sections)}")
        print(f"    Sections kept: {kept_count}")
        print(f"    Sections removed: {removed_count}")
    
    return filtered


def run_ml_filtering_on_file(
    input_path: str,
    output_path: str,
    model_path: str,
    threshold: float = 0.5,
    verbose: bool = True
):
    """
    Load a JSON file, filter sections with ML, and save result.
    
    This function can be integrated into simple_pipeline.py.
    """
    if verbose:
        print(f"  - Loading sections from: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"  - [Error] Input file not found: {input_path}")
        return
    
    # Load model
    if verbose:
        print(f"  - Loading model from: {model_path}")
    model, feature_columns, metrics = load_model(model_path)
    
    if verbose and metrics:
        print(f"    Model F1: {metrics.get('f1', 'N/A'):.3f}, Precision: {metrics.get('precision', 'N/A'):.3f}")
    
    # Load document
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle new format with page_metadata
    if isinstance(data, dict) and 'elements' in data:
        elements = data.get('elements', [])
        page_metadata = data.get('page_metadata', {})
    else:
        elements = data if isinstance(data, list) else []
        page_metadata = {}
    
    section_count_before = sum(1 for e in elements if e.get('type') == 'section')
    
    # Filter sections
    if verbose:
        print(f"  - Filtering sections (threshold={threshold})...")
    
    filtered_elements = filter_sections_with_model(
        elements, model, feature_columns, threshold, verbose
    )
    
    section_count_after = sum(1 for e in filtered_elements if e.get('type') == 'section')
    
    # Save result
    output_data = {
        'page_metadata': page_metadata,
        'elements': filtered_elements,
        'ml_filter_applied': True,
        'ml_threshold': threshold,
    }
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
    
    if verbose:
        print(f"  - Saved {len(filtered_elements)} elements ({section_count_after} sections)")
        print(f"  - Removed {section_count_before - section_count_after} false positive sections")
        print(f"  - Output: {output_path}")


# =============================================================================
# VALIDATION AGAINST TOC
# =============================================================================

def validate_against_toc(
    filtered_elements: List[Dict],
    toc_sections: Set[str],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate filtered sections against Table of Contents.
    
    Returns metrics comparing filtered output to TOC.
    """
    # Extract section numbers from filtered elements
    output_sections = set()
    for elem in filtered_elements:
        if elem.get('type') == 'section':
            section_num = elem.get('section_number', '')
            # Normalize
            normalized = section_num.replace(',', '.').replace('-', '.').strip('.')
            if normalized:
                output_sections.add(normalized)
    
    # Calculate metrics
    matched = toc_sections & output_sections
    missing = toc_sections - output_sections
    extra = output_sections - toc_sections
    
    total_unique = len(toc_sections | output_sections)
    
    results = {
        'toc_sections': len(toc_sections),
        'output_sections': len(output_sections),
        'matched': len(matched),
        'missing': len(missing),
        'extra': len(extra),
        'toc_coverage': (len(matched) / len(toc_sections) * 100) if toc_sections else 0,
        'precision': (len(matched) / len(output_sections) * 100) if output_sections else 0,
    }
    
    if verbose:
        print(f"\n  TOC Validation:")
        print(f"    TOC sections:    {results['toc_sections']}")
        print(f"    Output sections: {results['output_sections']}")
        print(f"    Matched:         {results['matched']}")
        print(f"    Missing:         {results['missing']}")
        print(f"    Extra:           {results['extra']}")
        print(f"    TOC Coverage:    {results['toc_coverage']:.1f}%")
        print(f"    Precision:       {results['precision']:.1f}%")
    
    return results


# =============================================================================
# ANALYSIS UTILITIES
# =============================================================================

def analyze_feature_correlations(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Analyze correlation of features with labels."""
    labeled = get_labeled_samples(df, min_label=0)
    
    if len(labeled) < 10:
        print("Not enough labeled samples for correlation analysis")
        return pd.DataFrame()
    
    labeled['_label'] = labeled['_label'].astype(float)
    feature_cols = [c for c in labeled.columns if not c.startswith('_')]
    
    correlations = []
    for col in feature_cols:
        if labeled[col].std() > 0:
            corr = labeled['_label'].corr(labeled[col])
            correlations.append({'feature': col, 'correlation': corr})
    
    corr_df = pd.DataFrame(correlations)
    corr_df['abs_correlation'] = corr_df['correlation'].abs()
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)
    
    print(f"\nTop {top_n} Feature Correlations with Label:")
    print("=" * 60)
    for _, row in corr_df.head(top_n).iterrows():
        indicator = "+++" if row['correlation'] > 0.3 else "++" if row['correlation'] > 0.2 else "+" if row['correlation'] > 0.1 else \
                   "---" if row['correlation'] < -0.3 else "--" if row['correlation'] < -0.2 else "-" if row['correlation'] < -0.1 else ""
        print(f"  {row['feature']:<45} {row['correlation']:>8.3f} {indicator}")
    
    return corr_df


def print_model_summary(model_path: str):
    """Print summary of a saved model."""
    model, feature_columns, metrics = load_model(model_path)
    
    print(f"\nModel Summary: {model_path}")
    print("=" * 60)
    print(f"Features used: {len(feature_columns)}")
    print(f"\nMetrics:")
    for key, value in metrics.items():
        if value is not None:
            print(f"  {key}: {value:.3f}")
    
    # Get feature importance from model
    importance = dict(zip(feature_columns, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    print(f"\nTop 15 Important Features:")
    for i, (feat, imp) in enumerate(list(importance.items())[:15]):
        print(f"  {i+1:2d}. {feat:<40s} {imp:.4f}")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ML-based section classification using XGBoost",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train a new model
    python section_classifier.py train -f training_features.csv -o model.joblib
    
    # Evaluate with cross-validation
    python section_classifier.py evaluate -f training_features.csv
    
    # Filter a document
    python section_classifier.py filter -i organized.json -o filtered.json -m model.joblib
    
    # Analyze feature correlations
    python section_classifier.py analyze -f training_features.csv
    
    # Print model summary
    python section_classifier.py summary -m model.joblib
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('-f', '--features', required=True, help='Path to training features CSV')
    train_parser.add_argument('-o', '--output', required=True, help='Path to save model')
    train_parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction (default: 0.2)')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model with cross-validation')
    eval_parser.add_argument('-f', '--features', required=True, help='Path to training features CSV')
    eval_parser.add_argument('--folds', type=int, default=5, help='Number of CV folds (default: 5)')
    
    # Filter command
    filter_parser = subparsers.add_parser('filter', help='Filter sections in a document')
    filter_parser.add_argument('-i', '--input', required=True, help='Input JSON file')
    filter_parser.add_argument('-o', '--output', required=True, help='Output JSON file')
    filter_parser.add_argument('-m', '--model', required=True, help='Path to trained model')
    filter_parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold (default: 0.5)')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze feature correlations')
    analyze_parser.add_argument('-f', '--features', required=True, help='Path to training features CSV')
    analyze_parser.add_argument('-n', '--top-n', type=int, default=20, help='Number of top features to show')
    
    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Print model summary')
    summary_parser.add_argument('-m', '--model', required=True, help='Path to model file')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        df = load_training_data(args.features)
        result = train_classifier(df, test_size=args.test_size)
        save_model(result, args.output)
        
    elif args.command == 'evaluate':
        df = load_training_data(args.features)
        
        # First do train/test split evaluation
        print("\n--- Train/Test Split Evaluation ---")
        result = train_classifier(df, test_size=0.2)
        
        # Then cross-validation
        print("\n--- Cross-Validation ---")
        cross_validate_model(df, n_folds=args.folds)
        
    elif args.command == 'filter':
        run_ml_filtering_on_file(
            args.input, args.output, args.model, 
            threshold=args.threshold, verbose=True
        )
        
    elif args.command == 'analyze':
        df = load_training_data(args.features)
        analyze_feature_correlations(df, top_n=args.top_n)
        
    elif args.command == 'summary':
        print_model_summary(args.model)
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()