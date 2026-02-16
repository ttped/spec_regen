"""
section_classifier.py - ML-based section classification using XGBoost.

Updated v3: Added linguistic features, enhanced blank title features, and position features.
Updated v4: Added OCR confidence and structural features from word-level Tesseract metadata.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from scipy.stats import uniform, randint
from typing import List, Dict, Tuple, Set, Any
from dataclasses import dataclass

# =============================================================================
# CURATED FEATURE LIST (v3)
# =============================================================================
# These features are selected based on correlation analysis and domain knowledge.
# New in v3:
#   - Linguistic features (lang_*): SpaCy-based title vs sentence detection
#   - Enhanced blank title features (title_is_blank, title_is_useless)
#   - Position features (is_in_bottom_third, late_page_*)
#   - Height features (relative_height, height_is_multiline)
#   - Major section magnitude features (major_section_gt_*)
# New in v4:
#   - OCR confidence features (ocr_conf_*): Word-level Tesseract confidence
#   - OCR structural features (ocr_spans_*, ocr_is_first_*): Block/paragraph hierarchy

SELECTED_FEATURES = [
    # === EXISTING HIGH-VALUE FEATURES ===
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
    
    # === NEW: LINGUISTIC FEATURES (SpaCy) ===
    # These help distinguish section titles from sentences
    'lang_root_is_verb',        # Root word is verb -> likely sentence, not title
    'lang_root_is_noun',        # Root word is noun -> likely title
    'lang_has_finite_verb',     # Has conjugated verb -> likely sentence
    'lang_has_verb',            # Has any verb
    'lang_starts_det',          # Starts with determiner (the, a, this)
    'lang_starts_imperative',   # Starts with imperative verb (command)
    'lang_has_subject',         # Has grammatical subject -> sentence structure
    'lang_is_complete_sentence', # Has subject + verb -> definitely sentence
    'lang_has_modal',           # Has modal verb (shall, must) -> requirement text
    
    # === NEW: ENHANCED BLANK TITLE FEATURES ===
    # Strong signals that a section is invalid
    'title_is_blank',           # Empty or whitespace only
    'title_is_useless',         # Blank, very short, or punctuation only
    'title_is_very_short',      # 1-2 characters only
    'title_is_punctuation_only', # Only punctuation/symbols
    'title_is_numeric_only',    # Title is just numbers
    
    # === NEW: POSITION WITHIN PAGE FEATURES ===
    # Late-page numbers are often junk (page numbers, table data, etc.)
    'page_position_y_pct',      # Vertical position (0=top, 1=bottom)
    'is_in_bottom_third',       # In bottom 33% of page
    'is_in_bottom_quarter',     # In bottom 25% of page
    'late_page_bad_sequence',   # Late page AND doesn't fit sequence
    'late_page_useless_title',  # Late page with blank/useless title
    
    # === NEW: HEIGHT FEATURES ===
    # Multi-line text blocks are not section headers
    'height_is_multiline',      # Bbox height > 1.8x median
    'height_is_very_tall',      # Bbox height > 3x median
    
    # === NEW: MAJOR SECTION MAGNITUDE ===
    # High major section numbers are suspicious
    'major_section_gt_6',       # Major section > 6 (soft signal)
    'major_section_gt_10',      # Major section > 10 (stronger signal)
    'major_section_gt_20',      # Major section > 20 (almost certainly wrong)
    
    # === NEW: OCR CONFIDENCE FEATURES ===
    # Word-level OCR confidence for the section's line
    'ocr_conf_mean',            # Average word confidence (0-100)
    'ocr_conf_min',             # Lowest word confidence in line
    'ocr_conf_std',             # Confidence spread across words
    'ocr_conf_below_50',        # Mean confidence < 50 -> likely misread
    'ocr_conf_below_30',        # Mean confidence < 30 -> very likely misread
    'ocr_has_low_conf_word',    # At least one word with conf < 50
    'ocr_mostly_low_conf',      # >50% of words have conf < 50
    'ocr_conf_high_spread',     # Max-min confidence > 50 -> mixed quality
    
    # === NEW: OCR STRUCTURAL FEATURES ===
    # Where the section sits in the Tesseract block/paragraph hierarchy
    'ocr_word_count',           # Words in the line (from Tesseract, not title)
    'ocr_spans_multiple_blocks', # Line spans >1 OCR block -> suspicious merge
    'ocr_spans_multiple_pars',  # Line spans >1 OCR paragraph -> suspicious
    'ocr_is_first_word_in_line', # First word in OCR line (word_num <= 1)
]

# =============================================================================
# MONOTONIC CONSTRAINTS
# =============================================================================
# Tell XGBoost which direction a feature is allowed to push predictions.
#   -1 = higher value can ONLY push prediction toward 0 (reject)
#   +1 = higher value can ONLY push prediction toward 1 (accept)
#    0 = no constraint (default, omitted features)
#
# This encodes domain knowledge without hard rules. The model still learns
# *how much* to penalize — it just can't learn the wrong direction.

MONOTONIC_CONSTRAINTS_MAP = {
    # --- Strong negative signals (higher = more likely false positive) ---
    'looks_like_date':            -1,  # Dates are never valid sections
    'title_is_blank':             -1,  # No title text = not a real section
    'title_is_useless':           -1,  # Blank/junk title
    'title_is_empty':             -1,  # Empty title
    'title_is_punctuation_only':  -1,  # Only punctuation
    'is_simple_number':           -1,  # Just a bare number, no title
    'content_is_empty':           -1,  # No content after the "section"
    'height_is_very_tall':        -1,  # Multi-line block, not a header
    'is_in_footer_zone':          -1,  # Footer area junk
    'lang_is_complete_sentence':  -1,  # Full sentence = body text, not title
    'lang_has_finite_verb':       -1,  # Conjugated verb = sentence
    'ocr_conf_below_30':          -1,  # Very low OCR confidence = misread
    'ocr_conf_below_50':          -1,  # Low OCR confidence
    'ocr_mostly_low_conf':        -1,  # Most words are low confidence
    'ocr_spans_multiple_blocks':  -1,  # Merged OCR blocks = suspicious
    'late_page_bad_sequence':     -1,  # Late page + bad sequence = junk
    'late_page_useless_title':    -1,  # Late page + no real title
    'major_section_gt_20':        -1,  # Section 20+ almost never real
    'section_num_ratio_suspicious': -1, # Ratio looks wrong
    
    # --- Strong positive signals (higher = more likely valid section) ---
    'toc_fuzzy_score':            +1,  # Matches table of contents
    'parent_exists':              +1,  # Has a parent section in the tree
    'sequence_fit_score':         +1,  # Fits the numbering sequence
    'extended_sequence_score':    +1,  # Extended sequence analysis
    'title_starts_capital':       +1,  # Capitalized title
    'format_consistency_score':   +1,  # Consistent with other sections
    'title_quality_score':        +1,  # Good title text
    'x_near_left_margin':         +1,  # Near left margin (typical for headers)
    'lang_root_is_noun':          +1,  # Noun-rooted = title-like
}


def _build_monotonic_constraints(available_features: list) -> tuple:
    """Build the monotonic_constraints tuple aligned to the available feature order."""
    return tuple(
        MONOTONIC_CONSTRAINTS_MAP.get(f, 0)
        for f in available_features
    )

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
    
    # Convert -1 labels to 0: these are valid numbers that appear inside tables,
    # so they should be treated as false positives (not real sections)
    table_section_count = (labeled_df['_label'] == -1).sum()
    labeled_df.loc[labeled_df['_label'] == -1, '_label'] = 0
    
    labeled_df = labeled_df[labeled_df['_label'].isin([0, 1])]
    
    pos_count = (labeled_df['_label'] == 1).sum()
    neg_count = (labeled_df['_label'] == 0).sum()
    print(f"  Labeled: {len(labeled_df)} ({pos_count} valid, {neg_count} false positive)")
    if table_section_count > 0:
        print(f"  (Converted {table_section_count} table sections [-1] to false positives [0])")
    
    if len(labeled_df) < 20:
        raise ValueError(f"Need >= 20 labeled rows, have {len(labeled_df)}")
    
    # Prepare features
    available = [f for f in SELECTED_FEATURES if f in df.columns]
    missing_feat = set(SELECTED_FEATURES) - set(available)
    print(f"  Features: {len(available)}/{len(SELECTED_FEATURES)}")
    
    X_labeled = labeled_df[available].fillna(0).replace([np.inf, -np.inf], 0)
    y_labeled = labeled_df['_label'].astype(int)
    
    scale_weight = (neg_count / pos_count) * 1.5 if pos_count > 0 else 1

    # Build monotonic constraints aligned to feature order
    monotonic_constraints = _build_monotonic_constraints(available)
    constrained_count = sum(1 for c in monotonic_constraints if c != 0)
    print(f"  Monotonic constraints: {constrained_count}/{len(available)} features constrained")

    # Search space for hyperparameter tuning
    param_distributions = {
        'n_estimators': randint(80, 300),
        'max_depth': randint(2, 7),
        'learning_rate': uniform(0.01, 0.19),       # 0.01 - 0.20
        'min_child_weight': randint(1, 7),
        'gamma': uniform(0.0, 0.5),
        'subsample': uniform(0.6, 0.4),             # 0.6 - 1.0
        'colsample_bytree': uniform(0.5, 0.5),      # 0.5 - 1.0
        'reg_alpha': uniform(0.0, 1.0),
        'reg_lambda': uniform(0.5, 2.0),             # 0.5 - 2.5
    }

    fixed_params = {
        'random_state': 42,
        'scale_pos_weight': scale_weight,
        'monotone_constraints': monotonic_constraints,
    }

    metrics = {}

    # =========================================================================
    # PASS 1: VALIDATION (80/20 Split) + Hyperparameter Tuning via CV
    # =========================================================================
    print(f"\n  --- Phase 1: Validation (80/20 Split) with Hyperparameter Tuning ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X_labeled, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=XGBClassifier(**fixed_params),
        param_distributions=param_distributions,
        n_iter=50,
        scoring='f1',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    
    best_params = {**fixed_params, **search.best_params_}
    print(f"  Best CV F1: {search.best_score_:.3f}")
    print(f"  Best params: { {k: (round(v, 4) if isinstance(v, float) else v) for k, v in search.best_params_.items()} }")

    val_model = search.best_estimator_
    val_pred = val_model.predict(X_test)
    v_prec, v_rec, v_f1, _ = precision_recall_fscore_support(y_test, val_pred, average='binary', pos_label=1)
    v_acc = accuracy_score(y_test, val_pred)
    v_conf = confusion_matrix(y_test, val_pred)
    
    metrics['val'] = {
        'precision': v_prec, 'recall': v_rec, 'f1': v_f1, 'accuracy': v_acc, 'confusion': v_conf
    }
    
    print(f"  Holdout Precision: {v_prec:.3f}, Recall: {v_rec:.3f}, F1: {v_f1:.3f}")
    print(f"  Confusion (Test Set): TN={v_conf[0,0]}, FP={v_conf[0,1]}, FN={v_conf[1,0]}, TP={v_conf[1,1]}")

    # =========================================================================
    # PASS 2: PRODUCTION - Retrain best params on 100% labeled data
    # =========================================================================
    print(f"\n  --- Phase 2: Full Training (100% Labeled Data) with Tuned Params ---")
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_labeled, y_labeled)
    
    # Check training error (sanity check)
    train_pred = final_model.predict(X_labeled)
    t_prec, t_rec, t_f1, _ = precision_recall_fscore_support(y_labeled, train_pred, average='binary', pos_label=1)
    t_acc = accuracy_score(y_labeled, train_pred)
    t_conf = confusion_matrix(y_labeled, train_pred)

    metrics['train'] = {
        'precision': t_prec, 'recall': t_rec, 'f1': t_f1, 'accuracy': t_acc, 'confusion': t_conf,
        'total_samples': len(labeled_df),
        'best_params': best_params,
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