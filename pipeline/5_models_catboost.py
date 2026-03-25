"""
Phase 5: Advanced Model Training with CatBoost, XGBoost, and LightGBM
Target: 55-60% accuracy with gradient boosting models
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try importing gradient boosting libraries
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost not available")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    print(f"⚠️  XGBoost not available: {str(e)[:100]}")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception as e:
    LIGHTGBM_AVAILABLE = False
    print(f"⚠️  LightGBM not available: {str(e)[:100]}")


def load_preprocessed_data(filepath):
    """Load the enhanced preprocessed data."""
    print(f"\n{'='*60}")
    print("STEP 1: Loading Enhanced Data")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        print("Please run pipeline/2_preprocess_enhanced.py first!")
        sys.exit(1)
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"✓ Step 1 completed successfully")
    
    return df


def prepare_train_test_split(df):
    """Split data into train and test sets."""
    print(f"\n{'='*60}")
    print("STEP 2: Preparing Train/Test Split")
    print(f"{'='*60}")
    
    X = df.drop(['FTR', 'Season'], axis=1)
    y = df['FTR']
    
    # Use stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    print(f"✓ Features: {X_train.shape[1]}")
    print(f"✓ Class distribution (train): {dict(y_train.value_counts())}")
    print(f"✓ Step 2 completed successfully")
    
    return X_train, X_test, y_train, y_test


def train_catboost(X_train, y_train, X_test, y_test):
    """Train CatBoost classifier with optimized hyperparameters."""
    print(f"\n{'='*60}")
    print("Training CatBoost Classifier")
    print(f"{'='*60}")
    
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        loss_function='MultiClass',
        eval_metric='TotalF1',
        random_seed=42,
        verbose=False,
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        verbose=False
    )
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"✓ Accuracy:  {accuracy*100:.2f}%")
    print(f"✓ Precision: {precision*100:.2f}%")
    print(f"✓ Recall:    {recall*100:.2f}%")
    print(f"✓ F1-Score:  {f1*100:.2f}%")
    
    return model, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier with optimized hyperparameters."""
    print(f"\n{'='*60}")
    print("Training XGBoost Classifier")
    print(f"{'='*60}")
    
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='mlogloss',
        early_stopping_rounds=50,
        verbose=False
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"✓ Accuracy:  {accuracy*100:.2f}%")
    print(f"✓ Precision: {precision*100:.2f}%")
    print(f"✓ Recall:    {recall*100:.2f}%")
    print(f"✓ F1-Score:  {f1*100:.2f}%")
    
    return model, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM classifier with optimized hyperparameters."""
    print(f"\n{'='*60}")
    print("Training LightGBM Classifier")
    print(f"{'='*60}")
    
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='multi_logloss'
    )
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"✓ Accuracy:  {accuracy*100:.2f}%")
    print(f"✓ Precision: {precision*100:.2f}%")
    print(f"✓ Recall:    {recall*100:.2f}%")
    print(f"✓ F1-Score:  {f1*100:.2f}%")
    
    return model, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train Gradient Boosting classifier."""
    print(f"\n{'='*60}")
    print("Training Gradient Boosting Classifier")
    print(f"{'='*60}")
    
    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"✓ Accuracy:  {accuracy*100:.2f}%")
    print(f"✓ Precision: {precision*100:.2f}%")
    print(f"✓ Recall:    {recall*100:.2f}%")
    print(f"✓ F1-Score:  {f1*100:.2f}%")
    
    return model, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def train_random_forest_baseline(X_train, y_train, X_test, y_test):
    """Train Random Forest as baseline."""
    print(f"\n{'='*60}")
    print("Training Random Forest (Baseline)")
    print(f"{'='*60}")
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"✓ Accuracy:  {accuracy*100:.2f}%")
    print(f"✓ Precision: {precision*100:.2f}%")
    print(f"✓ Recall:    {recall*100:.2f}%")
    print(f"✓ F1-Score:  {f1*100:.2f}%")
    
    return model, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def save_models(models_dict, output_dir):
    """Save all trained models."""
    print(f"\n{'='*60}")
    print("STEP 3: Saving Models")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for name, model in models_dict.items():
        filepath = os.path.join(output_dir, f'{name}.pkl')
        joblib.dump(model, filepath)
        print(f"✓ Saved: {name}.pkl")
    
    print(f"✓ Step 3 completed successfully")


def compare_results(results_dict):
    """Compare all model results."""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}\n")
    
    # Create comparison dataframe
    comparison = pd.DataFrame(results_dict).T
    comparison = comparison.sort_values('f1', ascending=False)
    
    print(comparison.to_string())
    
    # Find best model
    best_model = comparison.index[0]
    best_f1 = comparison.loc[best_model, 'f1']
    best_accuracy = comparison.loc[best_model, 'accuracy']
    
    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL: {best_model}")
    print(f"{'='*60}")
    print(f"Accuracy:  {best_accuracy*100:.2f}%")
    print(f"F1-Score:  {best_f1*100:.2f}%")
    print(f"{'='*60}\n")
    
    # Save comparison
    comparison.to_csv('output/model_comparison_catboost.csv')
    print("✓ Saved comparison to: output/model_comparison_catboost.csv")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADVANCED MODEL TRAINING - CATBOOST & GRADIENT BOOSTING")
    print("="*60)
    
    # Load data
    df = load_preprocessed_data("output/enhanced_data.csv")
    
    # Prepare split
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)
    
    # Train all models
    print(f"\n{'='*60}")
    print("TRAINING ALL MODELS")
    print(f"{'='*60}")
    
    models = {}
    results = {}
    
    # 1. Random Forest (baseline)
    rf_model, rf_results = train_random_forest_baseline(X_train, y_train, X_test, y_test)
    models['random_forest'] = rf_model
    results['Random Forest'] = rf_results
    
    # 2. Gradient Boosting
    gb_model, gb_results = train_gradient_boosting(X_train, y_train, X_test, y_test)
    models['gradient_boosting'] = gb_model
    results['Gradient Boosting'] = gb_results
    
    # 3. CatBoost
    if CATBOOST_AVAILABLE:
        cb_model, cb_results = train_catboost(X_train, y_train, X_test, y_test)
        models['catboost'] = cb_model
        results['CatBoost'] = cb_results
    else:
        print(f"\n{'='*60}")
        print("⚠️  Skipping CatBoost (not available)")
        print(f"{'='*60}")
    
    # 4. XGBoost
    if XGBOOST_AVAILABLE:
        xgb_model, xgb_results = train_xgboost(X_train, y_train, X_test, y_test)
        models['xgboost'] = xgb_model
        results['XGBoost'] = xgb_results
    else:
        print(f"\n{'='*60}")
        print("⚠️  Skipping XGBoost (not available)")
        print(f"{'='*60}")
    
    # 5. LightGBM
    if LIGHTGBM_AVAILABLE:
        lgbm_model, lgbm_results = train_lightgbm(X_train, y_train, X_test, y_test)
        models['lightgbm'] = lgbm_model
        results['LightGBM'] = lgbm_results
    else:
        print(f"\n{'='*60}")
        print("⚠️  Skipping LightGBM (not available)")
        print(f"{'='*60}")
    
    # Save models
    save_models(models, "output")
    
    # Compare results
    compare_results(results)
    
    print(f"\n{'='*60}")
    print("✓ ADVANCED MODEL TRAINING COMPLETED!")
    print(f"{'='*60}\n")
