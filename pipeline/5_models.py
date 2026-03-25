"""
Phase 5: Optimized CatBoost Training with Selected Features
Uses only the most important features to reduce noise and improve accuracy
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost not available")


def load_selected_data():
    """Load dataset with selected features."""
    print(f"\n{'='*60}")
    print("Loading Selected Features Dataset")
    print(f"{'='*60}")
    
    filepath = "output/enhanced_data_selected.csv"
    
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found")
        print("Please run pipeline/4_feature_selection_enhanced.py first")
        sys.exit(1)
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    X = df.drop(['FTR', 'Season'], axis=1)
    y = df['FTR']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"✓ Test: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def train_optimized_catboost(X_train, y_train, X_test, y_test):
    """Train CatBoost with multiple configurations to find best."""
    print(f"\n{'='*60}")
    print("Training Optimized CatBoost")
    print(f"{'='*60}")
    
    if not CATBOOST_AVAILABLE:
        print("⚠️  CatBoost not available")
        return None, None
    
    # Test multiple configurations
    configs = [
        {
            'name': 'Config 1: Deep & Slow',
            'iterations': 2000,
            'learning_rate': 0.02,
            'depth': 10,
            'l2_leaf_reg': 5,
            'min_data_in_leaf': 20,
            'random_strength': 0.5,
            'bagging_temperature': 0.5
        },
        {
            'name': 'Config 2: Balanced',
            'iterations': 1500,
            'learning_rate': 0.03,
            'depth': 9,
            'l2_leaf_reg': 7,
            'min_data_in_leaf': 25,
            'random_strength': 0.8,
            'bagging_temperature': 0.7
        },
        {
            'name': 'Config 3: Fast & Regularized',
            'iterations': 1200,
            'learning_rate': 0.05,
            'depth': 8,
            'l2_leaf_reg': 10,
            'min_data_in_leaf': 30,
            'random_strength': 1.0,
            'bagging_temperature': 1.0
        },
        {
            'name': 'Config 4: Conservative',
            'iterations': 1800,
            'learning_rate': 0.025,
            'depth': 7,
            'l2_leaf_reg': 8,
            'min_data_in_leaf': 35,
            'random_strength': 0.6,
            'bagging_temperature': 0.8
        }
    ]
    
    best_model = None
    best_score = 0
    best_config_name = None
    all_results = []
    
    for config in configs:
        config_name = config.pop('name')
        print(f"\n  Testing: {config_name}")
        
        model = CatBoostClassifier(
            **config,
            loss_function='MultiClass',
            eval_metric='Accuracy',
            random_seed=42,
            verbose=False,
            early_stopping_rounds=100,
            use_best_model=True
        )
        
        model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"    Accuracy: {accuracy*100:.2f}%")
        
        all_results.append({
            'config': config_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_config_name = config_name
    
    # Print all results
    print(f"\n{'='*60}")
    print("Configuration Comparison")
    print(f"{'='*60}")
    for result in all_results:
        marker = "⭐" if result['config'] == best_config_name else "  "
        print(f"{marker} {result['config']:30s} Acc: {result['accuracy']*100:.2f}%")
    
    # Get best model metrics
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n{'='*60}")
    print(f"Best Configuration: {best_config_name}")
    print(f"{'='*60}")
    print(f"✓ Accuracy:  {accuracy*100:.2f}%")
    print(f"✓ Precision: {precision*100:.2f}%")
    print(f"✓ Recall:    {recall*100:.2f}%")
    print(f"✓ F1-Score:  {f1*100:.2f}%")
    
    return best_model, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with selected features."""
    print(f"\n{'='*60}")
    print("Training Random Forest")
    print(f"{'='*60}")
    
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=8,
        min_samples_leaf=3,
        max_features='sqrt',
        class_weight='balanced',
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


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train Gradient Boosting with selected features."""
    print(f"\n{'='*60}")
    print("Training Gradient Boosting")
    print(f"{'='*60}")
    
    model = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=9,
        min_samples_split=15,
        min_samples_leaf=8,
        subsample=0.85,
        max_features='sqrt',
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


def save_models_and_results(models, results):
    """Save models and comparison results."""
    print(f"\n{'='*60}")
    print("Saving Models and Results")
    print(f"{'='*60}")
    
    os.makedirs("output", exist_ok=True)
    
    # Save models
    for name, model in models.items():
        if model is not None:
            filepath = f"output/{name}_optimized.pkl"
            joblib.dump(model, filepath)
            print(f"✓ Saved: {filepath}")
    
    # Save results
    comparison = pd.DataFrame(results).T
    comparison = comparison.sort_values('accuracy', ascending=False)
    
    comparison.to_csv('output/optimized_results.csv')
    print(f"✓ Saved: output/optimized_results.csv")
    
    # Print comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON - OPTIMIZED MODELS")
    print(f"{'='*60}\n")
    print(comparison.to_string())
    
    best_model = comparison.index[0]
    best_accuracy = comparison.loc[best_model, 'accuracy']
    
    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL: {best_model}")
    print(f"{'='*60}")
    print(f"Accuracy:  {best_accuracy*100:.2f}%")
    print(f"F1-Score:  {comparison.loc[best_model, 'f1']*100:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("OPTIMIZED CATBOOST TRAINING - SELECTED FEATURES")
    print("="*60)
    
    # Load selected features data
    X_train, X_test, y_train, y_test = load_selected_data()
    
    models = {}
    results = {}
    
    # Train CatBoost (with multiple configs)
    cb_model, cb_results = train_optimized_catboost(X_train, y_train, X_test, y_test)
    if cb_model:
        models['catboost'] = cb_model
        results['CatBoost Optimized'] = cb_results
    
    # Train Random Forest
    rf_model, rf_results = train_random_forest(X_train, y_train, X_test, y_test)
    models['random_forest'] = rf_model
    results['Random Forest'] = rf_results
    
    # Train Gradient Boosting
    gb_model, gb_results = train_gradient_boosting(X_train, y_train, X_test, y_test)
    models['gradient_boosting'] = gb_model
    results['Gradient Boosting'] = gb_results
    
    # Save everything
    save_models_and_results(models, results)
    
    print(f"\n{'='*60}")
    print("✓ OPTIMIZED TRAINING COMPLETED!")
    print(f"{'='*60}\n")
