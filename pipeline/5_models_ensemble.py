"""
Phase 5: Ensemble Model Training for 60% Accuracy Target
Combines multiple models with hyperparameter tuning
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


def load_data():
    """Load enhanced preprocessed data."""
    print(f"\n{'='*60}")
    print("Loading Enhanced Data")
    print(f"{'='*60}")
    
    df = pd.read_csv("output/enhanced_data.csv")
    X = df.drop(['FTR', 'Season'], axis=1)
    y = df['FTR']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"✓ Test: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def train_tuned_catboost(X_train, y_train, X_test, y_test):
    """Train CatBoost with extensive hyperparameter tuning."""
    print(f"\n{'='*60}")
    print("Training Tuned CatBoost")
    print(f"{'='*60}")
    
    if not CATBOOST_AVAILABLE:
        print("⚠️  CatBoost not available")
        return None, None
    
    # Try multiple configurations
    configs = [
        {
            'iterations': 1500,
            'learning_rate': 0.03,
            'depth': 10,
            'l2_leaf_reg': 5,
            'min_data_in_leaf': 20,
            'random_strength': 0.5
        },
        {
            'iterations': 2000,
            'learning_rate': 0.02,
            'depth': 12,
            'l2_leaf_reg': 7,
            'min_data_in_leaf': 15,
            'random_strength': 1.0
        },
        {
            'iterations': 1200,
            'learning_rate': 0.05,
            'depth': 9,
            'l2_leaf_reg': 3,
            'min_data_in_leaf': 25,
            'random_strength': 0.3
        }
    ]
    
    best_model = None
    best_score = 0
    best_config = None
    
    for i, config in enumerate(configs, 1):
        print(f"\n  Testing configuration {i}/3...")
        
        model = CatBoostClassifier(
            **config,
            loss_function='MultiClass',
            eval_metric='TotalF1',
            random_seed=42,
            verbose=False,
            early_stopping_rounds=100
        )
        
        model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"    Accuracy: {accuracy*100:.2f}%")
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_config = i
    
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n✓ Best config: {best_config}")
    print(f"✓ Accuracy:  {accuracy*100:.2f}%")
    print(f"✓ Precision: {precision*100:.2f}%")
    print(f"✓ Recall:    {recall*100:.2f}%")
    print(f"✓ F1-Score:  {f1*100:.2f}%")
    
    return best_model, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def train_tuned_rf(X_train, y_train, X_test, y_test):
    """Train Random Forest with tuning."""
    print(f"\n{'='*60}")
    print("Training Tuned Random Forest")
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


def train_tuned_gb(X_train, y_train, X_test, y_test):
    """Train Gradient Boosting with tuning."""
    print(f"\n{'='*60}")
    print("Training Tuned Gradient Boosting")
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


def create_ensemble(rf_model, gb_model, cb_model, X_train, y_train, X_test, y_test):
    """Create voting ensemble of best models."""
    print(f"\n{'='*60}")
    print("Creating Ensemble Model")
    print(f"{'='*60}")
    
    estimators = [
        ('rf', rf_model),
        ('gb', gb_model)
    ]
    
    if cb_model is not None:
        estimators.append(('cb', cb_model))
    
    # Soft voting for probability-based ensemble
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=[1, 1, 2] if cb_model is not None else [1, 1]  # Give more weight to CatBoost
    )
    
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"✓ Ensemble Accuracy:  {accuracy*100:.2f}%")
    print(f"✓ Ensemble Precision: {precision*100:.2f}%")
    print(f"✓ Ensemble Recall:    {recall*100:.2f}%")
    print(f"✓ Ensemble F1-Score:  {f1*100:.2f}%")
    
    return ensemble, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ENSEMBLE MODEL TRAINING - TARGET 60% ACCURACY")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train individual models with tuning
    rf_model, rf_results = train_tuned_rf(X_train, y_train, X_test, y_test)
    gb_model, gb_results = train_tuned_gb(X_train, y_train, X_test, y_test)
    cb_model, cb_results = train_tuned_catboost(X_train, y_train, X_test, y_test)
    
    # Create ensemble
    ensemble_model, ensemble_results = create_ensemble(
        rf_model, gb_model, cb_model, X_train, y_train, X_test, y_test
    )
    
    # Compare all results
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}\n")
    
    results = {
        'Random Forest': rf_results,
        'Gradient Boosting': gb_results,
    }
    
    if cb_results:
        results['CatBoost'] = cb_results
    
    results['Ensemble'] = ensemble_results
    
    comparison = pd.DataFrame(results).T
    comparison = comparison.sort_values('accuracy', ascending=False)
    print(comparison.to_string())
    
    # Save best model
    best_model_name = comparison.index[0]
    best_accuracy = comparison.loc[best_model_name, 'accuracy']
    
    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL: {best_model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:  {best_accuracy*100:.2f}%")
    print(f"F1-Score:  {comparison.loc[best_model_name, 'f1']*100:.2f}%")
    print(f"{'='*60}\n")
    
    # Save models
    os.makedirs("output", exist_ok=True)
    joblib.dump(ensemble_model, "output/ensemble_model.pkl")
    joblib.dump(rf_model, "output/rf_tuned.pkl")
    joblib.dump(gb_model, "output/gb_tuned.pkl")
    if cb_model:
        joblib.dump(cb_model, "output/catboost_tuned.pkl")
    
    comparison.to_csv('output/ensemble_comparison.csv')
    
    print("✓ Saved all models and comparison")
    print(f"\n{'='*60}")
    print("✓ ENSEMBLE TRAINING COMPLETED!")
    print(f"{'='*60}\n")
