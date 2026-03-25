"""
Phase 4: Feature Selection for Enhanced Data
Analyzes feature importance and selects top features to reduce noise and improve accuracy
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

sns.set_style("whitegrid")


def load_enhanced_data():
    """Load enhanced preprocessed data."""
    print(f"\n{'='*60}")
    print("Loading Enhanced Data")
    print(f"{'='*60}")
    
    filepath = "output/enhanced_data.csv"
    
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found")
        print("Please run pipeline/2_preprocess_enhanced.py first")
        sys.exit(1)
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


def analyze_feature_importance(df):
    """Analyze feature importance using both Random Forest and CatBoost."""
    print(f"\n{'='*60}")
    print("Analyzing Feature Importance")
    print(f"{'='*60}")
    
    X = df.drop(['FTR', 'Season'], axis=1)
    y = df['FTR']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    feature_names = X.columns.tolist()
    importance_scores = {}
    
    # 1. Random Forest Feature Importance
    print("\n1. Training Random Forest for feature importance...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    rf_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_scores['Random Forest'] = rf_importance
    print(f"✓ Random Forest importance calculated")
    
    # 2. CatBoost Feature Importance
    if CATBOOST_AVAILABLE:
        print("\n2. Training CatBoost for feature importance...")
        cb = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=8,
            random_seed=42,
            verbose=False
        )
        cb.fit(X_train, y_train, verbose=False)
        
        cb_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': cb.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_scores['CatBoost'] = cb_importance
        print(f"✓ CatBoost importance calculated")
    
    return importance_scores, feature_names


def select_top_features(importance_scores, feature_names, top_n=50):
    """Select top N features based on combined importance."""
    print(f"\n{'='*60}")
    print(f"Selecting Top {top_n} Features")
    print(f"{'='*60}")
    
    # Combine importance scores
    combined_scores = {}
    
    for model_name, importance_df in importance_scores.items():
        for _, row in importance_df.iterrows():
            feature = row['feature']
            importance = row['importance']
            
            if feature not in combined_scores:
                combined_scores[feature] = []
            combined_scores[feature].append(importance)
    
    # Average importance across models
    avg_importance = {
        feature: np.mean(scores) 
        for feature, scores in combined_scores.items()
    }
    
    # Sort by average importance
    sorted_features = sorted(
        avg_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Select top N features
    selected_features = [feature for feature, _ in sorted_features[:top_n]]
    
    print(f"\n✓ Selected {len(selected_features)} features")
    print(f"\nTop 20 Most Important Features:")
    print("="*60)
    for i, (feature, importance) in enumerate(sorted_features[:20], 1):
        print(f"{i:2d}. {feature:40s} {importance:.6f}")
    
    return selected_features, sorted_features


def create_feature_importance_chart(sorted_features, top_n=30):
    """Create feature importance visualization."""
    print(f"\n{'='*60}")
    print("Creating Feature Importance Chart")
    print(f"{'='*60}")
    
    # Get top N features
    top_features = sorted_features[:top_n]
    features = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]
    
    # Create horizontal bar chart
    plt.figure(figsize=(12, 10))
    y_pos = np.arange(len(features))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    
    plt.barh(y_pos, importances, color=colors)
    plt.yticks(y_pos, features, fontsize=9)
    plt.xlabel('Average Feature Importance', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    filepath = 'output/feature_importance_enhanced.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filepath}")


def save_selected_data(df, selected_features):
    """Save dataset with only selected features."""
    print(f"\n{'='*60}")
    print("Saving Selected Features Dataset")
    print(f"{'='*60}")
    
    # Keep selected features + target + season
    columns_to_keep = selected_features + ['FTR', 'Season']
    df_selected = df[columns_to_keep].copy()
    
    output_path = 'output/enhanced_data_selected.csv'
    df_selected.to_csv(output_path, index=False)
    
    print(f"✓ Saved: {output_path}")
    print(f"✓ Shape: {df_selected.shape[0]} rows, {df_selected.shape[1]} columns")
    
    # Save feature list
    feature_list_path = 'output/selected_features_enhanced.txt'
    with open(feature_list_path, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    
    print(f"✓ Saved feature list: {feature_list_path}")
    
    return df_selected


def compare_feature_sets(original_count, selected_count):
    """Compare original vs selected feature sets."""
    print(f"\n{'='*60}")
    print("Feature Set Comparison")
    print(f"{'='*60}")
    
    reduction = original_count - selected_count
    reduction_pct = (reduction / original_count) * 100
    
    print(f"\nOriginal features:  {original_count}")
    print(f"Selected features:  {selected_count}")
    print(f"Reduction:          {reduction} features ({reduction_pct:.1f}%)")
    print(f"\nBenefits of feature reduction:")
    print("  ✓ Reduced noise and randomness")
    print("  ✓ Faster training time")
    print("  ✓ Better generalization")
    print("  ✓ Reduced overfitting risk")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ENHANCED FEATURE SELECTION")
    print("="*60)
    
    # Load data
    df = load_enhanced_data()
    original_feature_count = df.shape[1] - 2  # Exclude FTR and Season
    
    # Analyze feature importance
    importance_scores, feature_names = analyze_feature_importance(df)
    
    # Select top features (try 50, 40, 30)
    # Start with 50 to keep most important features
    selected_features, sorted_features = select_top_features(
        importance_scores, 
        feature_names, 
        top_n=50
    )
    
    # Create visualization
    create_feature_importance_chart(sorted_features, top_n=30)
    
    # Save selected dataset
    df_selected = save_selected_data(df, selected_features)
    
    # Compare feature sets
    compare_feature_sets(original_feature_count, len(selected_features))
    
    print(f"\n{'='*60}")
    print("✓ FEATURE SELECTION COMPLETED!")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Run: python3 5_models_catboost_optimized.py")
    print("2. Compare accuracy with full feature set")
    print("3. If accuracy improves, use selected features")
    print("="*60 + "\n")
