"""
Phase 6: Evaluation for CatBoost and Enhanced Models
Evaluates the advanced models trained with enhanced features
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)


def load_data_and_models():
    """Load enhanced data and trained models."""
    print(f"\n{'='*60}")
    print("Loading Data and Models")
    print(f"{'='*60}")
    
    # Load enhanced data
    df = pd.read_csv("output/enhanced_data.csv")
    X = df.drop(['FTR', 'Season'], axis=1)
    y = df['FTR']
    
    # Same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Load models
    model_files = {
        "CatBoost (Tuned)": "catboost_tuned.pkl",
        "CatBoost": "catboost.pkl",
        "Ensemble": "ensemble_model.pkl",
        "Random Forest (Tuned)": "rf_tuned.pkl",
        "Gradient Boosting (Tuned)": "gb_tuned.pkl",
        "Random Forest": "random_forest.pkl",
        "Gradient Boosting": "gradient_boosting.pkl"
    }
    
    models = {}
    for name, filename in model_files.items():
        filepath = os.path.join("output", filename)
        if os.path.exists(filepath):
            try:
                models[name] = joblib.load(filepath)
                print(f"✓ Loaded: {name}")
            except Exception as e:
                print(f"⚠️  Failed to load {name}: {str(e)[:50]}")
    
    if not models:
        print("ERROR: No models found!")
        sys.exit(1)
    
    print(f"\n✓ Loaded {len(models)} models")
    
    return X_test, y_test, models


def evaluate_all_models(X_test, y_test, models):
    """Evaluate all models and return results."""
    print(f"\n{'='*60}")
    print("Evaluating All Models")
    print(f"{'='*60}\n")
    
    class_labels = [0, 1, 2]
    class_names = ["Away Win", "Draw", "Home Win"]
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            predictions[name] = y_pred
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            cm = confusion_matrix(y_test, y_pred, labels=class_labels)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm.tolist()
            }
            
            print(f"{name}:")
            print(f"  Accuracy:  {accuracy*100:.2f}%")
            print(f"  Precision: {precision*100:.2f}%")
            print(f"  Recall:    {recall*100:.2f}%")
            print(f"  F1-Score:  {f1*100:.2f}%")
            print()
            
        except Exception as e:
            print(f"⚠️  {name} failed: {str(e)[:50]}\n")
    
    return results, predictions


def create_comparison_chart(results):
    """Create model comparison chart."""
    print(f"\n{'='*60}")
    print("Creating Comparison Chart")
    print(f"{'='*60}")
    
    df = pd.DataFrame(results).T
    df = df.sort_values('accuracy', ascending=False)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(df))
    width = 0.2
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = width * (i - 1.5)
        ax.bar(x + offset, df[metric], width, label=metric.capitalize(), color=color, alpha=0.8)
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('CatBoost & Enhanced Models - Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=25, ha='right')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/catboost_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: output/catboost_comparison.png")


def create_confusion_matrices(y_test, predictions):
    """Create confusion matrix for best models."""
    print(f"\n{'='*60}")
    print("Creating Confusion Matrices")
    print(f"{'='*60}")
    
    class_labels = [0, 1, 2]
    class_names = ["Away Win", "Draw", "Home Win"]
    
    # Create for top 3 models
    top_models = list(predictions.keys())[:3]
    
    for name in top_models:
        if name not in predictions:
            continue
        
        y_pred = predictions[name]
        cm = confusion_matrix(y_test, y_pred, labels=class_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'},
            square=True,
            linewidths=1,
            linecolor='gray'
        )
        
        plt.title(f'Confusion Matrix - {name}', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted', fontsize=12, fontweight='bold')
        plt.ylabel('Actual', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        filename = f"cm_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(f'output/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: output/{filename}")


def save_results(results):
    """Save results to JSON and CSV."""
    print(f"\n{'='*60}")
    print("Saving Results")
    print(f"{'='*60}")
    
    # Save JSON
    with open('output/catboost_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Saved: output/catboost_results.json")
    
    # Save CSV
    df = pd.DataFrame(results).T
    df = df.drop('confusion_matrix', axis=1)
    df = df.sort_values('accuracy', ascending=False)
    df.to_csv('output/catboost_results.csv')
    print("✓ Saved: output/catboost_results.csv")


def print_summary(results):
    """Print final summary."""
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}\n")
    
    df = pd.DataFrame(results).T
    df = df.drop('confusion_matrix', axis=1)
    df = df.sort_values('accuracy', ascending=False)
    
    print(df.to_string())
    
    best_model = df.index[0]
    best_accuracy = df.loc[best_model, 'accuracy']
    best_f1 = df.loc[best_model, 'f1']
    
    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL: {best_model}")
    print(f"{'='*60}")
    print(f"Accuracy:  {best_accuracy*100:.2f}%")
    print(f"Precision: {df.loc[best_model, 'precision']*100:.2f}%")
    print(f"Recall:    {df.loc[best_model, 'recall']*100:.2f}%")
    print(f"F1-Score:  {best_f1*100:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CATBOOST & ENHANCED MODELS - EVALUATION")
    print("="*60)
    
    # Load data and models
    X_test, y_test, models = load_data_and_models()
    
    # Evaluate all models
    results, predictions = evaluate_all_models(X_test, y_test, models)
    
    # Create visualizations
    create_comparison_chart(results)
    create_confusion_matrices(y_test, predictions)
    
    # Save results
    save_results(results)
    
    # Print summary
    print_summary(results)
    
    print(f"\n{'='*60}")
    print("✓ EVALUATION COMPLETED!")
    print(f"{'='*60}\n")
