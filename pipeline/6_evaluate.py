"""
Phase 6: Model Evaluation
This script evaluates all trained models and generates comprehensive performance reports.
IMPORTANT: Uses consistent label mapping and generates reproducible results.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)


def load_test_data_and_models(output_dir):
    """Load test data and all trained models."""
    print(f"\n{'='*60}")
    print("STEP 1: Loading Test Data and Models")
    print(f"{'='*60}")
    
    # Load test data
    X_test_path = os.path.join(output_dir, 'X_test.csv')
    y_test_path = os.path.join(output_dir, 'y_test.csv')
    
    if not os.path.exists(X_test_path):
        print(f"ERROR: X_test.csv not found at {X_test_path}")
        print("Please run pipeline/5_models.py first")
        sys.exit(1)
    
    if not os.path.exists(y_test_path):
        print(f"ERROR: y_test.csv not found at {y_test_path}")
        print("Please run pipeline/5_models.py first")
        sys.exit(1)
    
    try:
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path)['FTR']
        
        print(f"✓ Test data loaded successfully")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_test shape: {y_test.shape}")
        
    except Exception as e:
        print(f"ERROR: Failed to load test data: {e}")
        sys.exit(1)
    
    # Load models
    model_files = {
        "Logistic Regression": "model_logistic_regression.pkl",
        "Decision Tree": "model_decision_tree.pkl",
        "Random Forest": "model_random_forest.pkl",
        "Naive Bayes": "model_naive_bayes.pkl",
        "SVM": "model_svm.pkl"
    }
    
    models = {}
    loaded_count = 0
    
    print(f"\nLoading trained models:")
    print(f"{'='*50}")
    
    for name, filename in model_files.items():
        filepath = os.path.join(output_dir, filename)
        
        if os.path.exists(filepath):
            try:
                model = joblib.load(filepath)
                models[name] = model
                loaded_count += 1
                print(f"  ✓ {name:<25} loaded from {filename}")
            except Exception as e:
                print(f"  ✗ {name:<25} failed to load: {e}")
        else:
            print(f"  ✗ {name:<25} file not found: {filename}")
    
    if not models:
        print(f"\nERROR: No models could be loaded")
        sys.exit(1)
    
    print(f"\n✓ Successfully loaded {loaded_count}/{len(model_files)} models")
    print(f"\n✓ Step 1 completed successfully")
    
    return X_test, y_test, models


def generate_predictions(models, X_test):
    """Generate predictions for all models."""
    print(f"\n{'='*60}")
    print("STEP 2: Generating Predictions")
    print(f"{'='*60}")
    
    predictions = {}
    
    print(f"\nGenerating predictions for {len(models)} models...")
    print(f"{'='*50}")
    
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            predictions[name] = y_pred
            print(f"  ✓ {name:<25} predictions generated")
        except Exception as e:
            print(f"  ✗ {name:<25} prediction failed: {e}")
    
    print(f"\n✓ Step 2 completed successfully")
    
    return predictions


def print_classification_reports(y_test, predictions):
    """Print detailed classification reports for all models."""
    print(f"\n{'='*60}")
    print("STEP 3: Classification Reports")
    print(f"{'='*60}")
    
    # Define class labels
    class_labels = [0, 1, 2]
    class_names = ["Away Win", "Draw", "Home Win"]
    
    for name, y_pred in predictions.items():
        print(f"\n{'='*60}")
        print(f"===== {name} =====")
        print(f"{'='*60}")
        
        try:
            report = classification_report(
                y_test,
                y_pred,
                labels=class_labels,
                target_names=class_names,
                zero_division=0
            )
            print(report)
        except Exception as e:
            print(f"ERROR generating report: {e}")
        
        print(f"{'='*60}")
    
    print(f"\n✓ Step 3 completed successfully")


def build_results_summary(y_test, predictions):
    """Build comprehensive results summary dictionary."""
    print(f"\n{'='*60}")
    print("STEP 4: Building Results Summary")
    print(f"{'='*60}")
    
    # Define class labels
    class_labels = [0, 1, 2]
    class_names = ["Away Win", "Draw", "Home Win"]
    
    results = {}
    
    print(f"\nComputing metrics for each model...")
    print(f"{'='*50}")
    
    for name, y_pred in predictions.items():
        try:
            # Compute accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Get classification report as dictionary
            report_dict = classification_report(
                y_test,
                y_pred,
                labels=class_labels,
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
            
            # Extract weighted averages
            precision = report_dict['weighted avg']['precision']
            recall = report_dict['weighted avg']['recall']
            f1 = report_dict['weighted avg']['f1-score']
            
            # Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=class_labels)
            
            # Store results (convert numpy types to Python native types)
            results[name] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "confusion_matrix": cm.tolist()  # Convert numpy array to list
            }
            
            print(f"  ✓ {name:<25} metrics computed")
            print(f"      Accuracy:  {accuracy:.4f}")
            print(f"      Precision: {precision:.4f}")
            print(f"      Recall:    {recall:.4f}")
            print(f"      F1-Score:  {f1:.4f}")
            
        except Exception as e:
            print(f"  ✗ {name:<25} failed: {e}")
    
    print(f"\n✓ Step 4 completed successfully")
    
    return results


def save_results_json(results, output_dir):
    """Save results summary to JSON file."""
    print(f"\n{'='*60}")
    print("STEP 5: Saving Results to JSON")
    print(f"{'='*60}")
    
    output_path = os.path.join(output_dir, 'model_results.json')
    
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
        print(f"\n✓ Step 5 completed successfully")
        
    except Exception as e:
        print(f"ERROR: Failed to save JSON: {e}")


def create_confusion_matrix_charts(y_test, predictions, output_dir):
    """Create confusion matrix heatmaps for all models."""
    print(f"\n{'='*60}")
    print("STEP 6: Creating Confusion Matrix Charts")
    print(f"{'='*60}")
    
    # Define class labels
    class_labels = [0, 1, 2]
    class_names = ["Away Win", "Draw", "Home Win"]
    
    saved_charts = []
    
    print(f"\nGenerating confusion matrix charts...")
    print(f"{'='*50}")
    
    for name, y_pred in predictions.items():
        try:
            # Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=class_labels)
            
            # Create heatmap
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
            
            # Save chart
            filename = f"cm_{name.lower().replace(' ', '_')}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            saved_charts.append(filepath)
            print(f"  ✓ {name:<25} → {filename}")
            
        except Exception as e:
            print(f"  ✗ {name:<25} failed: {e}")
    
    print(f"\n✓ Step 6 completed successfully")
    print(f"✓ Saved {len(saved_charts)} confusion matrix charts")
    
    return saved_charts


def create_model_comparison_chart(results, output_dir):
    """Create grouped bar chart comparing all models."""
    print(f"\n{'='*60}")
    print("STEP 7: Creating Model Comparison Chart")
    print(f"{'='*60}")
    
    if not results:
        print(f"⚠ WARNING: No results to plot")
        return None
    
    # Extract data for plotting
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Prepare data
    data = {metric: [results[model][metric] for model in model_names] for metric in metrics}
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(model_names))
    width = 0.2
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = width * (i - 1.5)
        ax.bar(x + offset, data[metric], width, label=metric.capitalize(), color=color, alpha=0.8)
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart
    output_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Comparison chart saved to: {output_path}")
    print(f"\n✓ Step 7 completed successfully")
    
    return output_path


if __name__ == "__main__":
    print("\n" + "="*60)
    print("EPL MATCH DATA - MODEL EVALUATION")
    print("="*60)
    
    # Define paths
    output_dir = "pipeline/output"
    
    # Step 1: Load test data and models
    X_test, y_test, models = load_test_data_and_models(output_dir)
    
    # Step 2: Generate predictions
    predictions = generate_predictions(models, X_test)
    
    # Step 3: Print classification reports
    print_classification_reports(y_test, predictions)
    
    # Step 4: Build results summary
    results = build_results_summary(y_test, predictions)
    
    # Step 5: Save results to JSON
    save_results_json(results, output_dir)
    
    # Step 6: Create confusion matrix charts
    cm_charts = create_confusion_matrix_charts(y_test, predictions, output_dir)
    
    # Step 7: Create model comparison chart
    comparison_chart = create_model_comparison_chart(results, output_dir)
    
    # Final Summary
    print(f"\n{'='*60}")
    print("✓ ALL EVALUATION STEPS COMPLETED!")
    print(f"{'='*60}")
    
    print(f"\nOutput Files Generated:")
    print(f"  1. model_results.json")
    print(f"  2. model_comparison.png")
    for i, name in enumerate(models.keys(), 3):
        filename = f"cm_{name.lower().replace(' ', '_')}.png"
        print(f"  {i}. {filename}")
    
    # Find best model
    if results:
        print(f"\n{'='*60}")
        print("MODEL PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        
        best_model = max(results.items(), key=lambda x: x[1]['f1'])
        best_name = best_model[0]
        best_f1 = best_model[1]['f1']
        
        print(f"\n🏆 Best Model (by F1-Score): {best_name}")
        print(f"   F1-Score: {best_f1:.4f}")
        print(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
        print(f"   Precision: {best_model[1]['precision']:.4f}")
        print(f"   Recall: {best_model[1]['recall']:.4f}")
        
        print(f"\nAll Model Rankings (by F1-Score):")
        sorted_models = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
        for i, (name, metrics) in enumerate(sorted_models, 1):
            print(f"  {i}. {name:<25} F1: {metrics['f1']:.4f}")
    
    print(f"\nYou have completed all 6 phases of the data mining pipeline!")
    print(f"Next steps: Analyze results and consider hyperparameter tuning")
    print("="*60 + "\n")
