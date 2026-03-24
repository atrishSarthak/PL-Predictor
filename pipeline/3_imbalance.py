"""
Phase 3: Handling Class Imbalance
This script analyzes class distribution and applies balancing techniques if needed.
IMPORTANT: Uses intelligent decision-making to avoid unnecessary SMOTE on sports data.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def load_cleaned_data(filepath):
    """Load the cleaned dataset from preprocessing step."""
    print(f"\n{'='*60}")
    print("STEP 1: Loading Cleaned Data")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        print("Please run pipeline/2_preprocess.py first to generate cleaned_data.csv")
        sys.exit(1)
    
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Successfully loaded data from {filepath}")
        print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"\nColumn Names: {list(df.columns)}")
        
        print(f"\n✓ Step 1 completed successfully")
        return df
    except Exception as e:
        print(f"ERROR: Failed to load CSV file: {e}")
        sys.exit(1)


def analyze_class_distribution(df, output_dir):
    """Analyze and visualize the class distribution of the target variable."""
    print(f"\n{'='*60}")
    print("STEP 2: Analyzing Class Distribution")
    print(f"{'='*60}")
    
    if 'FTR' not in df.columns:
        print(f"\nERROR: Target column 'FTR' not found in dataset")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Get class counts
    class_counts = df['FTR'].value_counts().sort_index()
    total_samples = len(df)
    
    print(f"\nClass Distribution (FTR - Full Time Result):")
    print(f"{'='*40}")
    
    class_labels = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    
    for class_val in sorted(class_counts.index):
        count = class_counts[class_val]
        percentage = (count / total_samples) * 100
        label = class_labels.get(class_val, f'Unknown ({class_val})')
        print(f"  Class {class_val} ({label:10s}): {count:5d} samples ({percentage:5.2f}%)")
    
    print(f"\nTotal Samples: {total_samples}")
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    
    # Prepare data for plotting
    plot_labels = [class_labels.get(val, f'Class {val}') for val in sorted(class_counts.index)]
    plot_values = [class_counts[val] for val in sorted(class_counts.index)]
    colors = ['#e74c3c', '#3498db', '#2ecc71']  # Away Win (red), Draw (blue), Home Win (green)
    
    bars = plt.bar(plot_labels, plot_values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/total_samples*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.title('Class Distribution Before Balancing', fontsize=16, fontweight='bold')
    plt.xlabel('Match Outcome', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'class_distribution_before.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Chart saved: {output_path}")
    print(f"\n✓ Step 2 completed successfully")
    
    return class_counts, total_samples


def define_features_and_target(df):
    """Separate features (X) and target (y) for modeling."""
    print(f"\n{'='*60}")
    print("STEP 3: Defining Features and Target")
    print(f"{'='*60}")
    
    # Columns to exclude from features
    exclude_columns = ['FTR', 'Date', 'Season', 'HomeTeam', 'AwayTeam']
    
    # Get feature columns (all except excluded ones)
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    if not feature_columns:
        print(f"\nERROR: No feature columns found after excluding {exclude_columns}")
        sys.exit(1)
    
    if 'FTR' not in df.columns:
        print(f"\nERROR: Target column 'FTR' not found")
        sys.exit(1)
    
    # Create X and y
    X = df[feature_columns].copy()
    y = df['FTR'].copy()
    
    print(f"\nFeature List ({len(feature_columns)} features):")
    for i, col in enumerate(feature_columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\nExcluded Columns: {exclude_columns}")
    print(f"\nShape of X (features): {X.shape}")
    print(f"Shape of y (target):   {y.shape}")
    
    print(f"\n✓ Step 3 completed successfully")
    
    return X, y, feature_columns


def decide_imbalance_strategy(class_counts, total_samples):
    """
    Intelligently decide whether to apply SMOTE based on class distribution.
    
    Decision criteria:
    - If all classes have >= 20% representation: NO SMOTE (use class weights)
    - If minority class has < 20%: APPLY SMOTE
    """
    print(f"\n{'='*60}")
    print("STEP 4: Deciding Imbalance Strategy")
    print(f"{'='*60}")
    
    # Calculate percentages
    percentages = (class_counts / total_samples) * 100
    min_percentage = percentages.min()
    max_percentage = percentages.max()
    
    print(f"\nClass Distribution Analysis:")
    print(f"  Minimum class percentage: {min_percentage:.2f}%")
    print(f"  Maximum class percentage: {max_percentage:.2f}%")
    print(f"  Imbalance ratio: {max_percentage / min_percentage:.2f}:1")
    
    # Decision threshold: 20%
    THRESHOLD = 20.0
    
    print(f"\n{'='*40}")
    print("DECISION LOGIC:")
    print(f"{'='*40}")
    
    if min_percentage >= THRESHOLD:
        apply_smote = False
        print(f"\n✓ DECISION: DO NOT APPLY SMOTE")
        print(f"\nREASON:")
        print(f"  - All classes have >= {THRESHOLD}% representation")
        print(f"  - Minimum class: {min_percentage:.2f}%")
        print(f"  - Class distribution is acceptable for sports match data")
        print(f"  - SMOTE could introduce unrealistic synthetic matches")
        print(f"  - RECOMMENDATION: Use class_weight='balanced' in models instead")
        print(f"\nBENEFITS OF THIS APPROACH:")
        print(f"  ✓ Preserves real match data integrity")
        print(f"  ✓ Avoids synthetic data artifacts")
        print(f"  ✓ Prevents potential overfitting")
        print(f"  ✓ Maintains temporal relationships in sports data")
    else:
        apply_smote = True
        print(f"\n✓ DECISION: APPLY SMOTE")
        print(f"\nREASON:")
        print(f"  - Minority class has only {min_percentage:.2f}% representation")
        print(f"  - This is below the {THRESHOLD}% threshold")
        print(f"  - Significant imbalance detected (ratio: {max_percentage / min_percentage:.2f}:1)")
        print(f"  - SMOTE will help balance the classes")
        print(f"\nWARNINGS:")
        print(f"  ⚠ SMOTE creates synthetic samples")
        print(f"  ⚠ May not capture true match dynamics")
        print(f"  ⚠ Could introduce unrealistic feature combinations")
        print(f"  ⚠ Monitor model performance on real test data")
    
    print(f"\n✓ Step 4 completed successfully")
    
    return apply_smote


def apply_smote_balancing(X, y, output_dir):
    """Apply SMOTE to balance the classes."""
    print(f"\n{'='*60}")
    print("STEP 5: Applying SMOTE")
    print(f"{'='*60}")
    
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print(f"\nERROR: imbalanced-learn library not found")
        print(f"Please install it using: pip install imbalanced-learn")
        sys.exit(1)
    
    print(f"\nClass Distribution BEFORE SMOTE:")
    class_counts_before = y.value_counts().sort_index()
    for class_val, count in class_counts_before.items():
        print(f"  Class {class_val}: {count} samples")
    
    # Apply SMOTE
    print(f"\nApplying SMOTE (Synthetic Minority Over-sampling Technique)...")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    print(f"\nClass Distribution AFTER SMOTE:")
    class_counts_after = pd.Series(y_balanced).value_counts().sort_index()
    for class_val, count in class_counts_after.items():
        print(f"  Class {class_val}: {count} samples")
    
    print(f"\nSamples Added:")
    total_before = len(y)
    total_after = len(y_balanced)
    print(f"  Before: {total_before}")
    print(f"  After:  {total_after}")
    print(f"  Added:  {total_after - total_before} synthetic samples")
    
    # Create bar chart for AFTER distribution
    plt.figure(figsize=(10, 6))
    
    class_labels = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    plot_labels = [class_labels.get(val, f'Class {val}') for val in sorted(class_counts_after.index)]
    plot_values = [class_counts_after[val] for val in sorted(class_counts_after.index)]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    bars = plt.bar(plot_labels, plot_values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/total_after*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.title('Class Distribution After SMOTE Balancing', fontsize=16, fontweight='bold')
    plt.xlabel('Match Outcome', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'class_distribution_after.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Chart saved: {output_path}")
    print(f"\n✓ Step 5 completed successfully")
    
    return X_balanced, y_balanced


def save_balanced_dataset(X, y, feature_columns, output_dir, smote_applied):
    """Save the final dataset (balanced or original)."""
    print(f"\n{'='*60}")
    print("STEP 6: Saving Output Dataset")
    print(f"{'='*60}")
    
    # Combine X and y back into a DataFrame
    df_output = X.copy()
    df_output['FTR'] = y
    
    # Save to CSV
    output_path = os.path.join(output_dir, 'balanced_data.csv')
    df_output.to_csv(output_path, index=False)
    
    print(f"\n✓ Dataset saved to: {output_path}")
    print(f"\nFinal Dataset Shape: {df_output.shape[0]} rows, {df_output.shape[1]} columns")
    print(f"\nSMOTE Applied: {'YES' if smote_applied else 'NO'}")
    
    if smote_applied:
        print(f"\nNOTE: This dataset contains synthetic samples generated by SMOTE")
        print(f"      Use this for training, but evaluate on real test data only")
    else:
        print(f"\nNOTE: This dataset contains only original samples")
        print(f"      Consider using class_weight='balanced' in your models")
    
    print(f"\n✓ Step 6 completed successfully")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("EPL MATCH DATA - CLASS IMBALANCE HANDLING")
    print("="*60)
    
    # Define paths
    input_path = "pipeline/output/cleaned_data.csv"
    output_dir = "pipeline/output"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load cleaned data
    df = load_cleaned_data(input_path)
    
    # Step 2: Analyze class distribution
    class_counts, total_samples = analyze_class_distribution(df, output_dir)
    
    # Step 3: Define features and target
    X, y, feature_columns = define_features_and_target(df)
    
    # Step 4: Decide whether to apply SMOTE
    apply_smote = decide_imbalance_strategy(class_counts, total_samples)
    
    # Step 5: Apply SMOTE if needed
    if apply_smote:
        X_final, y_final = apply_smote_balancing(X, y, output_dir)
        smote_applied = True
    else:
        print(f"\n{'='*60}")
        print("STEP 5: SMOTE Application")
        print(f"{'='*60}")
        print(f"\n✓ SMOTE SKIPPED due to acceptable class balance")
        print(f"\nThe original cleaned dataset will be used as-is.")
        print(f"No synthetic samples will be generated.")
        print(f"\n✓ Step 5 completed successfully")
        X_final = X
        y_final = y
        smote_applied = False
    
    # Step 6: Save final dataset
    save_balanced_dataset(X_final, y_final, feature_columns, output_dir, smote_applied)
    
    print(f"\n{'='*60}")
    print("✓ ALL CLASS IMBALANCE HANDLING STEPS COMPLETED!")
    print(f"{'='*60}")
    print(f"\nOutput Files Generated:")
    print(f"  1. {output_dir}/class_distribution_before.png")
    if smote_applied:
        print(f"  2. {output_dir}/class_distribution_after.png")
    print(f"  3. {output_dir}/balanced_data.csv")
    print(f"\nSMOTE Applied: {'YES' if smote_applied else 'NO'}")
    print(f"\nYou can now proceed to Phase 4: Feature Selection")
    print("="*60 + "\n")
