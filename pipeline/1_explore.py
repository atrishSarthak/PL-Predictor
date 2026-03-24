"""
Phase 1: Data Exploration and Visualization
This script loads the EPL match data and generates exploratory visualizations.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_data(filepath):
    """Load the EPL matches CSV file."""
    print(f"\n{'='*60}")
    print("STEP 1: Loading and Inspecting Data")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        print("Please ensure the CSV file is placed in the data/ folder.")
        sys.exit(1)
    
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Successfully loaded data from {filepath}")
        return df
    except Exception as e:
        print(f"ERROR: Failed to load CSV file: {e}")
        sys.exit(1)


def inspect_data(df):
    """Print basic information about the dataset."""
    print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print(f"\nColumn Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\nData Types:")
    print(df.dtypes)
    
    print(f"\nFirst 5 Rows:")
    print(df.head())
    
    print(f"\nMissing Values Count:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found")
    
    print(f"\n✓ Step 1 completed successfully")


def statistical_summary(df):
    """Generate statistical summary for goal columns."""
    print(f"\n{'='*60}")
    print("STEP 2: Statistical Summary")
    print(f"{'='*60}")
    
    goal_columns = ['FTHG', 'FTAG', 'HTHG', 'HTAG']
    available_columns = [col for col in goal_columns if col in df.columns]
    missing_columns = [col for col in goal_columns if col not in df.columns]
    
    if missing_columns:
        print(f"\n⚠ WARNING: The following columns are missing: {', '.join(missing_columns)}")
    
    if available_columns:
        print(f"\nDescriptive Statistics for Goal Columns:")
        stats = df[available_columns].describe()
        print(stats)
        
        print(f"\nMedian Values:")
        for col in available_columns:
            print(f"  {col}: {df[col].median()}")
    else:
        print("\n⚠ WARNING: None of the expected goal columns (FTHG, FTAG, HTHG, HTAG) were found")
    
    print(f"\n✓ Step 2 completed successfully")


def match_outcome_distribution(df, output_dir):
    """Visualize match outcome distribution."""
    print(f"\n{'='*60}")
    print("STEP 3: Match Outcome Distribution")
    print(f"{'='*60}")
    
    if 'FTR' not in df.columns:
        print("\n⚠ WARNING: FTR column not found. Skipping outcome distribution.")
        return
    
    outcome_counts = df['FTR'].value_counts()
    print(f"\nMatch Outcome Frequency:")
    print(outcome_counts)
    print(f"\nPercentages:")
    print(df['FTR'].value_counts(normalize=True) * 100)
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    outcome_counts.plot(kind='bar', color=['#2ecc71', '#3498db', '#e74c3c'])
    plt.title('Match Outcome Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Outcome (H=Home Win, D=Draw, A=Away Win)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'outcome_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Chart saved: {output_path}")
    print(f"✓ Step 3 completed successfully")


def goals_analysis(df, output_dir):
    """Create box plot for goals analysis."""
    print(f"\n{'='*60}")
    print("STEP 4: Goals Analysis")
    print(f"{'='*60}")
    
    if 'FTHG' not in df.columns or 'FTAG' not in df.columns:
        print("\n⚠ WARNING: FTHG or FTAG columns not found. Skipping goals analysis.")
        return
    
    # Create box plot
    plt.figure(figsize=(10, 6))
    data_to_plot = [df['FTHG'].dropna(), df['FTAG'].dropna()]
    box = plt.boxplot(data_to_plot, labels=['Home Goals (FTHG)', 'Away Goals (FTAG)'],
                      patch_artist=True)
    
    # Color the boxes
    colors = ['#3498db', '#e74c3c']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title('Distribution of Goals: Home vs Away', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Goals', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'goals_boxplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Chart saved: {output_path}")
    print(f"✓ Step 4 completed successfully")


def correlation_heatmap(df, output_dir):
    """Generate correlation heatmap for numeric columns."""
    print(f"\n{'='*60}")
    print("STEP 5: Correlation Heatmap")
    print(f"{'='*60}")
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    if numeric_df.empty:
        print("\n⚠ WARNING: No numeric columns found. Skipping correlation heatmap.")
        return
    
    print(f"\nNumeric columns found: {list(numeric_df.columns)}")
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap of Numeric Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Chart saved: {output_path}")
    print(f"✓ Step 5 completed successfully")


def season_trends(df, output_dir):
    """Plot season-wise outcome trends."""
    print(f"\n{'='*60}")
    print("STEP 6: Season-wise Outcome Trends")
    print(f"{'='*60}")
    
    if 'Season' not in df.columns or 'FTR' not in df.columns:
        print("\n⚠ WARNING: Season or FTR column not found. Skipping season trends.")
        return
    
    # Group by Season and FTR
    season_outcomes = df.groupby(['Season', 'FTR']).size().unstack(fill_value=0)
    
    print(f"\nSeason-wise outcome counts:")
    print(season_outcomes)
    
    # Create line chart
    plt.figure(figsize=(14, 6))
    if 'H' in season_outcomes.columns:
        plt.plot(season_outcomes.index, season_outcomes['H'], marker='o', 
                label='Home Win', linewidth=2, color='#2ecc71')
    if 'D' in season_outcomes.columns:
        plt.plot(season_outcomes.index, season_outcomes['D'], marker='s', 
                label='Draw', linewidth=2, color='#3498db')
    if 'A' in season_outcomes.columns:
        plt.plot(season_outcomes.index, season_outcomes['A'], marker='^', 
                label='Away Win', linewidth=2, color='#e74c3c')
    
    plt.title('Match Outcomes Trend by Season', fontsize=16, fontweight='bold')
    plt.xlabel('Season', fontsize=12)
    plt.ylabel('Number of Matches', fontsize=12)
    plt.legend(fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'season_trends.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Chart saved: {output_path}")
    print(f"✓ Step 6 completed successfully")


def top_teams_wins(df, output_dir):
    """Calculate and visualize top teams by total wins."""
    print(f"\n{'='*60}")
    print("STEP 7: Top Teams by Wins")
    print(f"{'='*60}")
    
    required_cols = ['HomeTeam', 'AwayTeam', 'FTR']
    if not all(col in df.columns for col in required_cols):
        print(f"\n⚠ WARNING: Required columns {required_cols} not found. Skipping top teams analysis.")
        return
    
    # Calculate home wins
    home_wins = df[df['FTR'] == 'H']['HomeTeam'].value_counts()
    
    # Calculate away wins
    away_wins = df[df['FTR'] == 'A']['AwayTeam'].value_counts()
    
    # Combine total wins
    total_wins = home_wins.add(away_wins, fill_value=0).sort_values(ascending=False)
    
    print(f"\nTop 20 Teams by Total Wins:")
    print(total_wins.head(20))
    
    # Create horizontal bar chart
    plt.figure(figsize=(12, 10))
    top_20 = total_wins.head(20)
    plt.barh(range(len(top_20)), top_20.values, color='#3498db', alpha=0.8)
    plt.yticks(range(len(top_20)), top_20.index)
    plt.xlabel('Total Wins', fontsize=12)
    plt.ylabel('Team', fontsize=12)
    plt.title('Top 20 Teams by Total Wins (Home + Away)', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()  # Highest at top
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'top_teams_wins.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Chart saved: {output_path}")
    print(f"✓ Step 7 completed successfully")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("EPL MATCH DATA EXPLORATION")
    print("="*60)
    
    # Define paths
    data_path = "data/epl_matches.csv"
    output_dir = "pipeline/output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n✓ Output directory ready: {output_dir}")
    
    # Load data
    df = load_data(data_path)
    
    # Step 1: Inspect data
    inspect_data(df)
    
    # Step 2: Statistical summary
    statistical_summary(df)
    
    # Step 3: Match outcome distribution
    match_outcome_distribution(df, output_dir)
    
    # Step 4: Goals analysis
    goals_analysis(df, output_dir)
    
    # Step 5: Correlation heatmap
    correlation_heatmap(df, output_dir)
    
    # Step 6: Season trends
    season_trends(df, output_dir)
    
    # Step 7: Top teams by wins
    top_teams_wins(df, output_dir)
    
    print(f"\n{'='*60}")
    print("✓ ALL STEPS COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"\nGenerated files in {output_dir}:")
    print("  1. outcome_distribution.png")
    print("  2. goals_boxplot.png")
    print("  3. correlation_heatmap.png")
    print("  4. season_trends.png")
    print("  5. top_teams_wins.png")
    print("\nYou can now proceed to Phase 2: Data Preprocessing")
    print("="*60 + "\n")
