"""
Phase 1: Data Exploration and Visualization
This script loads the EPL match data and generates exploratory visualizations.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
    
    # Try both old and new column names
    goal_columns_map = {
        'FTHG': 'FullTimeHomeGoals',
        'FTAG': 'FullTimeAwayGoals',
        'HTHG': 'HalfTimeHomeGoals',
        'HTAG': 'HalfTimeAwayGoals'
    }
    
    available_columns = []
    for old_name, new_name in goal_columns_map.items():
        if new_name in df.columns:
            available_columns.append(new_name)
        elif old_name in df.columns:
            available_columns.append(old_name)
    
    if available_columns:
        print(f"\nDescriptive Statistics for Goal Columns:")
        stats = df[available_columns].describe()
        print(stats)
        
        print(f"\nMedian Values:")
        for col in available_columns:
            print(f"  {col}: {df[col].median()}")
    else:
        print("\n⚠ WARNING: None of the expected goal columns were found")
    
    print(f"\n✓ Step 2 completed successfully")


def match_outcome_distribution(df, output_dir):
    """Visualize match outcome distribution."""
    print(f"\n{'='*60}")
    print("STEP 3: Match Outcome Distribution")
    print(f"{'='*60}")
    
    # Try both old and new column names
    ftr_col = 'FullTimeResult' if 'FullTimeResult' in df.columns else 'FTR'
    
    if ftr_col not in df.columns:
        print("\n⚠ WARNING: FTR/FullTimeResult column not found. Skipping outcome distribution.")
        return
    
    outcome_counts = df[ftr_col].value_counts()
    print(f"\nMatch Outcome Frequency:")
    print(outcome_counts)
    print(f"\nPercentages:")
    print(df[ftr_col].value_counts(normalize=True) * 100)
    
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
    
    # Try both old and new column names
    fthg_col = 'FullTimeHomeGoals' if 'FullTimeHomeGoals' in df.columns else 'FTHG'
    ftag_col = 'FullTimeAwayGoals' if 'FullTimeAwayGoals' in df.columns else 'FTAG'
    
    if fthg_col not in df.columns or ftag_col not in df.columns:
        print("\n⚠ WARNING: Goal columns not found. Skipping goals analysis.")
        return
    
    # Create box plot
    plt.figure(figsize=(10, 6))
    data_to_plot = [df[fthg_col].dropna(), df[ftag_col].dropna()]
    box = plt.boxplot(data_to_plot, labels=['Home Goals', 'Away Goals'],
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
    
    # Try both old and new column names
    ftr_col = 'FullTimeResult' if 'FullTimeResult' in df.columns else 'FTR'
    
    if 'Season' not in df.columns or ftr_col not in df.columns:
        print("\n⚠ WARNING: Season or FTR/FullTimeResult column not found. Skipping season trends.")
        return
    
    # Group by Season and FTR
    season_outcomes = df.groupby(['Season', ftr_col]).size().unstack(fill_value=0)
    
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
    
    # Try both old and new column names
    ftr_col = 'FullTimeResult' if 'FullTimeResult' in df.columns else 'FTR'
    
    required_cols = ['HomeTeam', 'AwayTeam', ftr_col]
    if not all(col in df.columns for col in required_cols):
        print(f"\n⚠ WARNING: Required columns not found. Skipping top teams analysis.")
        return
    
    # Calculate home wins
    home_wins = df[df[ftr_col] == 'H']['HomeTeam'].value_counts()
    
    # Calculate away wins
    away_wins = df[df[ftr_col] == 'A']['AwayTeam'].value_counts()
    
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


def shots_analysis(df, output_dir):
    """Analyze shots and shots on target statistics."""
    print(f"\n{'='*60}")
    print("STEP 8: Shots Analysis")
    print(f"{'='*60}")
    
    # Check for shot columns
    hs_col = 'HomeShots' if 'HomeShots' in df.columns else 'HS'
    as_col = 'AwayShots' if 'AwayShots' in df.columns else 'AS'
    hst_col = 'HomeShotsOnTarget' if 'HomeShotsOnTarget' in df.columns else 'HST'
    ast_col = 'AwayShotsOnTarget' if 'AwayShotsOnTarget' in df.columns else 'AST'
    
    if not all(col in df.columns for col in [hs_col, as_col, hst_col, ast_col]):
        print("\n⚠ WARNING: Shot columns not found. Skipping shots analysis.")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Average shots comparison
    avg_home_shots = df[hs_col].mean()
    avg_away_shots = df[as_col].mean()
    avg_home_sot = df[hst_col].mean()
    avg_away_sot = df[ast_col].mean()
    
    axes[0, 0].bar(['Home Shots', 'Away Shots', 'Home SOT', 'Away SOT'],
                   [avg_home_shots, avg_away_shots, avg_home_sot, avg_away_sot],
                   color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    axes[0, 0].set_title('Average Shots per Match', fontweight='bold')
    axes[0, 0].set_ylabel('Average Count')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. Shot accuracy distribution
    df['home_shot_accuracy'] = (df[hst_col] / df[hs_col] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    df['away_shot_accuracy'] = (df[ast_col] / df[as_col] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    
    axes[0, 1].hist([df['home_shot_accuracy'], df['away_shot_accuracy']], 
                    bins=20, label=['Home', 'Away'], color=['#3498db', '#e74c3c'], alpha=0.7)
    axes[0, 1].set_title('Shot Accuracy Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Accuracy (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. Shots vs Goals correlation
    axes[1, 0].scatter(df[hs_col], df['FullTimeHomeGoals'] if 'FullTimeHomeGoals' in df.columns else df['FTHG'],
                      alpha=0.5, color='#3498db', label='Home')
    axes[1, 0].scatter(df[as_col], df['FullTimeAwayGoals'] if 'FullTimeAwayGoals' in df.columns else df['FTAG'],
                      alpha=0.5, color='#e74c3c', label='Away')
    axes[1, 0].set_title('Shots vs Goals Scored', fontweight='bold')
    axes[1, 0].set_xlabel('Total Shots')
    axes[1, 0].set_ylabel('Goals Scored')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Shots on target vs Goals
    axes[1, 1].scatter(df[hst_col], df['FullTimeHomeGoals'] if 'FullTimeHomeGoals' in df.columns else df['FTHG'],
                      alpha=0.5, color='#2ecc71', label='Home')
    axes[1, 1].scatter(df[ast_col], df['FullTimeAwayGoals'] if 'FullTimeAwayGoals' in df.columns else df['FTAG'],
                      alpha=0.5, color='#f39c12', label='Away')
    axes[1, 1].set_title('Shots on Target vs Goals', fontweight='bold')
    axes[1, 1].set_xlabel('Shots on Target')
    axes[1, 1].set_ylabel('Goals Scored')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'shots_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Chart saved: {output_path}")
    print(f"✓ Step 8 completed successfully")


def discipline_analysis(df, output_dir):
    """Analyze fouls and cards statistics."""
    print(f"\n{'='*60}")
    print("STEP 9: Discipline Analysis (Fouls & Cards)")
    print(f"{'='*60}")
    
    # Check for discipline columns
    hf_col = 'HomeFouls' if 'HomeFouls' in df.columns else 'HF'
    af_col = 'AwayFouls' if 'AwayFouls' in df.columns else 'AF'
    hy_col = 'HomeYellowCards' if 'HomeYellowCards' in df.columns else 'HY'
    ay_col = 'AwayYellowCards' if 'AwayYellowCards' in df.columns else 'AY'
    
    if not all(col in df.columns for col in [hf_col, af_col, hy_col, ay_col]):
        print("\n⚠ WARNING: Discipline columns not found. Skipping discipline analysis.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Average fouls per match
    avg_home_fouls = df[hf_col].mean()
    avg_away_fouls = df[af_col].mean()
    
    axes[0, 0].bar(['Home Fouls', 'Away Fouls'],
                   [avg_home_fouls, avg_away_fouls],
                   color=['#3498db', '#e74c3c'])
    axes[0, 0].set_title('Average Fouls per Match', fontweight='bold')
    axes[0, 0].set_ylabel('Average Count')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. Average yellow cards
    avg_home_yellows = df[hy_col].mean()
    avg_away_yellows = df[ay_col].mean()
    
    axes[0, 1].bar(['Home Yellow Cards', 'Away Yellow Cards'],
                   [avg_home_yellows, avg_away_yellows],
                   color=['#f39c12', '#e67e22'])
    axes[0, 1].set_title('Average Yellow Cards per Match', fontweight='bold')
    axes[0, 1].set_ylabel('Average Count')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. Fouls distribution
    axes[1, 0].hist([df[hf_col], df[af_col]], bins=20, 
                    label=['Home', 'Away'], color=['#3498db', '#e74c3c'], alpha=0.7)
    axes[1, 0].set_title('Fouls Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Number of Fouls')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. Yellow cards distribution
    axes[1, 1].hist([df[hy_col], df[ay_col]], bins=15,
                    label=['Home', 'Away'], color=['#f39c12', '#e67e22'], alpha=0.7)
    axes[1, 1].set_title('Yellow Cards Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Number of Yellow Cards')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'discipline_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Chart saved: {output_path}")
    print(f"✓ Step 9 completed successfully")


def halftime_analysis(df, output_dir):
    """Analyze half-time vs full-time results."""
    print(f"\n{'='*60}")
    print("STEP 10: Half-Time Analysis")
    print(f"{'='*60}")
    
    # Check for half-time columns
    hthg_col = 'HalfTimeHomeGoals' if 'HalfTimeHomeGoals' in df.columns else 'HTHG'
    htag_col = 'HalfTimeAwayGoals' if 'HalfTimeAwayGoals' in df.columns else 'HTAG'
    htr_col = 'HalfTimeResult' if 'HalfTimeResult' in df.columns else 'HTR'
    ftr_col = 'FullTimeResult' if 'FullTimeResult' in df.columns else 'FTR'
    
    if not all(col in df.columns for col in [hthg_col, htag_col, htr_col, ftr_col]):
        print("\n⚠ WARNING: Half-time columns not found. Skipping half-time analysis.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Half-time vs Full-time goals
    axes[0, 0].scatter(df[hthg_col], df['FullTimeHomeGoals'] if 'FullTimeHomeGoals' in df.columns else df['FTHG'],
                      alpha=0.5, color='#3498db', label='Home')
    axes[0, 0].plot([0, df[hthg_col].max()], [0, df[hthg_col].max()], 'r--', alpha=0.5)
    axes[0, 0].set_title('Half-Time vs Full-Time Goals (Home)', fontweight='bold')
    axes[0, 0].set_xlabel('Half-Time Goals')
    axes[0, 0].set_ylabel('Full-Time Goals')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Half-time result distribution
    ht_counts = df[htr_col].value_counts()
    axes[0, 1].bar(ht_counts.index, ht_counts.values, color=['#2ecc71', '#3498db', '#e74c3c'])
    axes[0, 1].set_title('Half-Time Result Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Result (H=Home, D=Draw, A=Away)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. HT to FT conversion matrix
    ht_ft_matrix = pd.crosstab(df[htr_col], df[ftr_col], normalize='index') * 100
    im = axes[1, 0].imshow(ht_ft_matrix.values, cmap='YlOrRd', aspect='auto')
    axes[1, 0].set_xticks(range(len(ht_ft_matrix.columns)))
    axes[1, 0].set_yticks(range(len(ht_ft_matrix.index)))
    axes[1, 0].set_xticklabels(ht_ft_matrix.columns)
    axes[1, 0].set_yticklabels(ht_ft_matrix.index)
    axes[1, 0].set_title('HT to FT Conversion (% of HT results)', fontweight='bold')
    axes[1, 0].set_xlabel('Full-Time Result')
    axes[1, 0].set_ylabel('Half-Time Result')
    
    # Add percentage labels
    for i in range(len(ht_ft_matrix.index)):
        for j in range(len(ht_ft_matrix.columns)):
            axes[1, 0].text(j, i, f'{ht_ft_matrix.values[i, j]:.1f}%',
                          ha='center', va='center', color='black', fontweight='bold')
    
    # 4. Second half goals
    df['second_half_home_goals'] = (df['FullTimeHomeGoals'] if 'FullTimeHomeGoals' in df.columns else df['FTHG']) - df[hthg_col]
    df['second_half_away_goals'] = (df['FullTimeAwayGoals'] if 'FullTimeAwayGoals' in df.columns else df['FTAG']) - df[htag_col]
    
    axes[1, 1].hist([df['second_half_home_goals'], df['second_half_away_goals']], 
                    bins=10, label=['Home', 'Away'], color=['#3498db', '#e74c3c'], alpha=0.7)
    axes[1, 1].set_title('Second Half Goals Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Goals Scored in 2nd Half')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'halftime_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Chart saved: {output_path}")
    print(f"✓ Step 10 completed successfully")


def home_advantage_analysis(df, output_dir):
    """Analyze home advantage statistics."""
    print(f"\n{'='*60}")
    print("STEP 11: Home Advantage Analysis")
    print(f"{'='*60}")
    
    ftr_col = 'FullTimeResult' if 'FullTimeResult' in df.columns else 'FTR'
    
    if ftr_col not in df.columns:
        print("\n⚠ WARNING: FTR column not found. Skipping home advantage analysis.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Win percentage by venue
    home_wins = (df[ftr_col] == 'H').sum()
    away_wins = (df[ftr_col] == 'A').sum()
    draws = (df[ftr_col] == 'D').sum()
    total = len(df)
    
    percentages = [home_wins/total*100, draws/total*100, away_wins/total*100]
    axes[0, 0].bar(['Home Win', 'Draw', 'Away Win'], percentages,
                   color=['#2ecc71', '#3498db', '#e74c3c'])
    axes[0, 0].set_title('Overall Win Distribution', fontweight='bold')
    axes[0, 0].set_ylabel('Percentage (%)')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for i, v in enumerate(percentages):
        axes[0, 0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # 2. Home advantage by season
    season_ha = df.groupby('Season')[ftr_col].apply(lambda x: (x == 'H').sum() / len(x) * 100)
    axes[0, 1].plot(season_ha.index, season_ha.values, marker='o', linewidth=2, color='#2ecc71')
    axes[0, 1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% baseline')
    axes[0, 1].set_title('Home Win % by Season', fontweight='bold')
    axes[0, 1].set_xlabel('Season')
    axes[0, 1].set_ylabel('Home Win %')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Goals scored: Home vs Away
    fthg_col = 'FullTimeHomeGoals' if 'FullTimeHomeGoals' in df.columns else 'FTHG'
    ftag_col = 'FullTimeAwayGoals' if 'FullTimeAwayGoals' in df.columns else 'FTAG'
    
    if fthg_col in df.columns and ftag_col in df.columns:
        avg_home_goals = df[fthg_col].mean()
        avg_away_goals = df[ftag_col].mean()
        
        axes[1, 0].bar(['Home Goals', 'Away Goals'], [avg_home_goals, avg_away_goals],
                      color=['#2ecc71', '#e74c3c'])
        axes[1, 0].set_title('Average Goals per Match', fontweight='bold')
        axes[1, 0].set_ylabel('Average Goals')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        for i, v in enumerate([avg_home_goals, avg_away_goals]):
            axes[1, 0].text(i, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold')
    
    # 4. Goal difference distribution
    if fthg_col in df.columns and ftag_col in df.columns:
        df['goal_diff'] = df[fthg_col] - df[ftag_col]
        axes[1, 1].hist(df['goal_diff'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Draw line')
        axes[1, 1].set_title('Goal Difference Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Goal Difference (Home - Away)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'home_advantage_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Chart saved: {output_path}")
    print(f"✓ Step 11 completed successfully")


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
    
    # Step 8: Shots analysis
    shots_analysis(df, output_dir)
    
    # Step 9: Discipline analysis
    discipline_analysis(df, output_dir)
    
    # Step 10: Half-time analysis
    halftime_analysis(df, output_dir)
    
    # Step 11: Home advantage analysis
    home_advantage_analysis(df, output_dir)
    
    print(f"\n{'='*60}")
    print("✓ ALL STEPS COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"\nGenerated files in {output_dir}:")
    print("  1. outcome_distribution.png")
    print("  2. goals_boxplot.png")
    print("  3. correlation_heatmap.png")
    print("  4. season_trends.png")
    print("  5. top_teams_wins.png")
    print("  6. shots_analysis.png")
    print("  7. discipline_analysis.png")
    print("  8. halftime_analysis.png")
    print("  9. home_advantage_analysis.png")
    print("\nYou can now proceed to Phase 2: Data Preprocessing")
    print("="*60 + "\n")


