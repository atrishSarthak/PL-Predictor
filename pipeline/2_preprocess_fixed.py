"""
Phase 2: Data Preprocessing (FIXED - No Data Leakage)
This script creates PRE-MATCH features using only historical data.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_raw_data(filepath):
    """Load the raw EPL matches CSV file."""
    print(f"\n{'='*60}")
    print("STEP 1: Loading Raw Data")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        sys.exit(1)
    
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Successfully loaded data from {filepath}")
        print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Rename columns if needed
        column_mapping = {
            'MatchDate': 'Date',
            'FullTimeHomeGoals': 'FTHG',
            'FullTimeAwayGoals': 'FTAG',
            'FullTimeResult': 'FTR'
        }
        df = df.rename(columns=column_mapping)
        
        print(f"\n✓ Step 1 completed successfully")
        return df
    except Exception as e:
        print(f"ERROR: Failed to load CSV file: {e}")
        sys.exit(1)


def clean_basic_data(df):
    """Clean and prepare basic data."""
    print(f"\n{'='*60}")
    print("STEP 2: Basic Data Cleaning")
    print(f"{'='*60}")
    
    # Rename columns for consistency
    column_mapping = {
        'HomeShots': 'HS',
        'AwayShots': 'AS',
        'HomeShotsOnTarget': 'HST',
        'AwayShotsOnTarget': 'AST',
        'HomeFouls': 'HF',
        'AwayFouls': 'AF',
        'HomeYellowCards': 'HY',
        'AwayYellowCards': 'AY',
        'HalfTimeHomeGoals': 'HTHG',
        'HalfTimeAwayGoals': 'HTAG'
    }
    df = df.rename(columns=column_mapping)
    
    # Keep essential columns including match statistics
    required_cols = ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
                     'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HY', 'AY', 'HTHG', 'HTAG']
    existing_cols = [col for col in required_cols if col in df.columns]
    
    df = df[existing_cols].copy()
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Sort by date (CRITICAL for time-series features)
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"\n✓ Cleaned dataset: {df.shape[0]} rows")
    print(f"✓ Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"✓ Columns available: {len(existing_cols)}")
    print(f"\n✓ Step 2 completed successfully")
    
    return df


def calculate_team_form(df, team, is_home, n_matches=5):
    """Calculate team's form from last N matches."""
    # Filter matches for this team
    if is_home:
        team_matches = df[df['HomeTeam'] == team].copy()
        goals_for = 'FTHG'
        goals_against = 'FTAG'
        shots = 'HS'
        shots_on_target = 'HST'
        fouls = 'HF'
        yellows = 'HY'
        ht_goals_for = 'HTHG'
        ht_goals_against = 'HTAG'
        result_map = {'H': 3, 'D': 1, 'A': 0}
    else:
        team_matches = df[df['AwayTeam'] == team].copy()
        goals_for = 'FTAG'
        goals_against = 'FTHG'
        shots = 'AS'
        shots_on_target = 'AST'
        fouls = 'AF'
        yellows = 'AY'
        ht_goals_for = 'HTAG'
        ht_goals_against = 'HTHG'
        result_map = {'A': 3, 'D': 1, 'H': 0}
    
    if len(team_matches) < n_matches:
        return {
            'wins': 0, 'draws': 0, 'losses': 0,
            'goals_scored': 0.0, 'goals_conceded': 0.0, 'points': 0,
            'shots': 0.0, 'shots_on_target': 0.0, 'shot_accuracy': 0.0,
            'fouls': 0.0, 'yellows': 0.0,
            'ht_goals_scored': 0.0, 'ht_goals_conceded': 0.0
        }
    
    recent = team_matches.tail(n_matches)
    
    # Basic stats
    wins = sum(recent['FTR'].map(result_map) == 3)
    draws = sum(recent['FTR'].map(result_map) == 1)
    losses = sum(recent['FTR'].map(result_map) == 0)
    goals_scored = recent[goals_for].mean()
    goals_conceded = recent[goals_against].mean()
    points = recent['FTR'].map(result_map).sum()
    
    # Advanced stats
    avg_shots = recent[shots].mean() if shots in recent.columns else 0.0
    avg_shots_on_target = recent[shots_on_target].mean() if shots_on_target in recent.columns else 0.0
    shot_accuracy = (avg_shots_on_target / avg_shots * 100) if avg_shots > 0 else 0.0
    avg_fouls = recent[fouls].mean() if fouls in recent.columns else 0.0
    avg_yellows = recent[yellows].mean() if yellows in recent.columns else 0.0
    
    # Half-time stats
    ht_goals_scored = recent[ht_goals_for].mean() if ht_goals_for in recent.columns else 0.0
    ht_goals_conceded = recent[ht_goals_against].mean() if ht_goals_against in recent.columns else 0.0
    
    return {
        'wins': wins, 'draws': draws, 'losses': losses,
        'goals_scored': goals_scored, 'goals_conceded': goals_conceded, 'points': points,
        'shots': avg_shots, 'shots_on_target': avg_shots_on_target, 'shot_accuracy': shot_accuracy,
        'fouls': avg_fouls, 'yellows': avg_yellows,
        'ht_goals_scored': ht_goals_scored, 'ht_goals_conceded': ht_goals_conceded
    }


def calculate_h2h_stats(df, home_team, away_team, n_matches=3):
    """Calculate head-to-head statistics."""
    # Find all matches between these teams
    h2h = df[
        ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
        ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
    ].copy()
    
    if len(h2h) < n_matches:
        return {
            'h2h_home_wins': 0,
            'h2h_away_wins': 0,
            'h2h_draws': 0
        }
    
    # Get last N meetings
    recent_h2h = h2h.tail(n_matches)
    
    # Count results from home team's perspective
    home_wins = 0
    away_wins = 0
    draws = 0
    
    for _, match in recent_h2h.iterrows():
        if match['HomeTeam'] == home_team:
            if match['FTR'] == 'H':
                home_wins += 1
            elif match['FTR'] == 'A':
                away_wins += 1
            else:
                draws += 1
        else:  # home_team was away in this match
            if match['FTR'] == 'A':
                home_wins += 1
            elif match['FTR'] == 'H':
                away_wins += 1
            else:
                draws += 1
    
    return {
        'h2h_home_wins': home_wins,
        'h2h_away_wins': away_wins,
        'h2h_draws': draws
    }


def engineer_features(df):
    """Engineer pre-match features using only historical data."""
    print(f"\n{'='*60}")
    print("STEP 3: Engineering Pre-Match Features")
    print(f"{'='*60}")
    print("\n⚠️  This will take a few minutes as we calculate rolling statistics...")
    
    features_list = []
    
    for idx in range(len(df)):
        if idx % 500 == 0:
            print(f"  Processing match {idx}/{len(df)}...")
        
        current_match = df.iloc[idx]
        
        # Get all matches BEFORE this one (no data leakage!)
        historical_df = df.iloc[:idx]
        
        if len(historical_df) < 10:
            # Skip first few matches (not enough history)
            continue
        
        home_team = current_match['HomeTeam']
        away_team = current_match['AwayTeam']
        
        # Calculate home team form (last 5 HOME matches)
        home_form = calculate_team_form(historical_df, home_team, is_home=True, n_matches=5)
        
        # Calculate away team form (last 5 AWAY matches)
        away_form = calculate_team_form(historical_df, away_team, is_home=False, n_matches=5)
        
        # Calculate head-to-head stats
        h2h_stats = calculate_h2h_stats(historical_df, home_team, away_team, n_matches=3)
        
        # Create feature dictionary
        features = {
            # Match info
            'Season': current_match['Season'],
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            
            # Home team form
            'home_last5_wins': home_form['wins'],
            'home_last5_points': home_form['points'],
            'home_last5_goals_scored': home_form['goals_scored'],
            'home_last5_goals_conceded': home_form['goals_conceded'],
            
            # Away team form
            'away_last5_wins': away_form['wins'],
            'away_last5_points': away_form['points'],
            'away_last5_goals_scored': away_form['goals_scored'],
            'away_last5_goals_conceded': away_form['goals_conceded'],
            
            # Advanced stats - Shots
            'ShotDiff': home_form['shots'] - away_form['shots'],
            'ShotOnTargetDiff': home_form['shots_on_target'] - away_form['shots_on_target'],
            'ShotAccuracyHome': home_form['shot_accuracy'],
            'ShotAccuracyAway': away_form['shot_accuracy'],
            
            # Advanced stats - Discipline
            'FoulDiff': home_form['fouls'] - away_form['fouls'],
            'YellowCardDiff': home_form['yellows'] - away_form['yellows'],
            
            # Advanced stats - Half-time performance
            'GoalDiff_HT': home_form['ht_goals_scored'] - away_form['ht_goals_conceded'],
            
            # Strength indicators
            'WinRateDiff': (home_form['wins'] / 5.0) - (away_form['wins'] / 5.0),
            'StrengthDiff': home_form['points'] - away_form['points'],
            
            # Head-to-head
            'h2h_home_wins': h2h_stats['h2h_home_wins'],
            'h2h_away_wins': h2h_stats['h2h_away_wins'],
            
            # Target variable
            'FTR': current_match['FTR']
        }
        
        features_list.append(features)
    
    # Create new dataframe
    df_features = pd.DataFrame(features_list)
    
    print(f"\n✓ Created {len(df_features)} matches with pre-match features")
    print(f"✓ Features per match: {len(df_features.columns) - 1}")
    print(f"\n✓ Step 3 completed successfully")
    
    return df_features


def encode_target(df):
    """Encode target variable FTR."""
    print(f"\n{'='*60}")
    print("STEP 4: Encoding Target Variable")
    print(f"{'='*60}")
    
    # Encode FTR: H=2, D=1, A=0
    ftr_mapping = {'H': 2, 'D': 1, 'A': 0}
    df['FTR'] = df['FTR'].map(ftr_mapping)
    
    print(f"\n✓ Encoded FTR: H=2 (Home Win), D=1 (Draw), A=0 (Away Win)")
    print(f"\nClass distribution:")
    for val, count in df['FTR'].value_counts().sort_index().items():
        label = {2: 'Home Win', 1: 'Draw', 0: 'Away Win'}[val]
        pct = count / len(df) * 100
        print(f"  {val} ({label}): {count} ({pct:.1f}%)")
    
    print(f"\n✓ Step 4 completed successfully")
    
    return df


def encode_teams(df):
    """Encode team names as numeric values."""
    print(f"\n{'='*60}")
    print("STEP 5: Encoding Team Names")
    print(f"{'='*60}")
    
    # Use LabelEncoder for teams
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    
    # Get all unique teams
    all_teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
    
    # Fit encoders on all teams
    le_home.fit(all_teams)
    le_away.fit(all_teams)
    
    # Transform
    df['HomeTeam_encoded'] = le_home.transform(df['HomeTeam'])
    df['AwayTeam_encoded'] = le_away.transform(df['AwayTeam'])
    
    print(f"\n✓ Encoded {len(all_teams)} unique teams")
    print(f"✓ Created HomeTeam_encoded and AwayTeam_encoded features")
    
    # Drop original team names
    df = df.drop(['HomeTeam', 'AwayTeam'], axis=1)
    
    print(f"\n✓ Step 5 completed successfully")
    
    return df


def save_cleaned_data(df, output_dir, output_filename):
    """Save the cleaned dataset to CSV."""
    print(f"\n{'='*60}")
    print("STEP 6: Saving Cleaned Dataset")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Cleaned dataset saved to: {output_path}")
    print(f"\nFinal Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nFinal Features:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\n{'='*60}")
    print("IMPORTANT: NO DATA LEAKAGE!")
    print(f"{'='*60}")
    print("All features use only HISTORICAL data available before each match.")
    print("The model cannot 'cheat' by seeing the final score.")
    print("Expected accuracy: 50-55% (realistic for football prediction)")
    
    print(f"\n✓ Step 6 completed successfully")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("EPL MATCH DATA PREPROCESSING (FIXED - NO DATA LEAKAGE)")
    print("="*60)
    
    # Define paths
    input_path = "data/epl_matches.csv"
    output_dir = "pipeline/output"
    output_filename = "cleaned_data.csv"
    
    # Step 1: Load raw data
    df = load_raw_data(input_path)
    
    # Step 2: Basic cleaning
    df = clean_basic_data(df)
    
    # Step 3: Engineer pre-match features
    df = engineer_features(df)
    
    # Step 4: Encode target
    df = encode_target(df)
    
    # Step 5: Encode teams
    df = encode_teams(df)
    
    # Step 6: Save cleaned data
    save_cleaned_data(df, output_dir, output_filename)
    
    print(f"\n{'='*60}")
    print("✓ ALL PREPROCESSING STEPS COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"\nOutput File: {output_dir}/{output_filename}")
    print(f"\nYou can now proceed to Phase 3: Handling Class Imbalance")
    print("="*60 + "\n")
