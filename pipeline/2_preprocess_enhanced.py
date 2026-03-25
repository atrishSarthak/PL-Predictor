"""
Phase 2: Enhanced Preprocessing for 60% Accuracy Target
This script creates extensive features for maximum prediction accuracy.
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
        
        # Rename columns
        column_mapping = {
            'MatchDate': 'Date',
            'FullTimeHomeGoals': 'FTHG',
            'FullTimeAwayGoals': 'FTAG',
            'FullTimeResult': 'FTR',
            'HalfTimeHomeGoals': 'HTHG',
            'HalfTimeAwayGoals': 'HTAG',
            'HomeShots': 'HS',
            'AwayShots': 'AS',
            'HomeShotsOnTarget': 'HST',
            'AwayShotsOnTarget': 'AST',
            'HomeFouls': 'HF',
            'AwayFouls': 'AF',
            'HomeYellowCards': 'HY',
            'AwayYellowCards': 'AY'
        }
        df = df.rename(columns=column_mapping)
        
        # Keep all relevant columns
        required_cols = ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
                        'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HY', 'AY', 'HTHG', 'HTAG']
        existing_cols = [col for col in required_cols if col in df.columns]
        
        df = df[existing_cols].copy()
        df = df.dropna()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"✓ Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"✓ Step 1 completed successfully")
        
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def calculate_extended_team_stats(df, team, is_home, n_matches=10):
    """Calculate extended team statistics from last N matches."""
    if is_home:
        team_matches = df[df['HomeTeam'] == team].copy()
        gf, ga = 'FTHG', 'FTAG'
        shots, sot = 'HS', 'HST'
        fouls, yellows = 'HF', 'HY'
        htgf, htga = 'HTHG', 'HTAG'
        result_map = {'H': 3, 'D': 1, 'A': 0}
    else:
        team_matches = df[df['AwayTeam'] == team].copy()
        gf, ga = 'FTAG', 'FTHG'
        shots, sot = 'AS', 'AST'
        fouls, yellows = 'AF', 'AY'
        htgf, htga = 'HTAG', 'HTHG'
        result_map = {'A': 3, 'D': 1, 'H': 0}
    
    if len(team_matches) < n_matches:
        return {f'last{n_matches}_' + k: 0.0 for k in [
            'wins', 'points', 'gf', 'ga', 'gd', 'shots', 'sot', 'shot_acc',
            'fouls', 'yellows', 'htgf', 'htga', 'clean_sheets', 'failed_to_score',
            'win_rate', 'ppg', 'goals_per_shot'
        ]}
    
    recent = team_matches.tail(n_matches)
    
    wins = sum(recent['FTR'].map(result_map) == 3)
    points = recent['FTR'].map(result_map).sum()
    goals_for = recent[gf].sum()
    goals_against = recent[ga].sum()
    goal_diff = goals_for - goals_against
    
    total_shots = recent[shots].sum()
    total_sot = recent[sot].sum()
    shot_acc = (total_sot / total_shots * 100) if total_shots > 0 else 0
    
    total_fouls = recent[fouls].sum()
    total_yellows = recent[yellows].sum()
    
    ht_goals_for = recent[htgf].sum()
    ht_goals_against = recent[htga].sum()
    
    clean_sheets = sum(recent[ga] == 0)
    failed_to_score = sum(recent[gf] == 0)
    
    win_rate = wins / n_matches
    ppg = points / n_matches
    goals_per_shot = goals_for / total_shots if total_shots > 0 else 0
    
    return {
        f'last{n_matches}_wins': wins,
        f'last{n_matches}_points': points,
        f'last{n_matches}_gf': goals_for,
        f'last{n_matches}_ga': goals_against,
        f'last{n_matches}_gd': goal_diff,
        f'last{n_matches}_shots': total_shots,
        f'last{n_matches}_sot': total_sot,
        f'last{n_matches}_shot_acc': shot_acc,
        f'last{n_matches}_fouls': total_fouls,
        f'last{n_matches}_yellows': total_yellows,
        f'last{n_matches}_htgf': ht_goals_for,
        f'last{n_matches}_htga': ht_goals_against,
        f'last{n_matches}_clean_sheets': clean_sheets,
        f'last{n_matches}_failed_to_score': failed_to_score,
        f'last{n_matches}_win_rate': win_rate,
        f'last{n_matches}_ppg': ppg,
        f'last{n_matches}_goals_per_shot': goals_per_shot
    }


def calculate_season_stats(df, team, is_home, current_date):
    """Calculate season-to-date statistics."""
    season_matches = df[df['Date'] < current_date]
    
    if is_home:
        team_matches = season_matches[season_matches['HomeTeam'] == team]
        gf, ga = 'FTHG', 'FTAG'
        result_map = {'H': 3, 'D': 1, 'A': 0}
    else:
        team_matches = season_matches[season_matches['AwayTeam'] == team]
        gf, ga = 'FTAG', 'FTHG'
        result_map = {'A': 3, 'D': 1, 'H': 0}
    
    if len(team_matches) == 0:
        return {'season_points': 0, 'season_gf': 0, 'season_ga': 0, 'season_matches': 0}
    
    points = team_matches['FTR'].map(result_map).sum()
    goals_for = team_matches[gf].sum()
    goals_against = team_matches[ga].sum()
    
    return {
        'season_points': points,
        'season_gf': goals_for,
        'season_ga': goals_against,
        'season_matches': len(team_matches)
    }


def calculate_h2h_extended(df, home_team, away_team, n_matches=5):
    """Calculate extended head-to-head statistics."""
    h2h = df[
        ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
        ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
    ].copy()
    
    if len(h2h) < n_matches:
        return {
            'h2h_home_wins': 0, 'h2h_away_wins': 0, 'h2h_draws': 0,
            'h2h_home_gf': 0, 'h2h_away_gf': 0, 'h2h_matches': 0
        }
    
    recent_h2h = h2h.tail(n_matches)
    
    home_wins = away_wins = draws = 0
    home_goals = away_goals = 0
    
    for _, match in recent_h2h.iterrows():
        if match['HomeTeam'] == home_team:
            home_goals += match['FTHG']
            away_goals += match['FTAG']
            if match['FTR'] == 'H':
                home_wins += 1
            elif match['FTR'] == 'A':
                away_wins += 1
            else:
                draws += 1
        else:
            home_goals += match['FTAG']
            away_goals += match['FTHG']
            if match['FTR'] == 'A':
                home_wins += 1
            elif match['FTR'] == 'H':
                away_wins += 1
            else:
                draws += 1
    
    return {
        'h2h_home_wins': home_wins,
        'h2h_away_wins': away_wins,
        'h2h_draws': draws,
        'h2h_home_gf': home_goals,
        'h2h_away_gf': away_goals,
        'h2h_matches': len(recent_h2h)
    }


def engineer_features(df):
    """Engineer comprehensive features."""
    print(f"\n{'='*60}")
    print("STEP 2: Engineering Enhanced Features")
    print(f"{'='*60}")
    print("\n⚠️  This will take 5-10 minutes for comprehensive features...")
    
    features_list = []
    
    for idx in range(len(df)):
        if idx % 500 == 0:
            print(f"  Processing match {idx}/{len(df)}...")
        
        current_match = df.iloc[idx]
        historical_df = df.iloc[:idx]
        
        if len(historical_df) < 20:
            continue
        
        home_team = current_match['HomeTeam']
        away_team = current_match['AwayTeam']
        current_date = current_match['Date']
        
        # Get stats for different windows
        home_last5 = calculate_extended_team_stats(historical_df, home_team, True, 5)
        home_last10 = calculate_extended_team_stats(historical_df, home_team, True, 10)
        away_last5 = calculate_extended_team_stats(historical_df, away_team, False, 5)
        away_last10 = calculate_extended_team_stats(historical_df, away_team, False, 10)
        
        # Season stats
        home_season = calculate_season_stats(historical_df, home_team, True, current_date)
        away_season = calculate_season_stats(historical_df, away_team, False, current_date)
        
        # H2H stats
        h2h = calculate_h2h_extended(historical_df, home_team, away_team, 5)
        
        # Create features
        features = {
            'Season': current_match['Season'],
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'FTR': current_match['FTR']
        }
        
        # Add all home team features
        for key, val in home_last5.items():
            features[f'home_{key}'] = val
        for key, val in home_last10.items():
            features[f'home_{key}'] = val
        for key, val in home_season.items():
            features[f'home_{key}'] = val
        
        # Add all away team features
        for key, val in away_last5.items():
            features[f'away_{key}'] = val
        for key, val in away_last10.items():
            features[f'away_{key}'] = val
        for key, val in away_season.items():
            features[f'away_{key}'] = val
        
        # Add H2H features
        for key, val in h2h.items():
            features[key] = val
        
        # Add difference features
        features['points_diff_last5'] = home_last5['last5_points'] - away_last5['last5_points']
        features['points_diff_last10'] = home_last10['last10_points'] - away_last10['last10_points']
        features['gd_diff_last5'] = home_last5['last5_gd'] - away_last5['last5_gd']
        features['gd_diff_last10'] = home_last10['last10_gd'] - away_last10['last10_gd']
        features['shot_acc_diff'] = home_last5['last5_shot_acc'] - away_last5['last5_shot_acc']
        features['form_diff'] = home_last5['last5_win_rate'] - away_last5['last5_win_rate']
        features['season_points_diff'] = home_season['season_points'] - away_season['season_points']
        
        features_list.append(features)
    
    df_features = pd.DataFrame(features_list)
    
    print(f"\n✓ Created {len(df_features)} matches with {len(df_features.columns)-1} features")
    print(f"✓ Step 2 completed successfully")
    
    return df_features


def encode_and_save(df, output_dir):
    """Encode categorical variables and save."""
    print(f"\n{'='*60}")
    print("STEP 3: Encoding and Saving")
    print(f"{'='*60}")
    
    # Encode FTR
    ftr_mapping = {'H': 2, 'D': 1, 'A': 0}
    df['FTR'] = df['FTR'].map(ftr_mapping)
    
    # Encode teams
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    all_teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
    le_home.fit(all_teams)
    le_away.fit(all_teams)
    
    df['HomeTeam_encoded'] = le_home.transform(df['HomeTeam'])
    df['AwayTeam_encoded'] = le_away.transform(df['AwayTeam'])
    df = df.drop(['HomeTeam', 'AwayTeam'], axis=1)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'enhanced_data.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved to: {output_path}")
    print(f"✓ Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"✓ Step 3 completed successfully")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ENHANCED PREPROCESSING FOR 60% ACCURACY TARGET")
    print("="*60)
    
    df = load_raw_data("../data/epl_matches.csv")
    df_features = engineer_features(df)
    encode_and_save(df_features, "output")
    
    print(f"\n{'='*60}")
    print("✓ ENHANCED PREPROCESSING COMPLETED!")
    print(f"{'='*60}")
    print("\nNext: Run advanced model training with CatBoost")
    print("="*60 + "\n")
