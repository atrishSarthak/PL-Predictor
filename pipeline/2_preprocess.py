"""
Phase 2: Data Cleaning and Preprocessing
This script cleans and preprocesses the EPL match data for machine learning.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_raw_data(filepath):
    """Load the raw EPL matches CSV file."""
    print(f"\n{'='*60}")
    print("STEP 1: Loading Raw Data")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        print("Please ensure the CSV file is placed in the data/ folder.")
        sys.exit(1)
    
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Successfully loaded data from {filepath}")
        print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"\nColumn Names: {list(df.columns)}")
        
        missing_counts = df.isnull().sum()
        print(f"\nMissing Values Per Column:")
        if missing_counts.sum() > 0:
            print(missing_counts[missing_counts > 0])
        else:
            print("  No missing values found")
        
        print(f"\n✓ Step 1 completed successfully")
        return df
    except Exception as e:
        print(f"ERROR: Failed to load CSV file: {e}")
        sys.exit(1)


def select_relevant_columns(df):
    """Select only relevant columns for analysis."""
    print(f"\n{'='*60}")
    print("STEP 2: Selecting Relevant Columns")
    print(f"{'='*60}")
    
    # Map old column names to new ones if they exist
    column_mapping = {
        'MatchDate': 'Date',
        'FullTimeHomeGoals': 'FTHG',
        'FullTimeAwayGoals': 'FTAG',
        'HalfTimeHomeGoals': 'HTHG',
        'HalfTimeAwayGoals': 'HTAG',
        'HalfTimeResult': 'HTR',
        'FullTimeResult': 'FTR'
    }
    
    # Rename columns if they exist
    df = df.rename(columns=column_mapping)
    
    # Define required columns (after renaming)
    required_columns = [
        'Date', 'Season', 'HomeTeam', 'AwayTeam', 
        'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HTR', 'FTR'
    ]
    
    # Check which columns exist
    existing_columns = [col for col in required_columns if col in df.columns]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    print(f"\nColumns Found ({len(existing_columns)}):")
    for col in existing_columns:
        print(f"  ✓ {col}")
    
    if missing_columns:
        print(f"\n⚠ WARNING: Missing Columns ({len(missing_columns)}):")
        for col in missing_columns:
            print(f"  ✗ {col}")
    
    # Keep only existing columns
    df_filtered = df[existing_columns].copy()
    
    print(f"\nFinal Column List: {list(df_filtered.columns)}")
    print(f"\n✓ Step 2 completed successfully")
    
    return df_filtered, existing_columns


def handle_missing_values(df, existing_columns):
    """Handle missing values in the dataset."""
    print(f"\n{'='*60}")
    print("STEP 3: Handling Missing Values")
    print(f"{'='*60}")
    
    rows_before = len(df)
    print(f"\nRows Before Cleaning: {rows_before}")
    
    # Numeric columns - fill with median
    numeric_cols = ['FTHG', 'FTAG', 'HTHG', 'HTAG']
    numeric_cols_present = [col for col in numeric_cols if col in existing_columns]
    
    if numeric_cols_present:
        print(f"\nFilling missing values in numeric columns with median:")
        for col in numeric_cols_present:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  {col}: {missing_count} missing values filled with {median_val}")
            else:
                print(f"  {col}: No missing values")
    
    # Categorical columns - drop rows with missing values
    categorical_cols = ['HomeTeam', 'AwayTeam', 'HTR', 'FTR']
    categorical_cols_present = [col for col in categorical_cols if col in existing_columns]
    
    if categorical_cols_present:
        print(f"\nDropping rows with missing categorical values:")
        for col in categorical_cols_present:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                print(f"  {col}: {missing_count} missing values")
        
        df.dropna(subset=categorical_cols_present, inplace=True)
    
    rows_after = len(df)
    rows_dropped = rows_before - rows_after
    
    print(f"\nRows After Cleaning: {rows_after}")
    print(f"Rows Dropped: {rows_dropped}")
    
    print(f"\n✓ Step 3 completed successfully")
    
    return df


def standardize_team_names(df, existing_columns):
    """Standardize team names using mapping dictionary."""
    print(f"\n{'='*60}")
    print("STEP 4: Standardizing Team Names")
    print(f"{'='*60}")
    
    if 'HomeTeam' not in existing_columns:
        print("\n⚠ WARNING: HomeTeam column not found. Skipping team name standardization.")
        return df
    
    # Print unique team names BEFORE cleaning
    print(f"\nUnique HomeTeam Values BEFORE Cleaning ({df['HomeTeam'].nunique()}):")
    unique_teams_before = sorted(df['HomeTeam'].unique())
    for i, team in enumerate(unique_teams_before, 1):
        print(f"  {i}. {team}")
    
    # Team name mapping dictionary
    team_mapping = {
        'Man United': 'Manchester United',
        'Man City': 'Manchester City',
        'Newcastle': 'Newcastle United',
        'Tottenham': 'Tottenham Hotspur',
        'West Ham': 'West Ham United',
        'Wolves': 'Wolverhampton Wanderers',
        'Brighton': 'Brighton & Hove Albion',
        'Leicester': 'Leicester City',
        'Norwich': 'Norwich City',
        'Sheffield United': 'Sheffield Utd',
        'West Brom': 'West Bromwich Albion',
        'Birmingham': 'Birmingham City',
        'Bolton': 'Bolton Wanderers',
        'Blackburn': 'Blackburn Rovers',
        'QPR': 'Queens Park Rangers',
        'Stoke': 'Stoke City',
        'Swansea': 'Swansea City',
        'Hull': 'Hull City',
        'Cardiff': 'Cardiff City',
        'Wigan': 'Wigan Athletic',
        'Nott\'m Forest': 'Nottingham Forest',
        'Luton': 'Luton Town',
        'Ipswich': 'Ipswich Town'
    }
    
    # Apply mapping to HomeTeam (safe replace - won't fail if value not present)
    df['HomeTeam'] = df['HomeTeam'].replace(team_mapping)
    
    # Apply same mapping to AwayTeam if it exists
    if 'AwayTeam' in existing_columns:
        df['AwayTeam'] = df['AwayTeam'].replace(team_mapping)
        print(f"\n✓ Applied team name mapping to both HomeTeam and AwayTeam")
    else:
        print(f"\n✓ Applied team name mapping to HomeTeam only")
    
    # Print unique team names AFTER cleaning
    print(f"\nUnique HomeTeam Values AFTER Cleaning ({df['HomeTeam'].nunique()}):")
    unique_teams_after = sorted(df['HomeTeam'].unique())
    for i, team in enumerate(unique_teams_after, 1):
        print(f"  {i}. {team}")
    
    print(f"\n✓ Step 4 completed successfully")
    
    return df


def encode_categorical_columns(df, existing_columns):
    """Encode categorical columns to numeric values."""
    print(f"\n{'='*60}")
    print("STEP 5: Encoding Categorical Columns")
    print(f"{'='*60}")
    
    # Encode HTR: H=1, D=0, A=-1
    if 'HTR' in existing_columns:
        htr_mapping = {'H': 1, 'D': 0, 'A': -1}
        df['HTR'] = df['HTR'].map(htr_mapping)
        print(f"\n✓ Encoded HTR: H=1, D=0, A=-1")
        print(f"  Value counts: {df['HTR'].value_counts().to_dict()}")
    else:
        print(f"\n⚠ WARNING: HTR column not found. Skipping HTR encoding.")
    
    # Encode FTR: H=2, D=1, A=0 (TARGET VARIABLE)
    if 'FTR' in existing_columns:
        ftr_mapping = {'H': 2, 'D': 1, 'A': 0}
        df['FTR'] = df['FTR'].map(ftr_mapping)
        print(f"\n✓ Encoded FTR (TARGET): H=2, D=1, A=0")
        print(f"  Value counts after encoding:")
        value_counts = df['FTR'].value_counts().sort_index()
        for val, count in value_counts.items():
            outcome = {2: 'Home Win', 1: 'Draw', 0: 'Away Win'}.get(val, 'Unknown')
            print(f"    {val} ({outcome}): {count}")
    else:
        print(f"\n⚠ WARNING: FTR column not found. Skipping FTR encoding.")
    
    print(f"\n✓ Step 5 completed successfully")
    
    return df


def engineer_features(df, existing_columns):
    """Create new features from existing columns."""
    print(f"\n{'='*60}")
    print("STEP 6: Feature Engineering")
    print(f"{'='*60}")
    
    features_created = []
    
    # GoalDifference = FTHG - FTAG
    if 'FTHG' in existing_columns and 'FTAG' in existing_columns:
        df['GoalDifference'] = df['FTHG'] - df['FTAG']
        features_created.append('GoalDifference')
        print(f"\n✓ Created GoalDifference = FTHG - FTAG")
    else:
        print(f"\n⚠ WARNING: Cannot create GoalDifference (FTHG or FTAG missing)")
    
    # HTGoalDifference = HTHG - HTAG
    if 'HTHG' in existing_columns and 'HTAG' in existing_columns:
        df['HTGoalDifference'] = df['HTHG'] - df['HTAG']
        features_created.append('HTGoalDifference')
        print(f"✓ Created HTGoalDifference = HTHG - HTAG")
    else:
        print(f"⚠ WARNING: Cannot create HTGoalDifference (HTHG or HTAG missing)")
    
    # TotalGoals = FTHG + FTAG
    if 'FTHG' in existing_columns and 'FTAG' in existing_columns:
        df['TotalGoals'] = df['FTHG'] + df['FTAG']
        features_created.append('TotalGoals')
        print(f"✓ Created TotalGoals = FTHG + FTAG")
    else:
        print(f"⚠ WARNING: Cannot create TotalGoals (FTHG or FTAG missing)")
    
    if features_created:
        print(f"\nFirst 5 Rows of New Features:")
        print(df[features_created].head())
    else:
        print(f"\n⚠ WARNING: No new features were created")
    
    print(f"\n✓ Step 6 completed successfully")
    
    return df


def scale_features(df, existing_columns):
    """Apply StandardScaler to numeric features."""
    print(f"\n{'='*60}")
    print("STEP 7: Feature Scaling")
    print(f"{'='*60}")
    
    # IMPORTANT NOTE about data leakage
    print(f"\n⚠ NOTE: Scaling is done before train-test split for simplicity in this phase.")
    print(f"  In production, scaling MUST be applied AFTER splitting to avoid data leakage.")
    print(f"  The scaler should be fit on training data only, then applied to test data.")
    
    # Define columns to scale (DO NOT scale FTR - it's the target)
    scale_candidates = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 
                       'GoalDifference', 'HTGoalDifference', 'TotalGoals']
    
    # Only scale columns that exist
    columns_to_scale = [col for col in scale_candidates if col in df.columns]
    
    if not columns_to_scale:
        print(f"\n⚠ WARNING: No numeric columns found for scaling. Skipping scaling step.")
        print(f"\n✓ Step 7 completed (no scaling applied)")
        return df
    
    print(f"\nColumns to Scale: {columns_to_scale}")
    
    # Apply StandardScaler
    scaler = StandardScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    
    print(f"\n✓ Scaling Applied Successfully")
    print(f"\nScaled Features Statistics (should be ~0 mean, ~1 std):")
    for col in columns_to_scale:
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"  {col}:")
        print(f"    Mean: {mean_val:.6f} (should be ~0)")
        print(f"    Std:  {std_val:.6f} (should be ~1)")
    
    print(f"\n✓ Step 7 completed successfully")
    
    return df


def save_cleaned_data(df, output_dir, output_filename):
    """Save the cleaned dataset to CSV."""
    print(f"\n{'='*60}")
    print("STEP 8: Saving Cleaned Dataset")
    print(f"{'='*60}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Cleaned dataset saved to: {output_path}")
    print(f"\nFinal Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nFinal Column List:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\n✓ Step 8 completed successfully")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("EPL MATCH DATA PREPROCESSING")
    print("="*60)
    
    # Define paths
    input_path = "data/epl_matches.csv"
    output_dir = "pipeline/output"
    output_filename = "cleaned_data.csv"
    
    # Step 1: Load raw data
    df = load_raw_data(input_path)
    
    # Step 2: Select relevant columns
    df, existing_columns = select_relevant_columns(df)
    
    # Step 3: Handle missing values
    df = handle_missing_values(df, existing_columns)
    
    # Step 4: Standardize team names
    df = standardize_team_names(df, existing_columns)
    
    # Step 5: Encode categorical columns
    df = encode_categorical_columns(df, existing_columns)
    
    # Step 6: Engineer features
    df = engineer_features(df, existing_columns)
    
    # Step 7: Scale features
    df = scale_features(df, existing_columns)
    
    # Step 8: Save cleaned data
    save_cleaned_data(df, output_dir, output_filename)
    
    print(f"\n{'='*60}")
    print("✓ ALL PREPROCESSING STEPS COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"\nOutput File: {output_dir}/{output_filename}")
    print(f"\nYou can now proceed to Phase 3: Handling Class Imbalance")
    print("="*60 + "\n")
