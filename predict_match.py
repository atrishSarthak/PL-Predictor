"""
EPL Match Outcome Predictor - Using Best CatBoost Model
Predicts match outcomes using historical team statistics
"""
import joblib
import pandas as pd
import numpy as np

print("\n" + "="*70)
print("⚽ EPL MATCH OUTCOME PREDICTOR - CatBoost Model (53.31% Accuracy) ⚽")
print("="*70)

# Load best model
print("\n📥 Loading CatBoost Optimized model...")
try:
    model = joblib.load('pipeline/output/catboost_optimized.pkl')
    print("✓ CatBoost model loaded successfully (Best: 53.31% accuracy)")
except:
    print("⚠️  CatBoost model not found, loading Random Forest...")
    model = joblib.load('pipeline/output/random_forest_optimized.pkl')
print()

# Load feature names
try:
    with open('pipeline/output/selected_features_enhanced.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    print(f"✓ Loaded {len(feature_names)} features\n")
except:
    print("⚠️  Feature list not found, using default features\n")
    feature_names = None

# Outcome mapping
outcome_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}

print("="*70)
print("MATCH PREDICTION - ENTER TEAM STATISTICS")
print("="*70)
print("\nNote: This predictor uses historical team statistics")
print("Enter approximate values based on recent team performance\n")

try:
    # Get team information
    print("Teams:")
    home_team = input("  Home Team Name: ").strip()
    away_team = input("  Away Team Name: ").strip()
    
    print(f"\n{'='*70}")
    print(f"MATCH: {home_team} (Home) vs {away_team} (Away)")
    print(f"{'='*70}\n")
    
    # Simplified feature input (key features only)
    print("Enter recent team statistics (approximate values):\n")
    
    print("Season Performance:")
    home_season_points = int(input(f"  {home_team} season points (0-100): "))
    away_season_points = int(input(f"  {away_team} season points (0-100): "))
    
    print("\nLast 5 Matches Form:")
    home_last5_points = int(input(f"  {home_team} points in last 5 (0-15): "))
    away_last5_points = int(input(f"  {away_team} points in last 5 (0-15): "))
    
    print("\nLast 10 Matches Form:")
    home_last10_points = int(input(f"  {home_team} points in last 10 (0-30): "))
    away_last10_points = int(input(f"  {away_team} points in last 10 (0-30): "))
    
    print("\nGoal Statistics (Last 10 matches):")
    home_last10_gf = int(input(f"  {home_team} goals scored (0-40): "))
    home_last10_ga = int(input(f"  {home_team} goals conceded (0-40): "))
    away_last10_gf = int(input(f"  {away_team} goals scored (0-40): "))
    away_last10_ga = int(input(f"  {away_team} goals conceded (0-40): "))
    
    print("\nShot Statistics (Last 10 matches):")
    home_last10_shots = int(input(f"  {home_team} total shots (0-200): "))
    away_last10_shots = int(input(f"  {away_team} total shots (0-200): "))
    
    # Calculate derived features
    season_points_diff = home_season_points - away_season_points
    points_diff_last5 = home_last5_points - away_last5_points
    points_diff_last10 = home_last10_points - away_last10_points
    home_last10_gd = home_last10_gf - home_last10_ga
    away_last10_gd = away_last10_gf - away_last10_ga
    gd_diff_last10 = home_last10_gd - away_last10_gd
    
    # Create feature vector with default values for missing features
    if feature_names:
        # Create full feature vector
        features = {}
        
        # Set key features
        features['season_points_diff'] = season_points_diff
        features['points_diff_last5'] = points_diff_last5
        features['points_diff_last10'] = points_diff_last10
        features['gd_diff_last10'] = gd_diff_last10
        features['home_last10_shots'] = home_last10_shots
        features['away_last10_shots'] = away_last10_shots
        features['home_last10_gf'] = home_last10_gf
        features['home_last10_ga'] = home_last10_ga
        features['away_last10_gf'] = away_last10_gf
        features['away_last10_ga'] = away_last10_ga
        features['home_last10_gd'] = home_last10_gd
        features['away_last10_gd'] = away_last10_gd
        features['home_last5_points'] = home_last5_points
        features['away_last5_points'] = away_last5_points
        features['home_last10_points'] = home_last10_points
        features['away_last10_points'] = away_last10_points
        
        # Fill remaining features with reasonable defaults
        for fname in feature_names:
            if fname not in features:
                if 'encoded' in fname.lower():
                    features[fname] = 10  # Default team encoding
                elif 'shot_acc' in fname.lower():
                    features[fname] = 35.0  # Default shot accuracy
                elif 'goals_per_shot' in fname.lower():
                    features[fname] = 0.12  # Default conversion rate
                elif 'fouls' in fname.lower():
                    features[fname] = 12  # Default fouls
                elif 'yellows' in fname.lower():
                    features[fname] = 2  # Default yellow cards
                elif 'win_rate' in fname.lower():
                    features[fname] = 0.4  # Default win rate
                elif 'ppg' in fname.lower():
                    features[fname] = 1.5  # Default points per game
                elif 'h2h' in fname.lower():
                    features[fname] = 1  # Default H2H
                else:
                    features[fname] = 0  # Default zero
        
        # Create DataFrame with correct feature order
        X = pd.DataFrame([[features[f] for f in feature_names]], columns=feature_names)
    else:
        # Fallback to simple features
        X = pd.DataFrame([[
            season_points_diff, points_diff_last5, points_diff_last10,
            gd_diff_last10, home_last10_shots, away_last10_shots
        ]], columns=['season_points_diff', 'points_diff_last5', 'points_diff_last10',
                     'gd_diff_last10', 'home_last10_shots', 'away_last10_shots'])
    
    # Make prediction
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    print("\n" + "="*70)
    print("📊 MATCH ANALYSIS")
    print("="*70)
    print(f"\nSeason Points: {home_team} {home_season_points} - {away_season_points} {away_team}")
    print(f"Points Difference: {season_points_diff:+d} (favors {home_team if season_points_diff > 0 else away_team})")
    print(f"\nLast 5 Form: {home_team} {home_last5_points} - {away_last5_points} {away_team}")
    print(f"Last 10 Form: {home_team} {home_last10_points} - {away_last10_points} {away_team}")
    print(f"\nGoal Difference (Last 10): {home_team} {home_last10_gd:+d}, {away_team} {away_last10_gd:+d}")
    
    print("\n" + "="*70)
    print("🔮 PREDICTION RESULTS")
    print("="*70)
    
    print(f"\n🏆 PREDICTED OUTCOME: {outcome_map[prediction]}")
    
    print(f"\n📊 Confidence Scores:")
    print(f"   {'Away Win':<15} {'█' * int(probabilities[0] * 50)} {probabilities[0]:.1%}")
    print(f"   {'Draw':<15} {'█' * int(probabilities[1] * 50)} {probabilities[1]:.1%}")
    print(f"   {'Home Win':<15} {'█' * int(probabilities[2] * 50)} {probabilities[2]:.1%}")
    
    max_prob = max(probabilities)
    if max_prob > 0.6:
        confidence = "High"
        emoji = "🔥"
    elif max_prob > 0.45:
        confidence = "Moderate"
        emoji = "✨"
    else:
        confidence = "Low"
        emoji = "💫"
    
    print(f"\n{emoji} Confidence Level: {confidence} ({max_prob:.1%})")
    
    # Analysis
    print(f"\n📈 Key Factors:")
    if abs(season_points_diff) > 15:
        print(f"   • Strong season performance advantage: {home_team if season_points_diff > 0 else away_team}")
    if abs(points_diff_last5) > 6:
        print(f"   • Better recent form: {home_team if points_diff_last5 > 0 else away_team}")
    if abs(gd_diff_last10) > 5:
        print(f"   • Superior goal difference: {home_team if gd_diff_last10 > 0 else away_team}")
    if home_last10_shots > away_last10_shots * 1.2:
        print(f"   • More attacking play: {home_team}")
    elif away_last10_shots > home_last10_shots * 1.2:
        print(f"   • More attacking play: {away_team}")
    
    print("\n" + "="*70)
    print("Model: CatBoost Optimized | Accuracy: 53.31% | Features: 50")
    print("="*70)

except ValueError as e:
    print(f"\n❌ Error: Please enter valid numbers - {e}")
except KeyboardInterrupt:
    print("\n\n👋 Prediction cancelled")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print()
