"""
Interactive Match Prediction Script
Enter match details and get instant predictions!
"""
import joblib
import pandas as pd

print("\n" + "="*70)
print("⚽ EPL MATCH OUTCOME PREDICTOR ⚽")
print("="*70)

# Load model
print("\n📥 Loading Random Forest model...")
model = joblib.load('pipeline/output/model_random_forest.pkl')
print("✓ Model loaded successfully\n")

# Outcome mapping
outcome_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}

print("="*70)
print("ENTER MATCH DETAILS")
print("="*70)
print("\nNote: Enter the actual match statistics (goals scored)")
print("This is a demonstration - in real prediction, you'd use pre-match features\n")

try:
    # Get user input
    print("Full Time Goals:")
    fthg = int(input("  Home Team Goals (FTHG): "))
    ftag = int(input("  Away Team Goals (FTAG): "))
    
    print("\nHalf Time Goals:")
    hthg = int(input("  Home Team Goals (HTHG): "))
    htag = int(input("  Away Team Goals (HTAG): "))
    
    # Calculate derived features
    goal_diff = fthg - ftag
    total_goals = fthg + ftag
    ht_goal_diff = hthg - htag
    
    # Determine HTR
    if hthg > htag:
        htr = 1  # Home winning at half time
    elif hthg < htag:
        htr = -1  # Away winning at half time
    else:
        htr = 0  # Draw at half time
    
    # Create feature dictionary
    features = {
        'GoalDifference': goal_diff,
        'FTHG': fthg,
        'FTAG': ftag,
        'TotalGoals': total_goals,
        'HTR': htr,
        'HTGoalDifference': ht_goal_diff,
        'HTHG': hthg,
        'HTAG': htag
    }
    
    print("\n" + "="*70)
    print("MATCH SUMMARY")
    print("="*70)
    print(f"Full Time Score: {fthg} - {ftag}")
    print(f"Half Time Score: {hthg} - {htag}")
    print(f"Total Goals: {total_goals}")
    print(f"Goal Difference: {goal_diff:+d}")
    
    # Make prediction
    feature_names = ['GoalDifference', 'FTHG', 'FTAG', 'TotalGoals', 
                     'HTR', 'HTGoalDifference', 'HTHG', 'HTAG']
    X = pd.DataFrame([[features[f] for f in feature_names]], columns=feature_names)
    
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    print("\n" + "="*70)
    print("🔮 PREDICTION RESULTS")
    print("="*70)
    
    print(f"\n🏆 PREDICTED OUTCOME: {outcome_map[prediction]}")
    
    print(f"\n📊 Confidence Scores:")
    print(f"   {'Away Win':<15} {'█' * int(probabilities[0] * 50)} {probabilities[0]:.1%}")
    print(f"   {'Draw':<15} {'█' * int(probabilities[1] * 50)} {probabilities[1]:.1%}")
    print(f"   {'Home Win':<15} {'█' * int(probabilities[2] * 50)} {probabilities[2]:.1%}")
    
    max_prob = max(probabilities)
    if max_prob > 0.9:
        confidence = "Very High"
        emoji = "🔥"
    elif max_prob > 0.7:
        confidence = "High"
        emoji = "✨"
    else:
        confidence = "Moderate"
        emoji = "💫"
    
    print(f"\n{emoji} Confidence Level: {confidence} ({max_prob:.1%})")
    
    # Actual outcome based on goals
    if fthg > ftag:
        actual = "Home Win"
    elif fthg < ftag:
        actual = "Away Win"
    else:
        actual = "Draw"
    
    print(f"\n✅ Actual Outcome: {actual}")
    
    if outcome_map[prediction] == actual:
        print("🎯 PREDICTION CORRECT! ✓")
    else:
        print("❌ PREDICTION INCORRECT")
    
    print("\n" + "="*70)

except ValueError:
    print("\n❌ Error: Please enter valid numbers")
except KeyboardInterrupt:
    print("\n\n👋 Prediction cancelled")
except Exception as e:
    print(f"\n❌ Error: {e}")

print()
