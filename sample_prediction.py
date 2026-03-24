"""
Sample Prediction Script
Shows how to make predictions with different match scenarios
"""
import joblib
import pandas as pd
import numpy as np

print("\n" + "="*70)
print("EPL MATCH OUTCOME PREDICTION - SAMPLE SCENARIOS")
print("="*70)

# Load the best model (Random Forest)
print("\n📥 Loading Random Forest model...")
model = joblib.load('pipeline/output/model_random_forest.pkl')
print("✓ Model loaded successfully")

# Define outcome mapping
outcome_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}

print("\n" + "="*70)
print("SAMPLE SCENARIOS")
print("="*70)

# Sample scenarios with explanations
scenarios = [
    {
        "name": "Scenario 1: Home Team Dominance",
        "description": "Home team scored 3, Away team scored 0 (3-0)",
        "features": {
            "GoalDifference": 3,      # Home won by 3 goals
            "FTHG": 3,                 # Full Time Home Goals
            "FTAG": 0,                 # Full Time Away Goals
            "TotalGoals": 3,           # Total goals in match
            "HTR": 1,                  # Half Time Result: Home winning
            "HTGoalDifference": 2,     # Half time: Home leading by 2
            "HTHG": 2,                 # Half Time Home Goals
            "HTAG": 0                  # Half Time Away Goals
        }
    },
    {
        "name": "Scenario 2: Close Match - Draw",
        "description": "Both teams scored 1 goal each (1-1)",
        "features": {
            "GoalDifference": 0,       # Equal goals
            "FTHG": 1,                 # Full Time Home Goals
            "FTAG": 1,                 # Full Time Away Goals
            "TotalGoals": 2,           # Total goals
            "HTR": 0,                  # Half Time Result: Draw
            "HTGoalDifference": 0,     # Half time: Equal
            "HTHG": 0,                 # Half Time Home Goals
            "HTAG": 0                  # Half Time Away Goals
        }
    },
    {
        "name": "Scenario 3: Away Team Victory",
        "description": "Home team scored 0, Away team scored 2 (0-2)",
        "features": {
            "GoalDifference": -2,      # Away won by 2 goals
            "FTHG": 0,                 # Full Time Home Goals
            "FTAG": 2,                 # Full Time Away Goals
            "TotalGoals": 2,           # Total goals
            "HTR": -1,                 # Half Time Result: Away winning
            "HTGoalDifference": -1,    # Half time: Away leading by 1
            "HTHG": 0,                 # Half Time Home Goals
            "HTAG": 1                  # Half Time Away Goals
        }
    },
    {
        "name": "Scenario 4: High-Scoring Draw",
        "description": "Both teams scored 3 goals each (3-3)",
        "features": {
            "GoalDifference": 0,       # Equal goals
            "FTHG": 3,                 # Full Time Home Goals
            "FTAG": 3,                 # Full Time Away Goals
            "TotalGoals": 6,           # Total goals (high scoring!)
            "HTR": 1,                  # Half Time Result: Home was winning
            "HTGoalDifference": 1,     # Half time: Home leading by 1
            "HTHG": 2,                 # Half Time Home Goals
            "HTAG": 1                  # Half Time Away Goals
        }
    },
    {
        "name": "Scenario 5: Comeback Victory",
        "description": "Home team won 2-1 (was losing at half time)",
        "features": {
            "GoalDifference": 1,       # Home won by 1 goal
            "FTHG": 2,                 # Full Time Home Goals
            "FTAG": 1,                 # Full Time Away Goals
            "TotalGoals": 3,           # Total goals
            "HTR": -1,                 # Half Time Result: Away was winning!
            "HTGoalDifference": -1,    # Half time: Away leading by 1
            "HTHG": 0,                 # Half Time Home Goals
            "HTAG": 1                  # Half Time Away Goals
        }
    }
]

# Make predictions for each scenario
for i, scenario in enumerate(scenarios, 1):
    print(f"\n{'='*70}")
    print(f"{scenario['name']}")
    print(f"{'='*70}")
    print(f"📝 Description: {scenario['description']}")
    
    # Create feature vector
    features = scenario['features']
    feature_names = ['GoalDifference', 'FTHG', 'FTAG', 'TotalGoals', 
                     'HTR', 'HTGoalDifference', 'HTHG', 'HTAG']
    
    # Create DataFrame with features in correct order
    X = pd.DataFrame([[features[f] for f in feature_names]], columns=feature_names)
    
    print(f"\n📊 Input Features:")
    for fname, fval in features.items():
        print(f"   {fname:20s}: {fval}")
    
    # Make prediction
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    print(f"\n🔮 PREDICTION: {outcome_map[prediction]}")
    print(f"\n📈 Confidence Scores:")
    print(f"   Away Win: {probabilities[0]:.1%}")
    print(f"   Draw:     {probabilities[1]:.1%}")
    print(f"   Home Win: {probabilities[2]:.1%}")
    
    # Show the most confident prediction
    max_prob = max(probabilities)
    confidence_level = "Very High" if max_prob > 0.9 else "High" if max_prob > 0.7 else "Moderate"
    print(f"\n✨ Confidence Level: {confidence_level} ({max_prob:.1%})")

print("\n" + "="*70)
print("✓ ALL PREDICTIONS COMPLETE")
print("="*70)

print("\n💡 HOW TO USE THIS IN YOUR CODE:")
print("""
import joblib
import pandas as pd

# Load model
model = joblib.load('pipeline/output/model_random_forest.pkl')

# Create input (example: Home team won 2-1)
features = {
    'GoalDifference': 1,
    'FTHG': 2,
    'FTAG': 1,
    'TotalGoals': 3,
    'HTR': 0,
    'HTGoalDifference': 0,
    'HTHG': 1,
    'HTAG': 1
}

# Make prediction
X = pd.DataFrame([features])
prediction = model.predict(X)[0]
probabilities = model.predict_proba(X)[0]

print(f"Prediction: {prediction}")  # 0=Away Win, 1=Draw, 2=Home Win
print(f"Probabilities: {probabilities}")
""")

print("\n" + "="*70)
print()
