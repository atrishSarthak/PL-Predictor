# 🎯 Sample Inputs for EPL Match Prediction

## Understanding the Features

The model uses **8 features** to predict match outcomes:

| Feature | Description | Example |
|---------|-------------|---------|
| **GoalDifference** | FTHG - FTAG | 2 (Home won by 2) |
| **FTHG** | Full Time Home Goals | 3 |
| **FTAG** | Full Time Away Goals | 1 |
| **TotalGoals** | FTHG + FTAG | 4 |
| **HTR** | Half Time Result | 1 (Home winning) |
| **HTGoalDifference** | HTHG - HTAG | 1 |
| **HTHG** | Half Time Home Goals | 2 |
| **HTAG** | Half Time Away Goals | 1 |

### HTR Values:
- `1` = Home team winning at half time
- `0` = Draw at half time
- `-1` = Away team winning at half time

---

## 📋 Sample Input Scenarios

### Scenario 1: Clear Home Victory (3-0)
```
GoalDifference:    3
FTHG:              3
FTAG:              0
TotalGoals:        3
HTR:               1
HTGoalDifference:  2
HTHG:              2
HTAG:              0
```
**Expected Prediction:** Home Win (100% confidence)

---

### Scenario 2: Draw Match (1-1)
```
GoalDifference:    0
FTHG:              1
FTAG:              1
TotalGoals:        2
HTR:               0
HTGoalDifference:  0
HTHG:              0
HTAG:              0
```
**Expected Prediction:** Draw (81% confidence)

---

### Scenario 3: Away Team Victory (0-2)
```
GoalDifference:    -2
FTHG:              0
FTAG:              2
TotalGoals:        2
HTR:               -1
HTGoalDifference:  -1
HTHG:              0
HTAG:              1
```
**Expected Prediction:** Away Win (98% confidence)

---

### Scenario 4: High-Scoring Draw (3-3)
```
GoalDifference:    0
FTHG:              3
FTAG:              3
TotalGoals:        6
HTR:               1
HTGoalDifference:  1
HTHG:              2
HTAG:              1
```
**Expected Prediction:** Draw (83% confidence)

---

### Scenario 5: Narrow Home Win (2-1)
```
GoalDifference:    1
FTHG:              2
FTAG:              1
TotalGoals:        3
HTR:               0
HTGoalDifference:  0
HTHG:              1
HTAG:              1
```
**Expected Prediction:** Home Win (100% confidence)

---

### Scenario 6: Comeback Victory (2-1, was losing at HT)
```
GoalDifference:    1
FTHG:              2
FTAG:              1
TotalGoals:        3
HTR:               -1
HTGoalDifference:  -1
HTHG:              0
HTAG:              1
```
**Expected Prediction:** Home Win (100% confidence)

---

## 🖥️ How to Use These Inputs

### Method 1: Using the Interactive Script
```bash
python3 predict_match.py
```
Then enter the values when prompted:
```
Full Time Goals:
  Home Team Goals (FTHG): 3
  Away Team Goals (FTAG): 0

Half Time Goals:
  Home Team Goals (HTHG): 2
  Away Team Goals (HTAG): 0
```

---

### Method 2: Using Python Code
```python
import joblib
import pandas as pd

# Load model
model = joblib.load('pipeline/output/model_random_forest.pkl')

# Define input (Home team won 3-0)
features = {
    'GoalDifference': 3,
    'FTHG': 3,
    'FTAG': 0,
    'TotalGoals': 3,
    'HTR': 1,
    'HTGoalDifference': 2,
    'HTHG': 2,
    'HTAG': 0
}

# Make prediction
feature_names = ['GoalDifference', 'FTHG', 'FTAG', 'TotalGoals', 
                 'HTR', 'HTGoalDifference', 'HTHG', 'HTAG']
X = pd.DataFrame([[features[f] for f in feature_names]], columns=feature_names)

prediction = model.predict(X)[0]
probabilities = model.predict_proba(X)[0]

# Show results
outcome_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
print(f"Prediction: {outcome_map[prediction]}")
print(f"Confidence: {max(probabilities):.1%}")
```

---

### Method 3: Using the Dashboard
1. Open http://localhost:3000/predict
2. Enter values in the form:
   - GoalDifference: 3
   - FTHG: 3
   - FTAG: 0
   - TotalGoals: 3
   - HTR: 1
   - HTGoalDifference: 2
   - HTHG: 2
   - HTAG: 0
3. Click "Predict Match Outcome"

---

## 🎲 Try Your Own Scenarios!

### Famous EPL Matches to Test:

**Manchester City 6-1 Manchester United (2011)**
```
GoalDifference:    5
FTHG:              6
FTAG:              1
TotalGoals:        7
HTR:               1
HTGoalDifference:  1
HTHG:              1
HTAG:              0
```

**Liverpool 4-0 Barcelona (2019 - Champions League)**
```
GoalDifference:    4
FTHG:              4
FTAG:              0
TotalGoals:        4
HTR:               1
HTGoalDifference:  1
HTHG:              1
HTAG:              0
```

**Arsenal 7-3 Newcastle (2012)**
```
GoalDifference:    4
FTHG:              7
FTAG:              3
TotalGoals:        10
HTR:               1
HTGoalDifference:  2
HTHG:              4
HTAG:              2
```

---

## 📊 Understanding the Output

### Prediction Format:
```
🔮 PREDICTION: Home Win

📈 Confidence Scores:
   Away Win: 0.0%
   Draw:     0.0%
   Home Win: 100.0%

✨ Confidence Level: Very High (100.0%)
```

### Confidence Levels:
- **Very High** (>90%): Model is extremely confident
- **High** (70-90%): Model is confident
- **Moderate** (<70%): Model is less certain

---

## ⚠️ Important Note

**Data Leakage Warning:**
The current model uses **GoalDifference** which is calculated from the final score. This means:
- ✅ Perfect for understanding how the model works
- ❌ Not suitable for real pre-match predictions
- 💡 In production, you'd use only pre-match features like:
  - Team form (last 5 matches)
  - Head-to-head history
  - Home/away records
  - Player availability
  - League position

---

## 🚀 Quick Test Commands

```bash
# Run sample predictions (5 scenarios)
python3 sample_prediction.py

# Interactive prediction
python3 predict_match.py

# View model details
python3 view_models.py

# Test a specific scenario
python3 test_prediction.py
```

---

**Now you have everything you need to test the prediction system! 🎉**
