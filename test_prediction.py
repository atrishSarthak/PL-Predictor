"""
Quick test to make predictions with a trained model
"""
import joblib
import pandas as pd
import numpy as np

print("\n" + "="*60)
print("TESTING MODEL PREDICTIONS")
print("="*60)

# Load a model (let's use Random Forest - usually the best)
print("\n📥 Loading Random Forest model...")
model = joblib.load('pipeline/output/model_random_forest.pkl')
print("✓ Model loaded successfully")

# Load test data
print("\n📥 Loading test data...")
X_test = pd.read_csv('pipeline/output/X_test.csv')
y_test = pd.read_csv('pipeline/output/y_test.csv')
print(f"✓ Test data loaded: {len(X_test)} samples")

# Show first 5 test samples
print("\n📊 First 5 Test Samples:")
print(X_test.head())

# Make predictions on first 10 samples
print("\n🔮 Making Predictions on First 10 Samples:")
print("="*60)

sample_X = X_test.head(10)
sample_y = y_test.head(10)

predictions = model.predict(sample_X)
probabilities = model.predict_proba(sample_X)

outcome_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}

for i in range(10):
    actual = sample_y.iloc[i, 0]
    predicted = predictions[i]
    probs = probabilities[i]
    
    correct = "✓" if actual == predicted else "✗"
    
    print(f"\nSample {i+1}:")
    print(f"  Actual:    {actual} ({outcome_map[actual]})")
    print(f"  Predicted: {predicted} ({outcome_map[predicted]}) {correct}")
    print(f"  Confidence:")
    print(f"    Away Win: {probs[0]:.1%}")
    print(f"    Draw:     {probs[1]:.1%}")
    print(f"    Home Win: {probs[2]:.1%}")

# Calculate accuracy on these 10 samples
accuracy = (predictions == sample_y.values.flatten()).sum() / len(predictions)
print(f"\n📈 Accuracy on these 10 samples: {accuracy:.1%}")

print("\n" + "="*60)
print("✓ PREDICTION TEST COMPLETE")
print("="*60)
print("\n💡 The model is working! Phase 6 will evaluate all models on the full test set.")
print()
