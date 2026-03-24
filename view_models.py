"""
Quick script to inspect trained model files
"""
import joblib
import os

print("\n" + "="*60)
print("INSPECTING TRAINED MODELS")
print("="*60)

model_dir = "pipeline/output"
model_files = [
    "model_logistic_regression.pkl",
    "model_decision_tree.pkl",
    "model_random_forest.pkl",
    "model_naive_bayes.pkl",
    "model_svm.pkl"
]

for i, model_file in enumerate(model_files, 1):
    model_path = os.path.join(model_dir, model_file)
    
    if not os.path.exists(model_path):
        print(f"\n{i}. {model_file}")
        print("   ❌ File not found")
        continue
    
    # Load the model
    model = joblib.load(model_path)
    
    # Get file size
    file_size = os.path.getsize(model_path)
    size_kb = file_size / 1024
    
    print(f"\n{i}. {model_file}")
    print(f"   📦 File Size: {size_kb:.2f} KB")
    print(f"   🤖 Model Type: {type(model).__name__}")
    print(f"   📊 Model Class: {model.__class__.__module__}.{model.__class__.__name__}")
    
    # Show model parameters
    print(f"   ⚙️  Parameters:")
    params = model.get_params()
    for key, value in list(params.items())[:5]:  # Show first 5 params
        print(f"      - {key}: {value}")
    if len(params) > 5:
        print(f"      ... and {len(params) - 5} more parameters")
    
    # Show model-specific info
    if hasattr(model, 'n_features_in_'):
        print(f"   📈 Features Used: {model.n_features_in_}")
    
    if hasattr(model, 'classes_'):
        print(f"   🎯 Classes: {list(model.classes_)}")
    
    if hasattr(model, 'n_estimators') and hasattr(model, 'estimators_'):
        print(f"   🌲 Number of Trees: {len(model.estimators_)}")
    
    if hasattr(model, 'coef_'):
        print(f"   📐 Coefficients Shape: {model.coef_.shape}")
    
    if hasattr(model, 'feature_importances_'):
        print(f"   ⭐ Has Feature Importances: Yes")

print("\n" + "="*60)
print("✓ INSPECTION COMPLETE")
print("="*60)

print("\n💡 TIP: To use a model for prediction:")
print("   import joblib")
print("   model = joblib.load('pipeline/output/model_random_forest.pkl')")
print("   predictions = model.predict(X_test)")
print()
