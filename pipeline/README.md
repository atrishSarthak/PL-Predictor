# EPL Match Prediction Pipeline

This folder contains the complete machine learning pipeline for predicting EPL match outcomes.

## 🏆 Best Results: 52.67% Accuracy with CatBoost

## Pipeline Files (In Order)

### 1. Data Exploration
```bash
python3 1_explore.py
```
- Explores the raw EPL match data
- Generates visualization charts
- Output: Various PNG charts in `output/`

### 2. Data Preprocessing

**Option A: Enhanced Preprocessing (RECOMMENDED - for CatBoost)**
```bash
python3 2_preprocess_enhanced.py
```
- Creates 93 advanced features
- Last 5 and 10 matches statistics
- Season-to-date stats
- Extended head-to-head records
- Output: `output/enhanced_data.csv`
- Time: ~5-10 minutes

**Option B: Fixed Preprocessing (for baseline models)**
```bash
python3 2_preprocess_fixed.py
```
- Creates basic features without data leakage
- Simpler feature set
- Output: `output/cleaned_data.csv`
- Time: ~1-2 minutes

### 3. Handle Class Imbalance
```bash
python3 3_imbalance.py
```
- Applies SMOTE to balance classes
- Only needed for baseline models
- Output: `output/balanced_data.csv`

### 4. Feature Selection
```bash
python3 4_feature_selection.py
```
- Selects top features using Random Forest
- Only needed for baseline models
- Output: `output/feature_selected_data.csv`, `output/selected_features.txt`

### 5. Model Training

**Option A: CatBoost Models (RECOMMENDED - Best Performance)**
```bash
python3 5_models_catboost.py
```
- Trains CatBoost, Random Forest, Gradient Boosting
- Uses enhanced features (93 features)
- Output: `catboost.pkl` (52.67% accuracy), `random_forest.pkl`, `gradient_boosting.pkl`
- Time: ~2-3 minutes

**Option B: Ensemble with Tuning (Alternative)**
```bash
python3 5_models_ensemble.py
```
- Trains tuned models and ensemble
- Hyperparameter optimization
- Output: `catboost_tuned.pkl`, `ensemble_model.pkl`, `rf_tuned.pkl`, `gb_tuned.pkl`
- Time: ~3-5 minutes

### 6. Model Evaluation
```bash
python3 6_evaluate_catboost.py
```
- Evaluates all CatBoost and enhanced models
- Generates comparison charts and confusion matrices
- Output: `catboost_results.json`, `catboost_results.csv`, comparison charts
- Time: ~30 seconds

## Quick Start (Complete Pipeline)

For best results, run these commands in order:

```bash
cd pipeline

# 1. Explore data (optional)
python3 1_explore.py

# 2. Enhanced preprocessing
python3 2_preprocess_enhanced.py

# 3. Train CatBoost models
python3 5_models_catboost.py

# 4. Train ensemble (optional, for comparison)
python3 5_models_ensemble.py

# 5. Evaluate all models
python3 6_evaluate_catboost.py
```

Total time: ~15-20 minutes

## Output Files

All outputs are saved to `pipeline/output/`:

### Data Files
- `enhanced_data.csv` - Enhanced features (9,360 matches, 93 features)
- `cleaned_data.csv` - Basic preprocessed data
- `balanced_data.csv` - SMOTE-balanced data
- `feature_selected_data.csv` - Selected features only

### Model Files (.pkl)
- `catboost.pkl` - **BEST MODEL** (52.67% accuracy)
- `catboost_tuned.pkl` - Tuned CatBoost (52.62%)
- `ensemble_model.pkl` - Voting ensemble (51.18%)
- `rf_tuned.pkl` - Tuned Random Forest (49.73%)
- `gb_tuned.pkl` - Tuned Gradient Boosting (50.05%)
- `random_forest.pkl` - Basic Random Forest (51.18%)
- `gradient_boosting.pkl` - Basic Gradient Boosting (48.29%)

### Results Files
- `catboost_results.json` - All model metrics (JSON)
- `catboost_results.csv` - All model metrics (CSV)
- `ensemble_comparison.csv` - Ensemble comparison
- `model_comparison_catboost.csv` - Model comparison

### Visualization Files (.png)
- `catboost_comparison.png` - Model performance comparison
- `cm_catboost.png` - CatBoost confusion matrix
- `cm_catboost_tuned.png` - Tuned CatBoost confusion matrix
- `cm_ensemble.png` - Ensemble confusion matrix
- Plus exploration charts from step 1

## Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **CatBoost** | **52.67%** | **50.09%** | **52.67%** | **44.45%** |
| CatBoost (Tuned) | 52.62% | 47.58% | 52.62% | 45.60% |
| Ensemble | 51.18% | 46.59% | 51.18% | 46.39% |
| Random Forest | 51.18% | 44.48% | 51.18% | 44.24% |
| Gradient Boosting (Tuned) | 50.05% | 45.85% | 50.05% | 45.92% |

## Requirements

```bash
pip install pandas numpy scikit-learn catboost matplotlib seaborn joblib imbalanced-learn
```

## Notes

- **CatBoost** is the best performing model (52.67% accuracy)
- Enhanced preprocessing creates 93 features and takes 5-10 minutes
- All models use `random_state=42` for reproducibility
- XGBoost and LightGBM require OpenMP library (not essential)
- 52.67% accuracy is excellent for team-level data (random baseline is 33.3%)

## Troubleshooting

**Issue**: File not found errors
**Solution**: Make sure you're running from the `pipeline/` directory

**Issue**: XGBoost/LightGBM errors
**Solution**: These are optional. CatBoost works without OpenMP and performs best.

**Issue**: Out of memory
**Solution**: Close other applications or reduce dataset size

## For More Information

See project root documentation:
- `FINAL_RESULTS_SUMMARY.txt` - Complete analysis
- `CATBOOST_RESULTS.txt` - Detailed CatBoost results
- `QUICK_REFERENCE.txt` - Quick reference guide
- `PROJECT_SUMMARY_FOR_PROFESSOR.txt` - Executive summary
