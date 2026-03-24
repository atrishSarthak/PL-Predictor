# Complete Setup & Testing Guide for EPL Predictor

## 🎯 Overview

This guide will walk you through setting up and testing the complete Premier League Match Outcome Prediction System from scratch.

---

## 📋 Prerequisites

Before starting, ensure you have:
- **Python 3.10+** installed (`python --version`)
- **Node.js 16+** installed (`node --version`)
- **npm** installed (`npm --version`)
- **Git** installed (already done since you cloned the repo)

---

## 🗂️ Step 1: Get the Dataset

### Where to Download
The dataset is **NOT included** in the repository. You must download it from Kaggle:

**Dataset Link**: https://www.kaggle.com/datasets/marcohuiii/english-premier-league-epl-match-data-2000-2025

### How to Download
1. Go to the Kaggle link above
2. Click **"Download"** button (you may need to create a free Kaggle account)
3. You'll get a file named something like `epl_matches.csv` or `archive.zip`
4. If it's a ZIP file, extract it

### Where to Place the Dataset
```bash
# Create the data directory in your project root
mkdir data

# Move the downloaded CSV file to data/epl_matches.csv
# The file MUST be named exactly: epl_matches.csv
mv ~/Downloads/epl_matches.csv data/epl_matches.csv
```

**CRITICAL**: The pipeline expects the file at exactly this path:
```
PL-Predictor/
└── data/
    └── epl_matches.csv    ← Must be here with this exact name
```

---

## 🔧 Step 2: Install Dependencies

### Install Python Dependencies
```bash
# From project root directory
pip install -r requirements.txt

# This installs:
# - pandas (data manipulation)
# - numpy (numerical operations)
# - matplotlib (visualization)
# - seaborn (statistical plots)
# - scikit-learn (machine learning)
# - imbalanced-learn (SMOTE)
# - joblib (model serialization)
```

### Install API Server Dependencies
```bash
cd api
npm install
cd ..

# This installs:
# - express (web server)
# - cors (cross-origin support)
```

### Install Dashboard Dependencies
```bash
cd dashboard
npm install
cd ..

# This installs:
# - react, react-dom (UI framework)
# - react-router-dom (routing)
# - recharts (charts)
# - vite (build tool)
```

---

## 🚀 Step 3: Run the ML Pipeline (6 Phases)

The pipeline must be run in sequence. Each script depends on the output of the previous one.

### Phase 1: Data Exploration
```bash
python pipeline/1_explore.py
```

**What it does**:
- Loads `data/epl_matches.csv`
- Generates summary statistics
- Creates 5 visualization charts

**Output files** (saved to `pipeline/output/`):
- `outcome_distribution.png` - Bar chart of Home/Draw/Away wins
- `goals_boxplot.png` - Goals distribution analysis
- `season_trends.png` - Outcomes over seasons
- `top_teams_wins.png` - Top 10 teams by wins
- `exploration_summary.json` - Statistical summary

**Expected runtime**: 10-30 seconds

---

### Phase 2: Data Preprocessing
```bash
python pipeline/2_preprocess.py
```

**What it does**:
- Loads raw CSV
- Handles missing values
- Encodes categorical features (teams, seasons)
- Engineers features (rolling averages, win rates, head-to-head stats)
- Creates target variable (0=Home Win, 1=Draw, 2=Away Win)

**Output files**:
- `cleaned_data.csv` - Preprocessed dataset with engineered features

**Expected runtime**: 30-60 seconds

---

### Phase 3: Class Imbalance Handling
```bash
python pipeline/3_imbalance.py
```

**What it does**:
- Analyzes class distribution
- Intelligently decides whether to apply SMOTE
- Balances dataset if needed

**Output files**:
- `balanced_data.csv` - Balanced dataset
- `class_distribution_before.png` - Before balancing
- `class_distribution_after.png` - After balancing

**Expected runtime**: 20-40 seconds

---

### Phase 4: Feature Selection
```bash
python pipeline/4_feature_selection.py
```

**What it does**:
- Computes feature importance using Random Forest
- Ranks features by importance
- Selects top 10 most important features
- Analyzes feature correlations

**Output files**:
- `feature_selected_data.csv` - Dataset with only selected features
- `selected_features.txt` - List of selected feature names
- `feature_importance.png` - Feature importance bar chart
- `feature_correlations.png` - Correlation heatmap

**Expected runtime**: 30-60 seconds

---

### Phase 5: Model Training
```bash
python pipeline/5_models.py
```

**What it does**:
- Splits data into train/test sets (80/20)
- Trains 5 classification models:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. Naive Bayes
  5. SVM (RBF kernel)
- Saves trained models

**Output files**:
- `model_logistic_regression.pkl`
- `model_decision_tree.pkl`
- `model_random_forest.pkl`
- `model_naive_bayes.pkl`
- `model_svm.pkl`
- `X_test.csv` - Test features
- `y_test.csv` - Test labels

**Expected runtime**: 1-3 minutes (SVM is slowest)

---

### Phase 6: Model Evaluation
```bash
python pipeline/6_evaluate.py
```

**What it does**:
- Loads all trained models
- Evaluates on test set
- Computes accuracy, precision, recall, F1-score
- Generates confusion matrices
- Creates model comparison chart

**Output files**:
- `model_results.json` - All evaluation metrics
- `model_comparison.png` - Side-by-side comparison chart
- `cm_logistic_regression.png` - Confusion matrix
- `cm_decision_tree.png`
- `cm_random_forest.png`
- `cm_naive_bayes.png`
- `cm_svm.png`

**Expected runtime**: 30-60 seconds

---

## 🌐 Step 4: Start the API Server

```bash
cd api
PORT=3001 npm start
```

**What it does**:
- Starts Express server on port 3001
- Serves pipeline results via REST API
- Serves static files (PNGs, JSONs)

**You should see**:
```
🚀 API Server running on http://localhost:3001
📁 Serving files from: /path/to/pipeline/output
```

**Test the API**:
Open a new terminal and run:
```bash
# Health check
curl http://localhost:3001/api/health

# Get model results
curl http://localhost:3001/api/results

# Get selected features
curl http://localhost:3001/api/features
```

**Keep this terminal running!**

---

## 🎨 Step 5: Start the Dashboard

Open a **NEW terminal** (keep API server running):

```bash
cd dashboard
npm run dev
```

**What it does**:
- Starts Vite development server
- Serves React dashboard
- Connects to API at http://localhost:3001

**You should see**:
```
  VITE v4.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

**Open your browser**: http://localhost:5173

---

## 🧪 Step 6: Test the Dashboard

### Overview Page (http://localhost:5173/)
**What you should see**:
- 4 exploration charts:
  - Outcome distribution
  - Goals boxplot
  - Season trends
  - Top teams by wins

**Test**:
- All images should load
- No broken image icons
- Charts should be clear and readable

---

### Team Analysis Page (http://localhost:5173/team)
**What you should see**:
- Feature importance chart
- Feature correlations heatmap
- Class distribution (before/after balancing)

**Test**:
- All 4 images should load
- Hover over charts for interactivity

---

### Model Results Page (http://localhost:5173/models)
**What you should see**:
- Interactive bar chart comparing all 5 models
- Metrics table sorted by F1-score (best model highlighted in green)
- 5 confusion matrix heatmaps (one per model)

**Test**:
- Bar chart should be interactive (hover to see values)
- Table should show all metrics
- Best model row should have green background
- All confusion matrices should display

---

### Predictor Page (http://localhost:5173/predict)
**What you should see**:
- Form with input fields for features
- Best model information card
- Submit button

**Test**:
- Form should render
- Best model name and accuracy should display
- (Note: Prediction is placeholder - not real ML yet)

---

## 📊 Where Are the CSV Files Created?

All intermediate and output files are saved in `pipeline/output/`:

```
pipeline/output/
├── cleaned_data.csv              ← Phase 2 output
├── balanced_data.csv             ← Phase 3 output
├── feature_selected_data.csv     ← Phase 4 output
├── selected_features.txt         ← Phase 4 output
├── X_test.csv                    ← Phase 5 output
├── y_test.csv                    ← Phase 5 output
├── model_results.json            ← Phase 6 output
├── *.pkl                         ← Phase 5 output (5 model files)
└── *.png                         ← All phases (visualization charts)
```

**To view CSV files**:
```bash
# List all output files
ls -lh pipeline/output/

# View first 10 rows of cleaned data
head -10 pipeline/output/cleaned_data.csv

# Count rows in balanced data
wc -l pipeline/output/balanced_data.csv

# View selected features
cat pipeline/output/selected_features.txt
```

---

## 🔍 Troubleshooting

### Problem: "FileNotFoundError: data/epl_matches.csv"
**Solution**: Download the dataset from Kaggle and place it in `data/epl_matches.csv`

### Problem: "ModuleNotFoundError: No module named 'pandas'"
**Solution**: Run `pip install -r requirements.txt`

### Problem: API returns 404 for /api/results
**Solution**: Run all 6 pipeline scripts first to generate output files

### Problem: Dashboard shows "Failed to fetch data"
**Solution**: 
1. Make sure API server is running on port 3001
2. Check `dashboard/src/api.js` has `BASE_URL = 'http://localhost:3001'`

### Problem: Images not loading in dashboard
**Solution**: 
1. Check that pipeline generated PNG files in `pipeline/output/`
2. Verify API server is serving static files correctly
3. Check browser console for errors

### Problem: Port 3001 already in use
**Solution**: 
```bash
# Find process using port 3001
lsof -i :3001

# Kill the process
kill -9 <PID>

# Or use a different port
PORT=3002 npm start
# Then update dashboard/src/api.js to use 3002
```

---

## ✅ Complete Testing Checklist

- [ ] Dataset downloaded and placed in `data/epl_matches.csv`
- [ ] Python dependencies installed
- [ ] API dependencies installed
- [ ] Dashboard dependencies installed
- [ ] Phase 1 completed (5 PNG files created)
- [ ] Phase 2 completed (cleaned_data.csv created)
- [ ] Phase 3 completed (balanced_data.csv created)
- [ ] Phase 4 completed (feature_selected_data.csv + selected_features.txt created)
- [ ] Phase 5 completed (5 .pkl files + X_test.csv + y_test.csv created)
- [ ] Phase 6 completed (model_results.json + 6 PNG files created)
- [ ] API server running on port 3001
- [ ] Dashboard running on port 5173
- [ ] Overview page displays all 4 charts
- [ ] Team Analysis page displays all 4 charts
- [ ] Model Results page displays comparison chart + table + 5 confusion matrices
- [ ] Predictor page displays form and best model info

---

## 🎉 Success!

If all checkboxes are checked, your EPL Predictor is fully operational!

**Next Steps**:
- Explore different teams and seasons
- Compare model performance
- Experiment with different features
- Try different ML algorithms
- Deploy to production (optional)

---

## 📞 Need Help?

- Check the main README.md for additional information
- Review pipeline script logs for error messages
- Check browser console for frontend errors
- Verify all file paths are correct

Happy predicting! ⚽📊
