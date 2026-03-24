# 📊 Complete Data Flow Diagram

## 🔄 End-to-End System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCE                                  │
│  📥 Kaggle Dataset: EPL Match Data 2000-2025                        │
│  https://www.kaggle.com/datasets/marcohuiii/english-premier-league  │
│                                                                      │
│  ⬇️ Download & Place Here                                           │
│  data/epl_matches.csv                                               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: EXPLORATION                              │
│  📜 Script: pipeline/1_explore.py                                   │
│                                                                      │
│  Input:  data/epl_matches.csv                                       │
│  Output: pipeline/output/                                           │
│    ├── outcome_distribution.png                                     │
│    ├── goals_boxplot.png                                            │
│    ├── season_trends.png                                            │
│    ├── top_teams_wins.png                                           │
│    └── exploration_summary.json                                     │
│                                                                      │
│  ⏱️  Runtime: ~30 seconds                                           │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PHASE 2: PREPROCESSING                              │
│  📜 Script: pipeline/2_preprocess.py                                │
│                                                                      │
│  Input:  data/epl_matches.csv                                       │
│  Process:                                                            │
│    • Handle missing values                                          │
│    • Encode teams & seasons (Label Encoding)                        │
│    • Engineer features:                                             │
│      - Rolling averages (last 5 matches)                            │
│      - Win rates (last 10 matches)                                  │
│      - Head-to-head statistics                                      │
│      - Days since last match                                        │
│    • Create target variable (H→0, D→1, A→2)                         │
│                                                                      │
│  Output: pipeline/output/cleaned_data.csv                           │
│    Columns: ~20 engineered features + target                        │
│                                                                      │
│  ⏱️  Runtime: ~60 seconds                                           │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│               PHASE 3: CLASS IMBALANCE HANDLING                      │
│  📜 Script: pipeline/3_imbalance.py                                 │
│                                                                      │
│  Input:  pipeline/output/cleaned_data.csv                           │
│  Process:                                                            │
│    • Analyze class distribution                                     │
│    • Intelligent SMOTE decision:                                    │
│      - If imbalance > 1.5x → Apply SMOTE                            │
│      - If balanced → Skip SMOTE                                     │
│    • Generate before/after visualizations                           │
│                                                                      │
│  Output: pipeline/output/                                           │
│    ├── balanced_data.csv                                            │
│    ├── class_distribution_before.png                                │
│    └── class_distribution_after.png                                 │
│                                                                      │
│  ⏱️  Runtime: ~40 seconds                                           │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 PHASE 4: FEATURE SELECTION                           │
│  📜 Script: pipeline/4_feature_selection.py                         │
│                                                                      │
│  Input:  pipeline/output/balanced_data.csv                          │
│  Process:                                                            │
│    • Compute feature importance (Random Forest)                     │
│    • Rank features by importance                                    │
│    • Select top 10 features                                         │
│    • Analyze feature correlations                                   │
│                                                                      │
│  Output: pipeline/output/                                           │
│    ├── feature_selected_data.csv (only top 10 features)            │
│    ├── selected_features.txt (feature names)                        │
│    ├── feature_importance.png                                       │
│    └── feature_correlations.png                                     │
│                                                                      │
│  ⏱️  Runtime: ~60 seconds                                           │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   PHASE 5: MODEL TRAINING                            │
│  📜 Script: pipeline/5_models.py                                    │
│                                                                      │
│  Input:  pipeline/output/feature_selected_data.csv                  │
│  Process:                                                            │
│    • Split data: 80% train, 20% test                                │
│    • Train 5 models:                                                │
│      1. Logistic Regression                                         │
│      2. Decision Tree                                               │
│      3. Random Forest                                               │
│      4. Naive Bayes                                                 │
│      5. SVM (RBF kernel)                                            │
│    • Save trained models                                            │
│                                                                      │
│  Output: pipeline/output/                                           │
│    ├── model_logistic_regression.pkl                                │
│    ├── model_decision_tree.pkl                                      │
│    ├── model_random_forest.pkl                                      │
│    ├── model_naive_bayes.pkl                                        │
│    ├── model_svm.pkl                                                │
│    ├── X_test.csv                                                   │
│    └── y_test.csv                                                   │
│                                                                      │
│  ⏱️  Runtime: ~2 minutes (SVM is slowest)                          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PHASE 6: MODEL EVALUATION                           │
│  📜 Script: pipeline/6_evaluate.py                                  │
│                                                                      │
│  Input:  pipeline/output/                                           │
│    • All 5 .pkl model files                                         │
│    • X_test.csv, y_test.csv                                         │
│                                                                      │
│  Process:                                                            │
│    • Load each model                                                │
│    • Predict on test set                                            │
│    • Compute metrics:                                               │
│      - Accuracy                                                     │
│      - Precision (per class)                                        │
│      - Recall (per class)                                           │
│      - F1-Score (weighted)                                          │
│    • Generate confusion matrices                                    │
│    • Create comparison chart                                        │
│                                                                      │
│  Output: pipeline/output/                                           │
│    ├── model_results.json (all metrics)                             │
│    ├── model_comparison.png                                         │
│    ├── cm_logistic_regression.png                                   │
│    ├── cm_decision_tree.png                                         │
│    ├── cm_random_forest.png                                         │
│    ├── cm_naive_bayes.png                                           │
│    └── cm_svm.png                                                   │
│                                                                      │
│  ⏱️  Runtime: ~60 seconds                                           │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      API SERVER LAYER                                │
│  📜 Script: api/server.js                                           │
│  🌐 Port: 3001                                                      │
│                                                                      │
│  Endpoints:                                                          │
│    GET /api/health          → Health check                          │
│    GET /api/results         → model_results.json                    │
│    GET /api/features        → selected_features.txt                 │
│    GET /api/charts          → List of PNG files                     │
│    GET /api/models          → List of model files                   │
│    GET /output/<filename>   → Serve static files                    │
│                                                                      │
│  Technology: Node.js + Express + CORS                               │
│                                                                      │
│  ⏱️  Response Time: <100ms                                          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FRONTEND DASHBOARD                                │
│  📜 Framework: React + Vite                                         │
│  🌐 Port: 5173                                                      │
│                                                                      │
│  Pages:                                                              │
│                                                                      │
│  1️⃣  OVERVIEW (/)                                                   │
│     • Displays 4 exploration charts                                 │
│     • Fetches PNGs from API                                         │
│                                                                      │
│  2️⃣  TEAM ANALYSIS (/team)                                          │
│     • Feature importance chart                                      │
│     • Feature correlations                                          │
│     • Class distribution                                            │
│                                                                      │
│  3️⃣  MODEL RESULTS (/models)                                        │
│     • Interactive comparison chart (Recharts)                       │
│     • Metrics table (sorted by F1)                                  │
│     • 5 confusion matrices                                          │
│     • Best model highlighted                                        │
│                                                                      │
│  4️⃣  PREDICTOR (/predict)                                           │
│     • Feature input form                                            │
│     • Best model info                                               │
│     • Prediction interface                                          │
│                                                                      │
│  Technology: React Router + Recharts + CSS Modules                  │
│                                                                      │
│  ⏱️  Load Time: <2 seconds                                          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    👤 USER BROWSER
                    http://localhost:5173
```

---

## 📁 Complete File Structure

```
PL-Predictor/
│
├── data/                           ← YOU CREATE THIS
│   └── epl_matches.csv            ← DOWNLOAD FROM KAGGLE
│
├── pipeline/                       ← PYTHON ML PIPELINE
│   ├── 1_explore.py               ← Phase 1
│   ├── 2_preprocess.py            ← Phase 2
│   ├── 3_imbalance.py             ← Phase 3
│   ├── 4_feature_selection.py     ← Phase 4
│   ├── 5_models.py                ← Phase 5
│   ├── 6_evaluate.py              ← Phase 6
│   │
│   └── output/                     ← ALL OUTPUT FILES GO HERE
│       ├── cleaned_data.csv
│       ├── balanced_data.csv
│       ├── feature_selected_data.csv
│       ├── selected_features.txt
│       ├── X_test.csv
│       ├── y_test.csv
│       ├── model_*.pkl (5 files)
│       ├── model_results.json
│       └── *.png (15+ charts)
│
├── api/                            ← NODE.JS API SERVER
│   ├── server.js                  ← Express server
│   ├── package.json
│   └── node_modules/
│
├── dashboard/                      ← REACT FRONTEND
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api.js                 ← API client
│   │   ├── pages/
│   │   │   ├── Overview.jsx
│   │   │   ├── TeamAnalysis.jsx
│   │   │   ├── ModelResults.jsx
│   │   │   └── Predictor.jsx
│   │   ├── components/
│   │   │   └── ConfusionMatrix.jsx
│   │   └── styles/
│   │       └── layout.module.css
│   ├── package.json
│   └── node_modules/
│
├── requirements.txt                ← Python dependencies
├── README.md                       ← Project overview
├── SETUP_GUIDE.md                 ← Detailed setup guide
├── QUICK_START.md                 ← Quick reference
└── DATA_FLOW.md                   ← This file
```

---

## 🔄 Data Transformation Journey

### Raw Data → Cleaned Data
```
epl_matches.csv (10,000 rows × 20 cols)
    ↓ [Phase 2: Preprocessing]
cleaned_data.csv (10,000 rows × 25 cols)
    • Added: rolling averages, win rates, h2h stats
    • Encoded: teams, seasons
    • Target: H/D/A → 0/1/2
```

### Cleaned Data → Balanced Data
```
cleaned_data.csv (10,000 rows)
    ↓ [Phase 3: SMOTE if needed]
balanced_data.csv (10,000-15,000 rows)
    • Class distribution balanced
    • Synthetic samples added if needed
```

### Balanced Data → Feature Selected Data
```
balanced_data.csv (25 features)
    ↓ [Phase 4: Feature Selection]
feature_selected_data.csv (10 features)
    • Only top 10 most important features
    • Reduced dimensionality
```

### Feature Selected Data → Trained Models
```
feature_selected_data.csv
    ↓ [Phase 5: Train/Test Split 80/20]
X_train (80%) → Train 5 models → 5 .pkl files
X_test (20%) → Saved for evaluation
```

### Trained Models → Evaluation Results
```
5 .pkl files + X_test.csv + y_test.csv
    ↓ [Phase 6: Evaluation]
model_results.json + 6 PNG files
    • Metrics for all models
    • Confusion matrices
    • Comparison chart
```

---

## 📊 Data Size Estimates

| File | Rows | Columns | Size |
|------|------|---------|------|
| epl_matches.csv | ~10,000 | 20 | ~2 MB |
| cleaned_data.csv | ~10,000 | 25 | ~3 MB |
| balanced_data.csv | ~10,000-15,000 | 25 | ~4 MB |
| feature_selected_data.csv | ~10,000-15,000 | 10 | ~2 MB |
| X_test.csv | ~2,000-3,000 | 10 | ~500 KB |
| y_test.csv | ~2,000-3,000 | 1 | ~50 KB |
| model_*.pkl | - | - | ~1-5 MB each |
| *.png | - | - | ~50-200 KB each |

**Total Output Size**: ~50-100 MB

---

## 🎯 Key Takeaways

1. **Dataset is NOT included** - You must download from Kaggle
2. **Pipeline must run in sequence** - Each phase depends on previous
3. **All outputs go to `pipeline/output/`** - One central location
4. **API serves pipeline outputs** - No database needed
5. **Dashboard fetches from API** - Clean separation of concerns

---

## 🚀 Quick Commands Reference

```bash
# Download dataset
# → Place in data/epl_matches.csv

# Run complete pipeline
python pipeline/1_explore.py && \
python pipeline/2_preprocess.py && \
python pipeline/3_imbalance.py && \
python pipeline/4_feature_selection.py && \
python pipeline/5_models.py && \
python pipeline/6_evaluate.py

# Start API (Terminal 1)
cd api && PORT=3001 npm start

# Start Dashboard (Terminal 2)
cd dashboard && npm run dev

# Open browser
# → http://localhost:5173
```

---

**Now you understand the complete data flow! 🎉**
