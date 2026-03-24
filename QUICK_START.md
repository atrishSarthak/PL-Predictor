# ⚡ Quick Start Guide - EPL Predictor

## 🎯 TL;DR - Get Running in 5 Minutes

### 1️⃣ Get Dataset (REQUIRED)
```bash
# Download from: https://www.kaggle.com/datasets/marcohuiii/english-premier-league-epl-match-data-2000-2025
mkdir data
# Place downloaded CSV as: data/epl_matches.csv
```

### 2️⃣ Install Everything
```bash
pip install -r requirements.txt
cd api && npm install && cd ..
cd dashboard && npm install && cd ..
```

### 3️⃣ Run Pipeline (All 6 Phases)
```bash
python pipeline/1_explore.py
python pipeline/2_preprocess.py
python pipeline/3_imbalance.py
python pipeline/4_feature_selection.py
python pipeline/5_models.py
python pipeline/6_evaluate.py
```

### 4️⃣ Start Servers
```bash
# Terminal 1 - API Server
cd api && PORT=3001 npm start

# Terminal 2 - Dashboard
cd dashboard && npm run dev
```

### 5️⃣ Open Browser
```
http://localhost:5173
```

---

## 📁 Where Are Files Created?

### Input (You Provide)
```
data/epl_matches.csv    ← Download from Kaggle
```

### Output (Pipeline Creates)
```
pipeline/output/
├── cleaned_data.csv              ← Phase 2
├── balanced_data.csv             ← Phase 3
├── feature_selected_data.csv     ← Phase 4
├── selected_features.txt         ← Phase 4
├── X_test.csv, y_test.csv        ← Phase 5
├── model_*.pkl (5 files)         ← Phase 5
├── model_results.json            ← Phase 6
└── *.png (15+ charts)            ← All phases
```

---

## 🔗 Dataset Information

**Name**: English Premier League (EPL) Match Data 2000-2025

**Link**: https://www.kaggle.com/datasets/marcohuiii/english-premier-league-epl-match-data-2000-2025

**What it contains**:
- 25 years of EPL match data (2000-2025)
- ~10,000 matches
- Columns: Date, Season, HomeTeam, AwayTeam, FTHG, FTAG, FTR, etc.

**Required columns**:
- `Date` - Match date
- `HomeTeam` - Home team name
- `AwayTeam` - Away team name
- `FTR` - Full Time Result (H/D/A)
- `FTHG` - Full Time Home Goals
- `FTAG` - Full Time Away Goals

---

## 🧪 Quick Test Commands

### Test Pipeline Output
```bash
# Check if all files were created
ls -lh pipeline/output/

# Count output files
ls pipeline/output/ | wc -l
# Should show 20+ files

# View selected features
cat pipeline/output/selected_features.txt

# Check model results
cat pipeline/output/model_results.json | head -20
```

### Test API Server
```bash
# Health check
curl http://localhost:3001/api/health

# Get results
curl http://localhost:3001/api/results | head -50

# Get features
curl http://localhost:3001/api/features
```

### Test Dashboard
```
Open browser: http://localhost:5173

Pages to test:
- / (Overview)
- /team (Team Analysis)
- /models (Model Results)
- /predict (Predictor)
```

---

## ⚠️ Common Issues

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: data/epl_matches.csv` | Download dataset from Kaggle |
| `ModuleNotFoundError: pandas` | Run `pip install -r requirements.txt` |
| API returns 404 | Run all 6 pipeline scripts first |
| Dashboard shows "Failed to fetch" | Start API server on port 3001 |
| Port 3001 in use | Use `PORT=3002 npm start` |
| Images not loading | Check `pipeline/output/` has PNG files |

---

## 📊 Expected Results

### Pipeline Runtime
- Phase 1: ~30 seconds
- Phase 2: ~60 seconds
- Phase 3: ~40 seconds
- Phase 4: ~60 seconds
- Phase 5: ~2 minutes (SVM is slow)
- Phase 6: ~60 seconds
- **Total: ~5-7 minutes**

### Model Performance (Typical)
- Accuracy: 45-55%
- Best Model: Usually Random Forest or Logistic Regression
- F1-Score: 0.45-0.55

*Note: Football match prediction is inherently difficult. 50% accuracy is considered good!*

---

## 🎯 What Each Page Shows

### Overview (`/`)
- 4 exploration charts
- Match outcome distribution
- Goals analysis
- Season trends
- Top teams

### Team Analysis (`/team`)
- Feature importance rankings
- Feature correlations
- Class distribution before/after balancing

### Model Results (`/models`)
- Interactive comparison chart
- Metrics table (sorted by F1)
- 5 confusion matrices
- Best model highlighted

### Predictor (`/predict`)
- Feature input form
- Best model info
- Prediction interface

---

## 🚀 One-Line Commands

### Run All Pipeline Scripts
```bash
python pipeline/1_explore.py && python pipeline/2_preprocess.py && python pipeline/3_imbalance.py && python pipeline/4_feature_selection.py && python pipeline/5_models.py && python pipeline/6_evaluate.py
```

### Start Both Servers (macOS/Linux)
```bash
# Terminal 1
cd api && PORT=3001 npm start

# Terminal 2
cd dashboard && npm run dev
```

---

## ✅ Success Checklist

- [ ] Dataset at `data/epl_matches.csv`
- [ ] All dependencies installed
- [ ] Pipeline completed (check `pipeline/output/`)
- [ ] API running on port 3001
- [ ] Dashboard running on port 5173
- [ ] All pages load without errors
- [ ] Charts display correctly

---

## 📚 Full Documentation

For detailed explanations, see:
- `SETUP_GUIDE.md` - Complete step-by-step guide
- `README.md` - Project overview and features

---

**Ready to predict some matches? ⚽📊**
