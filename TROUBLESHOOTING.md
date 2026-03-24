# 🔧 Troubleshooting Guide

## Common Issues and Solutions

---

## 🚨 Pipeline Issues

### Issue 1: FileNotFoundError: data/epl_matches.csv

**Error Message**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/epl_matches.csv'
```

**Cause**: Dataset file is missing

**Solution**:
1. Download dataset from Kaggle: https://www.kaggle.com/datasets/marcohuiii/english-premier-league-epl-match-data-2000-2025
2. Create `data/` directory: `mkdir data`
3. Place CSV file as `data/epl_matches.csv` (exact name required)
4. Verify: `ls -lh data/epl_matches.csv`

---

### Issue 2: ModuleNotFoundError: No module named 'pandas'

**Error Message**:
```
ModuleNotFoundError: No module named 'pandas'
```

**Cause**: Python dependencies not installed

**Solution**:
```bash
pip install -r requirements.txt

# Or install individually:
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
```

**Verify Installation**:
```bash
python -c "import pandas; print(pandas.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
```

---

### Issue 3: KeyError: 'FTR' or Missing Column

**Error Message**:
```
KeyError: 'FTR'
```

**Cause**: CSV file has different column names

**Solution**:
1. Open `data/epl_matches.csv` and check column names
2. Required columns: `Date`, `HomeTeam`, `AwayTeam`, `FTR`, `FTHG`, `FTAG`
3. If columns are named differently, either:
   - Rename columns in CSV to match expected names
   - Modify pipeline scripts to use actual column names

**Check Columns**:
```bash
head -1 data/epl_matches.csv
```

---

### Issue 4: Pipeline Script Crashes Midway

**Symptoms**: Script stops with error, no output files created

**Common Causes**:
- Insufficient memory (dataset too large)
- Corrupted CSV file
- Missing dependencies

**Solution**:
```bash
# Check available memory
free -h  # Linux
vm_stat  # macOS

# Verify CSV file integrity
wc -l data/epl_matches.csv
head -10 data/epl_matches.csv
tail -10 data/epl_matches.csv

# Re-run with verbose output
python -u pipeline/1_explore.py 2>&1 | tee explore.log
```

---

### Issue 5: SMOTE Error - Not Enough Samples

**Error Message**:
```
ValueError: Expected n_neighbors <= n_samples, but n_neighbors = 5, n_samples = 3
```

**Cause**: One class has too few samples for SMOTE

**Solution**: This is handled automatically in Phase 3. If you see this error:
1. Check `pipeline/3_imbalance.py` has the intelligent SMOTE decision logic
2. The script should skip SMOTE if any class has < 10 samples
3. If error persists, check your dataset has enough data

---

### Issue 6: Model Training Takes Too Long

**Symptoms**: Phase 5 runs for >10 minutes

**Cause**: SVM with large dataset is slow

**Solution**:
```bash
# Option 1: Wait it out (SVM can take 5-10 minutes)

# Option 2: Skip SVM temporarily
# Edit pipeline/5_models.py and comment out SVM:
# "SVM": SVC(kernel="rbf", random_state=42, probability=True)

# Option 3: Use smaller dataset for testing
# Take first 1000 rows of CSV for quick testing
head -1001 data/epl_matches.csv > data/epl_matches_small.csv
```

---

## 🌐 API Server Issues

### Issue 7: API Returns 404 for /api/results

**Error Message** (in browser console):
```
GET http://localhost:3001/api/results 404 (Not Found)
```

**Cause**: Pipeline output files don't exist

**Solution**:
1. Check if pipeline completed: `ls -lh pipeline/output/`
2. Verify `model_results.json` exists: `ls pipeline/output/model_results.json`
3. If missing, run Phase 6: `python pipeline/6_evaluate.py`
4. Restart API server

---

### Issue 8: Port 3001 Already in Use

**Error Message**:
```
Error: listen EADDRINUSE: address already in use :::3001
```

**Cause**: Another process is using port 3001

**Solution**:
```bash
# Option 1: Find and kill the process
lsof -i :3001
kill -9 <PID>

# Option 2: Use a different port
PORT=3002 npm start

# Then update dashboard/src/api.js:
# const BASE_URL = 'http://localhost:3002';
```

---

### Issue 9: CORS Error in Browser Console

**Error Message**:
```
Access to fetch at 'http://localhost:3001/api/results' from origin 'http://localhost:5173' has been blocked by CORS policy
```

**Cause**: CORS not configured properly

**Solution**:
1. Check `api/server.js` has `app.use(cors())`
2. Restart API server
3. Clear browser cache (Ctrl+Shift+R or Cmd+Shift+R)

---

### Issue 10: API Server Won't Start

**Error Message**:
```
Error: Cannot find module 'express'
```

**Cause**: Node dependencies not installed

**Solution**:
```bash
cd api
npm install
npm start
```

---

## 🎨 Dashboard Issues

### Issue 11: Dashboard Shows "Failed to fetch data"

**Symptoms**: Dashboard loads but shows error message

**Cause**: API server not running or wrong URL

**Solution**:
1. Check API server is running: `curl http://localhost:3001/api/health`
2. Check `dashboard/src/api.js` has correct `BASE_URL`:
   ```javascript
   const BASE_URL = 'http://localhost:3001';
   ```
3. Restart both servers
4. Check browser console for detailed error

---

### Issue 12: Images Not Loading (Broken Image Icons)

**Symptoms**: Dashboard shows broken image icons instead of charts

**Cause**: PNG files missing or API not serving them

**Solution**:
1. Check PNG files exist: `ls pipeline/output/*.png`
2. Test direct URL: `http://localhost:3001/output/outcome_distribution.png`
3. Check browser console for 404 errors
4. Verify API server is serving static files correctly

---

### Issue 13: Dashboard Won't Start

**Error Message**:
```
Error: Cannot find module 'react'
```

**Cause**: Node dependencies not installed

**Solution**:
```bash
cd dashboard
npm install
npm run dev
```

---

### Issue 14: Blank White Page

**Symptoms**: Dashboard loads but shows nothing

**Cause**: JavaScript error or build issue

**Solution**:
1. Open browser console (F12)
2. Check for errors
3. Common fixes:
   ```bash
   cd dashboard
   rm -rf node_modules package-lock.json
   npm install
   npm run dev
   ```

---

### Issue 15: Charts Not Interactive

**Symptoms**: Charts display but hover/click doesn't work

**Cause**: Recharts not loaded properly

**Solution**:
1. Check Recharts is installed: `npm list recharts`
2. Reinstall if needed: `npm install recharts`
3. Clear browser cache
4. Restart dev server

---

## 🔍 Debugging Commands

### Check Pipeline Output
```bash
# List all output files
ls -lh pipeline/output/

# Count files
ls pipeline/output/ | wc -l

# Check specific files exist
ls pipeline/output/model_results.json
ls pipeline/output/*.pkl
ls pipeline/output/*.png

# View JSON structure
cat pipeline/output/model_results.json | python -m json.tool | head -50

# Check CSV row counts
wc -l pipeline/output/*.csv
```

### Test API Endpoints
```bash
# Health check
curl http://localhost:3001/api/health

# Get results (first 100 chars)
curl http://localhost:3001/api/results | head -c 100

# Get features
curl http://localhost:3001/api/features

# Test image serving
curl -I http://localhost:3001/output/outcome_distribution.png
```

### Check Processes
```bash
# Check if API server is running
lsof -i :3001

# Check if dashboard is running
lsof -i :5173

# Check all Node processes
ps aux | grep node

# Check all Python processes
ps aux | grep python
```

### Browser Console Commands
```javascript
// Check API connection
fetch('http://localhost:3001/api/health')
  .then(r => r.json())
  .then(console.log)

// Check results endpoint
fetch('http://localhost:3001/api/results')
  .then(r => r.json())
  .then(console.log)

// Check image loading
fetch('http://localhost:3001/output/outcome_distribution.png')
  .then(r => console.log(r.status, r.headers.get('content-type')))
```

---

## 🧪 Verification Checklist

Use this checklist to verify everything is working:

### Pipeline Verification
```bash
# ✅ Dataset exists
[ -f data/epl_matches.csv ] && echo "✅ Dataset found" || echo "❌ Dataset missing"

# ✅ Phase 1 output
[ -f pipeline/output/outcome_distribution.png ] && echo "✅ Phase 1 complete" || echo "❌ Phase 1 incomplete"

# ✅ Phase 2 output
[ -f pipeline/output/cleaned_data.csv ] && echo "✅ Phase 2 complete" || echo "❌ Phase 2 incomplete"

# ✅ Phase 3 output
[ -f pipeline/output/balanced_data.csv ] && echo "✅ Phase 3 complete" || echo "❌ Phase 3 incomplete"

# ✅ Phase 4 output
[ -f pipeline/output/feature_selected_data.csv ] && echo "✅ Phase 4 complete" || echo "❌ Phase 4 incomplete"

# ✅ Phase 5 output
[ -f pipeline/output/model_random_forest.pkl ] && echo "✅ Phase 5 complete" || echo "❌ Phase 5 incomplete"

# ✅ Phase 6 output
[ -f pipeline/output/model_results.json ] && echo "✅ Phase 6 complete" || echo "❌ Phase 6 incomplete"
```

### API Verification
```bash
# ✅ API health check
curl -s http://localhost:3001/api/health | grep -q "ok" && echo "✅ API healthy" || echo "❌ API not responding"

# ✅ Results endpoint
curl -s http://localhost:3001/api/results | grep -q "accuracy" && echo "✅ Results available" || echo "❌ Results not available"

# ✅ Image serving
curl -s -I http://localhost:3001/output/outcome_distribution.png | grep -q "image/png" && echo "✅ Images serving" || echo "❌ Images not serving"
```

### Dashboard Verification
```bash
# ✅ Dashboard running
curl -s http://localhost:5173 | grep -q "<!DOCTYPE html>" && echo "✅ Dashboard running" || echo "❌ Dashboard not running"
```

---

## 🆘 Still Having Issues?

### Collect Debug Information
```bash
# Create debug report
echo "=== System Info ===" > debug_report.txt
python --version >> debug_report.txt
node --version >> debug_report.txt
npm --version >> debug_report.txt

echo -e "\n=== Python Packages ===" >> debug_report.txt
pip list >> debug_report.txt

echo -e "\n=== Pipeline Output ===" >> debug_report.txt
ls -lh pipeline/output/ >> debug_report.txt

echo -e "\n=== API Test ===" >> debug_report.txt
curl -s http://localhost:3001/api/health >> debug_report.txt

cat debug_report.txt
```

### Reset Everything
```bash
# Nuclear option - start fresh
rm -rf pipeline/output/*
rm -rf api/node_modules
rm -rf dashboard/node_modules

# Reinstall
pip install -r requirements.txt
cd api && npm install && cd ..
cd dashboard && npm install && cd ..

# Re-run pipeline
python pipeline/1_explore.py
# ... continue with other phases
```

---

## 📞 Getting Help

If you're still stuck:

1. **Check the logs**: Look for error messages in terminal output
2. **Check browser console**: Press F12 and look for errors
3. **Verify file paths**: Make sure all paths are correct
4. **Check permissions**: Ensure you have read/write permissions
5. **Try with smaller dataset**: Test with first 1000 rows
6. **Review documentation**: Check README.md and SETUP_GUIDE.md

---

**Most issues are caused by:**
1. Missing dataset file (50%)
2. Dependencies not installed (30%)
3. Pipeline not run completely (15%)
4. Port conflicts (5%)

**Good luck! 🍀**
