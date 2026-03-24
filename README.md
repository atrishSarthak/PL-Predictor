# Premier League Match Outcome Prediction System

A comprehensive data mining project that predicts Premier League match outcomes (Home Win, Draw, Away Win) using historical match data from 2000-2025. The system includes a complete machine learning pipeline, REST API, and interactive React dashboard.

## 🎯 Project Overview

This academic data mining project demonstrates:
- **Data Exploration & Visualization** - Comprehensive EDA with statistical analysis
- **Data Preprocessing** - Feature engineering, encoding, and scaling
- **Class Imbalance Handling** - Intelligent SMOTE application
- **Feature Selection** - Random Forest-based importance ranking
- **Model Training** - 5 classification algorithms (Logistic Regression, Decision Tree, Random Forest, Naive Bayes, SVM)
- **Model Evaluation** - Detailed performance metrics and confusion matrices
- **REST API** - Node.js/Express backend serving results
- **Interactive Dashboard** - React frontend for visualization and analysis

## 📊 Tech Stack

### Backend (Python)
- pandas, numpy - Data manipulation
- matplotlib, seaborn - Visualization
- scikit-learn - Machine learning
- imbalanced-learn - SMOTE balancing

### API Server (Node.js)
- Express - Web framework
- CORS - Cross-origin support

### Frontend (React)
- React 18 - UI framework
- React Router - Navigation
- Recharts - Interactive charts
- Vite - Build tool

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 16+
- EPL dataset (download from Kaggle)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/atrishSarthak/PL-Predictor.git
cd PL-Predictor
```

2. **Set up data directory**
```bash
mkdir data
# Download EPL dataset from Kaggle and place as data/epl_matches.csv
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Install API dependencies**
```bash
cd api
npm install
cd ..
```

5. **Install Dashboard dependencies**
```bash
cd dashboard
npm install
cd ..
```

## 📈 Running the Pipeline

Execute the 6-phase data mining pipeline in sequence:

```bash
# Phase 1: Data Exploration
python pipeline/1_explore.py

# Phase 2: Data Preprocessing
python pipeline/2_preprocess.py

# Phase 3: Class Imbalance Handling
python pipeline/3_imbalance.py

# Phase 4: Feature Selection
python pipeline/4_feature_selection.py

# Phase 5: Model Training
python pipeline/5_models.py

# Phase 6: Model Evaluation
python pipeline/6_evaluate.py
```

**Output**: All results saved to `pipeline/output/` including:
- Visualization charts (PNG)
- Processed datasets (CSV)
- Trained models (PKL)
- Evaluation results (JSON)

## 🌐 Running the Application

### Start API Server
```bash
cd api
PORT=3001 npm start
```
API will be available at: http://localhost:3001

### Start Dashboard
```bash
cd dashboard
npm run dev
```
Dashboard will be available at: http://localhost:3000

## 📱 Dashboard Features

### Overview Page
- Match outcome distribution
- Goals analysis
- Season-wise trends
- Top teams by wins

### Team Analysis Page
- Feature importance rankings
- Feature correlations
- Class distribution analysis

### Model Results Page
- Interactive performance comparison chart
- Detailed metrics table (ranked by F1-score)
- Confusion matrices for all models
- Best model highlighting

### Predictor Page
- Feature input form
- Best model information
- Prediction interface (placeholder)

## 📁 Project Structure

```
PL-Predictor/
├── pipeline/              # Python ML pipeline
│   ├── 1_explore.py      # Data exploration
│   ├── 2_preprocess.py   # Data preprocessing
│   ├── 3_imbalance.py    # Class balancing
│   ├── 4_feature_selection.py
│   ├── 5_models.py       # Model training
│   ├── 6_evaluate.py     # Model evaluation
│   └── output/           # Generated files
├── api/                  # Node.js API server
│   ├── server.js
│   └── package.json
├── dashboard/            # React frontend
│   ├── src/
│   │   ├── pages/       # Page components
│   │   ├── components/  # Reusable components
│   │   ├── styles/      # CSS modules
│   │   └── api.js       # API service
│   └── package.json
├── data/                 # Dataset (not in repo)
│   └── epl_matches.csv
└── requirements.txt      # Python dependencies
```

## 🔬 Machine Learning Models

The system trains and evaluates 5 classification models:

1. **Logistic Regression** - Linear baseline model
2. **Decision Tree** - Non-linear tree-based model
3. **Random Forest** - Ensemble of decision trees
4. **Naive Bayes** - Probabilistic classifier
5. **SVM (RBF kernel)** - Support vector machine

Models are evaluated using:
- Accuracy
- Precision (per class)
- Recall (per class)
- F1-Score (weighted average)
- Confusion matrices

## 📊 Dataset

**Source**: Kaggle - English Premier League (EPL) Match Data 2000-2025

**Key Columns**:
- Date, Season
- HomeTeam, AwayTeam
- FTHG, FTAG (Full Time Goals)
- FTR (Full Time Result: H/D/A)
- HTHG, HTAG (Half Time Goals)
- HTR (Half Time Result)

**Note**: Dataset not included in repository. Download from Kaggle and place in `data/` folder.

## 🛠️ API Endpoints

- `GET /api/health` - Health check
- `GET /api/results` - Model evaluation results
- `GET /api/features` - Selected features list
- `GET /api/charts` - Available chart files
- `GET /api/models` - Trained model files
- `GET /output/<filename>` - Static file serving

## ⚙️ Configuration

### API Port
Default: 5000 (configurable via `PORT` environment variable)

To run on port 3001:
```bash
PORT=3001 npm start
```

### Dashboard API URL
Edit `dashboard/src/api.js`:
```javascript
const BASE_URL = 'http://localhost:3001';
```

## 🧪 Testing

The system includes comprehensive error handling:
- Graceful fallbacks for missing data
- API connection error handling
- Image loading fallbacks
- Loading and error states in UI

## 📝 Development Notes

### Data Flow
```
Raw CSV → Cleaned → Balanced → Feature Selected → Trained Models → Evaluated → JSON Results → API → React Dashboard
```

### Key Features
- ✅ Independent script execution with `__main__` guards
- ✅ Progress logging for all operations
- ✅ Robust error handling (no crashes)
- ✅ Cross-platform path handling
- ✅ Reproducible results (random_state=42)
- ✅ Modular, maintainable code structure

## 🤝 Contributing

This is an academic project. Feel free to fork and adapt for your own use.

## 📄 License

MIT License - See LICENSE file for details

## 👤 Author

Sarthak Atrish
- GitHub: [@atrishSarthak](https://github.com/atrishSarthak)

## 🙏 Acknowledgments

- Dataset: Kaggle EPL Match Data
- Libraries: scikit-learn, React, Express
- Inspiration: Academic data mining coursework

---

**Note**: This project is for educational purposes. Predictions are based on historical data and should not be used for betting or commercial purposes.
