# EPL Match Prediction - Jupyter Notebooks

This folder contains Jupyter notebook versions of all pipeline scripts for interactive data analysis and model training.

## Notebooks

1. **1_explore.ipynb** - Data Exploration and Visualization
   - Load and inspect EPL match data
   - Generate exploratory visualizations
   - Statistical analysis

2. **2_preprocess.ipynb** - Data Preprocessing
   - Data cleaning and missing value handling
   - Feature engineering
   - Data scaling and encoding

3. **3_imbalance.ipynb** - Class Imbalance Handling
   - Analyze class distribution
   - Apply SMOTE if needed
   - Balance dataset

4. **4_feature_selection.ipynb** - Feature Selection
   - Correlation analysis
   - Random Forest feature importance
   - Select top features

5. **5_models.ipynb** - Model Training
   - Train multiple classification models
   - Save trained models
   - Compare training times

6. **6_evaluate.ipynb** - Model Evaluation
   - Evaluate model performance
   - Generate confusion matrices
   - Compare models and select best

## Usage

### Running Notebooks

1. Install Jupyter:
```bash
pip install jupyter notebook
```

2. Start Jupyter Notebook:
```bash
jupyter notebook
```

3. Navigate to the `notebooks/` folder and open any notebook

4. Run cells sequentially using Shift+Enter

### Running from Command Line

If you prefer running Python scripts instead:
```bash
python pipeline/1_explore.py
python pipeline/2_preprocess.py
python pipeline/3_imbalance.py
python pipeline/4_feature_selection.py
python pipeline/5_models.py
python pipeline/6_evaluate.py
```

## Requirements

Make sure you have all dependencies installed:
```bash
pip install -r requirements.txt
```

## Notes

- Notebooks should be run in order (1 through 6)
- Each notebook depends on outputs from previous notebooks
- All outputs are saved to `pipeline/output/` directory
- Adjust file paths if running from different directories
