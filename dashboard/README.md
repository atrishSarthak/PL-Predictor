# EPL Match Prediction Dashboard

React-based dashboard for visualizing EPL match prediction results and model performance.

## Prerequisites

- Node.js 16+ installed
- API server running on http://localhost:3001
- Pipeline scripts executed to generate data files

## Installation

```bash
cd dashboard
npm install
```

## Running the Dashboard

```bash
npm run dev
```

The dashboard will be available at: **http://localhost:3000**

## Pages

### 1. Overview (/)
- Displays exploratory data analysis charts
- Match outcome distribution
- Goals analysis
- Season trends
- Top teams by wins

### 2. Team Analysis (/team)
- Feature importance rankings
- Feature correlations
- Class distribution (before/after balancing)

### 3. Model Results (/models)
- Interactive performance comparison chart
- Detailed metrics table (ranked by F1-score)
- Confusion matrices for all models
- Best model highlighted

### 4. Predictor (/predict)
- Input form for feature values
- Displays best performing model
- Mock prediction interface (placeholder)

## API Configuration

The dashboard connects to the API server at `http://localhost:3001`.

To change the API URL, edit `src/api.js`:

```javascript
const BASE_URL = 'http://localhost:3001';
```

## Troubleshooting

### Images Not Loading
- Ensure API server is running
- Verify pipeline scripts have been executed
- Check browser console for CORS errors

### API Connection Failed
- Verify API server is running on port 3001
- Check CORS is enabled on the API server
- Ensure no firewall blocking localhost connections

### Charts Not Rendering
- Ensure model_results.json exists in pipeline/output/
- Check browser console for errors
- Verify data format matches expected structure

## Build for Production

```bash
npm run build
```

Built files will be in the `dist/` directory.

## Technologies Used

- React 18
- React Router DOM 6
- Recharts (charting library)
- Vite (build tool)
- CSS Modules (styling)
