import { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { fetchResults, getStaticUrl } from '../api';
import ConfusionMatrix from '../components/ConfusionMatrix';
import styles from '../styles/layout.module.css';

/**
 * Model Results Page
 * Displays model performance metrics, comparison chart, and confusion matrices
 */
export default function ModelResults() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadResults();
  }, []);

  async function loadResults() {
    setLoading(true);
    setError(null);
    
    const data = await fetchResults();
    
    if (data) {
      setResults(data);
    } else {
      setError('Failed to load model results. Please ensure the API is running and pipeline/6_evaluate.py has been executed.');
    }
    
    setLoading(false);
  }

  if (loading) {
    return (
      <div className={styles.container}>
        <div className={styles.loading}>Loading model results...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={styles.container}>
        <div className={styles.error}>{error}</div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className={styles.container}>
        <div className={styles.error}>No results available</div>
      </div>
    );
  }

  // Convert results to array format for Recharts (filter out metadata)
  const chartData = Object.entries(results)
    .filter(([modelName]) => modelName !== '_metadata')
    .map(([modelName, metrics]) => ({
      name: modelName,
      accuracy: (metrics.accuracy * 100).toFixed(2),
      precision: (metrics.precision * 100).toFixed(2),
      recall: (metrics.recall * 100).toFixed(2),
      f1: (metrics.f1 * 100).toFixed(2)
    }));

  // Sort models by F1 score for table (filter out metadata)
  const sortedModels = Object.entries(results)
    .filter(([name]) => name !== '_metadata')
    .map(([name, metrics]) => ({ name, ...metrics }))
    .sort((a, b) => b.f1 - a.f1);

  const bestModel = sortedModels[0];
  
  // Check if we have optimized models
  const hasOptimizedModels = Object.keys(results).some(key => 
    key.includes('Optimized') || key.includes('CatBoost')
  );

  return (
    <div className={styles.container}>
      <h1 className={styles.pageTitle}>Model Performance Results</h1>
      <p className={styles.pageDescription}>
        Classification Models - EPL Match Outcome Prediction
      </p>
      
      {hasOptimizedModels && (
        <div style={{ 
          marginBottom: '1.5rem', 
          padding: '1rem', 
          backgroundColor: '#d4edda', 
          borderRadius: '8px',
          border: '2px solid #28a745'
        }}>
          <strong>🚀 Optimized Models with Feature Selection!</strong> Using top 50 features (45% reduction) for better accuracy.
          Best model: {bestModel.name} with {(bestModel.accuracy * 100).toFixed(2)}% accuracy
        </div>
      )}

      {/* Performance Comparison Chart */}
      <div className={styles.card} style={{ marginBottom: '2rem' }}>
        <h2 className={styles.cardTitle}>Model Performance Comparison</h2>
        <p className={styles.cardDescription}>
          Comparison of accuracy, precision, recall, and F1-score across all models
        </p>
        
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" angle={-15} textAnchor="end" height={100} />
            <YAxis label={{ value: 'Score (%)', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Bar dataKey="accuracy" fill="#3498db" name="Accuracy" />
            <Bar dataKey="precision" fill="#2ecc71" name="Precision" />
            <Bar dataKey="recall" fill="#e74c3c" name="Recall" />
            <Bar dataKey="f1" fill="#f39c12" name="F1-Score" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Performance Table */}
      <div className={styles.card} style={{ marginBottom: '2rem' }}>
        <h2 className={styles.cardTitle}>Detailed Metrics</h2>
        <p className={styles.cardDescription}>
          Models ranked by F1-Score (best model highlighted in green)
        </p>
        
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Model</th>
              <th>Accuracy</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-Score</th>
            </tr>
          </thead>
          <tbody>
            {sortedModels.map((model, index) => (
              <tr 
                key={model.name}
                className={index === 0 ? styles.bestModel : ''}
              >
                <td>{index + 1}</td>
                <td>{model.name}</td>
                <td>{(model.accuracy * 100).toFixed(2)}%</td>
                <td>{(model.precision * 100).toFixed(2)}%</td>
                <td>{(model.recall * 100).toFixed(2)}%</td>
                <td>{(model.f1 * 100).toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>

        {bestModel && (
          <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#d4edda', borderRadius: '4px' }}>
            <strong>🏆 Best Model:</strong> {bestModel.name} with F1-Score of {(bestModel.f1 * 100).toFixed(2)}%
          </div>
        )}
      </div>

      {/* Confusion Matrices */}
      <h2 className={styles.cardTitle} style={{ marginTop: '2rem', marginBottom: '1rem' }}>
        Confusion Matrices
      </h2>
      <p className={styles.cardDescription} style={{ marginBottom: '1rem' }}>
        Detailed prediction breakdown for each model (Actual vs Predicted)
      </p>
      
      <div className={styles.grid}>
        {Object.keys(results)
          .filter(key => key !== '_metadata')
          .map((modelName) => {
            const filename = `cm_${modelName.toLowerCase().replace(/ /g, '_')}.png`;
            const imageUrl = getStaticUrl(filename);
            
            return (
              <ConfusionMatrix
                key={modelName}
                title={modelName}
                imageUrl={imageUrl}
              />
            );
          })}
      </div>
    </div>
  );
}
