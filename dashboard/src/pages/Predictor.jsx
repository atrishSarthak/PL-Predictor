import { useState, useEffect } from 'react';
import { fetchFeatures, fetchResults } from '../api';
import styles from '../styles/layout.module.css';

/**
 * Predictor Page
 * Allows users to input feature values for match prediction
 */
export default function Predictor() {
  const [features, setFeatures] = useState([]);
  const [formData, setFormData] = useState({});
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [bestModel, setBestModel] = useState(null);

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    setLoading(true);
    
    // Load features
    const featuresList = await fetchFeatures();
    setFeatures(featuresList);
    
    // Initialize form data with zeros
    const initialData = {};
    featuresList.forEach(feature => {
      initialData[feature] = 0;
    });
    setFormData(initialData);
    
    // Load results to determine best model
    const results = await fetchResults();
    if (results) {
      const models = Object.entries(results).map(([name, metrics]) => ({
        name,
        ...metrics
      }));
      const best = models.sort((a, b) => b.f1 - a.f1)[0];
      setBestModel(best);
    }
    
    setLoading(false);
  }

  function handleInputChange(feature, value) {
    setFormData(prev => ({
      ...prev,
      [feature]: parseFloat(value) || 0
    }));
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setPredicting(true);
    
    // Simulate prediction (placeholder)
    // In a real implementation, this would send data to a prediction endpoint
    setTimeout(() => {
      // Mock prediction result
      const outcomes = ['Home Win', 'Draw', 'Away Win'];
      const randomOutcome = outcomes[Math.floor(Math.random() * outcomes.length)];
      
      setPrediction({
        outcome: randomOutcome,
        confidence: (Math.random() * 30 + 40).toFixed(2) // Random confidence between 40-70%
      });
      
      setPredicting(false);
    }, 1000);
  }

  if (loading) {
    return (
      <div className={styles.container}>
        <div className={styles.loading}>Loading predictor...</div>
      </div>
    );
  }

  if (features.length === 0) {
    return (
      <div className={styles.container}>
        <div className={styles.error}>
          No features available. Please run pipeline/4_feature_selection.py first.
        </div>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <h1 className={styles.pageTitle}>Match Outcome Predictor</h1>
      <p className={styles.pageDescription}>
        Enter feature values to predict match outcome (Home Win, Draw, or Away Win)
      </p>

      {bestModel && (
        <div className={styles.card} style={{ marginBottom: '2rem', background: 'linear-gradient(135deg, #fff9e6 0%, #fffef5 100%)' }}>
          <h3 style={{ fontWeight: 800, marginBottom: '0.5rem' }}>🤖 Prediction Model</h3>
          <p style={{ margin: '0.5rem 0' }}>
            <strong>Model:</strong> {bestModel.name}
          </p>
          <p style={{ margin: '0.5rem 0' }}>
            <strong>Accuracy:</strong> {(bestModel.accuracy * 100).toFixed(2)}%
          </p>
          <p style={{ margin: '0.5rem 0' }}>
            <strong>F1-Score:</strong> {(bestModel.f1 * 100).toFixed(2)}%
          </p>
        </div>
      )}

      <div className={styles.card}>
        <h2 className={styles.cardTitle}>Input Features</h2>
        <p className={styles.cardDescription}>
          Enter values for each feature. Note: Features should be scaled according to the training data.
        </p>

        <form onSubmit={handleSubmit} className={styles.form}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
            {features.map((feature) => (
              <div key={feature} className={styles.formGroup}>
                <label htmlFor={feature} className={styles.formLabel}>
                  {feature}
                </label>
                <input
                  type="number"
                  id={feature}
                  step="1"
                  value={formData[feature]}
                  onChange={(e) => handleInputChange(feature, e.target.value)}
                  className={styles.formInput}
                  required
                />
              </div>
            ))}
          </div>

          <button 
            type="submit" 
            className={styles.button}
            disabled={predicting}
          >
            {predicting ? 'Predicting...' : 'Predict Match Outcome'}
          </button>
        </form>

        {prediction && (
          <div className={styles.predictionResult}>
            <h3>Prediction Result</h3>
            <p><strong>Predicted Outcome:</strong> {prediction.outcome}</p>
            <p><strong>Confidence:</strong> {prediction.confidence}%</p>
            <p style={{ fontSize: '0.9rem', marginTop: '1rem', color: '#666' }}>
             In production, this would use the trained {bestModel?.name} model to make real predictions.
            </p>
          </div>
        )}
      </div>

      <div className={styles.card} style={{ marginTop: '2rem', background: 'linear-gradient(135deg, #fff9e6 0%, #fffef5 100%)' }}>
        <h3 style={{ fontWeight: 800, marginBottom: '0.5rem' }}>ℹ️ About This Predictor</h3>
        <p style={{ margin: '0.5rem 0' }}>
          This is a demonstration interface. To implement real predictions:
        </p>
        <ul style={{ marginLeft: '1.5rem', marginTop: '0.5rem' }}>
          <li>Create a prediction API endpoint in the backend</li>
          <li>Load the trained model (best performing model)</li>
          <li>Apply the same preprocessing and scaling as training</li>
          <li>Return prediction with confidence scores</li>
        </ul>
      </div>
    </div>
  );
}
