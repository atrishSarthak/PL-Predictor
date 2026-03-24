import { useState } from 'react';
import { getStaticUrl } from '../api';
import styles from '../styles/layout.module.css';

/**
 * Team Analysis Page
 * Displays feature analysis and class distribution charts
 */
export default function TeamAnalysis() {
  const charts = [
    {
      filename: 'feature_importance.png',
      title: 'Feature Importance',
      description: 'Random Forest feature importance ranking for match prediction'
    },
    {
      filename: 'feature_correlations.png',
      title: 'Feature Correlations',
      description: 'Correlation of features with match outcome (exploratory)'
    },
    {
      filename: 'class_distribution_before.png',
      title: 'Class Distribution (Before Balancing)',
      description: 'Original distribution of match outcomes in the dataset'
    },
    {
      filename: 'class_distribution_after.png',
      title: 'Class Distribution (After Balancing)',
      description: 'Distribution after applying SMOTE balancing (if applied)'
    }
  ];

  return (
    <div className={styles.container}>
      <h1 className={styles.pageTitle}>Team & Feature Analysis</h1>
      <p className={styles.pageDescription}>
        Feature importance, correlations, and class distribution analysis
      </p>

      <div className={styles.grid}>
        {charts.map((chart) => (
          <ChartCard key={chart.filename} {...chart} />
        ))}
      </div>
    </div>
  );
}

function ChartCard({ filename, title, description }) {
  const [imageError, setImageError] = useState(false);
  const imageUrl = getStaticUrl(filename);

  return (
    <div className={styles.card}>
      <h2 className={styles.cardTitle}>{title}</h2>
      <p className={styles.cardDescription}>{description}</p>
      
      <div className={styles.imageContainer}>
        {!imageError ? (
          <img
            src={imageUrl}
            alt={title}
            className={styles.chartImage}
            onError={() => setImageError(true)}
          />
        ) : (
          <div className={styles.imageFallback}>
            <p>⚠️ Chart not available</p>
            <p style={{ fontSize: '0.9rem', marginTop: '0.5rem' }}>
              {filename.includes('after') 
                ? 'This chart is only generated if SMOTE was applied'
                : `Run the pipeline scripts to generate this chart`}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
