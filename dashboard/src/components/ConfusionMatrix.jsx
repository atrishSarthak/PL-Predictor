import { useState } from 'react';
import styles from '../styles/layout.module.css';

/**
 * ConfusionMatrix Component
 * Displays a confusion matrix image with fallback handling
 */
export default function ConfusionMatrix({ title, imageUrl }) {
  const [imageError, setImageError] = useState(false);

  const handleImageError = () => {
    setImageError(true);
  };

  return (
    <div className={styles.card}>
      <h3 className={styles.cardTitle}>{title}</h3>
      
      <div className={styles.imageContainer}>
        {!imageError ? (
          <img
            src={imageUrl}
            alt={`Confusion Matrix for ${title}`}
            className={styles.chartImage}
            onError={handleImageError}
          />
        ) : (
          <div className={styles.imageFallback}>
            <p>⚠️ Confusion matrix image not available</p>
            <p style={{ fontSize: '0.9rem', marginTop: '0.5rem' }}>
              File: {imageUrl.split('/').pop()}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
