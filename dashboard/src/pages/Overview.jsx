import { useState } from 'react';
import { getStaticUrl } from '../api';
import styles from '../styles/layout.module.css';

/**
 * Overview Page
 * Displays exploratory data analysis visualizations
 */
export default function Overview() {
  const charts = [
    {
      filename: 'outcome_distribution.png',
      title: 'Match Outcome Distribution',
      description: 'Distribution of Home Wins, Draws, and Away Wins across all matches'
    },
    {
      filename: 'goals_boxplot.png',
      title: 'Goals Analysis',
      description: 'Box plot comparing home goals vs away goals distribution'
    },
    {
      filename: 'season_trends.png',
      title: 'Season-wise Outcome Trends',
      description: 'Trends in match outcomes across different seasons'
    },
    {
      filename: 'top_teams_wins.png',
      title: 'Top Teams by Wins',
      description: 'Top 20 teams ranked by total wins (home + away)'
    },
    {
      filename: 'shots_analysis.png',
      title: 'Shots Analysis',
      description: 'Analysis of shots, shots on target, and shot accuracy'
    },
    {
      filename: 'discipline_analysis.png',
      title: 'Discipline Analysis',
      description: 'Fouls and yellow cards statistics'
    },
    {
      filename: 'halftime_analysis.png',
      title: 'Half-Time Analysis',
      description: 'Half-time vs full-time results and second half performance'
    },
    {
      filename: 'home_advantage_analysis.png',
      title: 'Home Advantage Analysis',
      description: 'Analysis of home advantage and goal differences'
    }
  ];

  return (
    <div className={styles.container}>
      <h1 className={styles.pageTitle}>Data Overview</h1>
      <p className={styles.pageDescription}>
        Exploratory data analysis and visualizations of EPL match data from 2000-2025
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
              Run pipeline/1_explore.py to generate this chart
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
