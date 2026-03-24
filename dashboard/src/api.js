/**
 * API Service for EPL Dashboard
 * Handles all communication with the backend API
 */

const BASE_URL = 'http://localhost:3001';

/**
 * Fetch model evaluation results
 * @returns {Promise<Object|null>} Model results or null on error
 */
export async function fetchResults() {
  try {
    const response = await fetch(`${BASE_URL}/api/results`);
    
    if (!response.ok) {
      console.error('Failed to fetch results:', response.status);
      return null;
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching results:', error);
    return null;
  }
}

/**
 * Fetch selected features list
 * @returns {Promise<Array>} Array of feature names or empty array on error
 */
export async function fetchFeatures() {
  try {
    const response = await fetch(`${BASE_URL}/api/features`);
    
    if (!response.ok) {
      console.error('Failed to fetch features:', response.status);
      return [];
    }
    
    const data = await response.json();
    return data.features || [];
  } catch (error) {
    console.error('Error fetching features:', error);
    return [];
  }
}

/**
 * Fetch available charts
 * @returns {Promise<Array>} Array of chart objects or empty array on error
 */
export async function fetchCharts() {
  try {
    const response = await fetch(`${BASE_URL}/api/charts`);
    
    if (!response.ok) {
      console.error('Failed to fetch charts:', response.status);
      return [];
    }
    
    const data = await response.json();
    return data.charts || [];
  } catch (error) {
    console.error('Error fetching charts:', error);
    return [];
  }
}

/**
 * Get full URL for static files
 * @param {string} filename - Name of the file
 * @returns {string} Full URL to the file
 */
export function getStaticUrl(filename) {
  return `${BASE_URL}/output/${filename}`;
}

export { BASE_URL };
