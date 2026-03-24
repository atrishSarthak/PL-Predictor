import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import Overview from './pages/Overview';
import TeamAnalysis from './pages/TeamAnalysis';
import ModelResults from './pages/ModelResults';
import Predictor from './pages/Predictor';
import styles from './styles/layout.module.css';

/**
 * Main App Component
 * Handles routing and navigation
 */
export default function App() {
  return (
    <BrowserRouter>
      <div>
        {/* Navigation Bar */}
        <nav className={styles.navbar}>
          <ul className={styles.navList}>
            <li>⚽ EPL Match Predictor</li>
            <li>
              <NavLink 
                to="/" 
                className={({ isActive }) => 
                  isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink
                }
              >
                Overview
              </NavLink>
            </li>
            <li>
              <NavLink 
                to="/team" 
                className={({ isActive }) => 
                  isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink
                }
              >
                Team Analysis
              </NavLink>
            </li>
            <li>
              <NavLink 
                to="/models" 
                className={({ isActive }) => 
                  isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink
                }
              >
                Model Results
              </NavLink>
            </li>
            <li>
              <NavLink 
                to="/predict" 
                className={({ isActive }) => 
                  isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink
                }
              >
                Predictor
              </NavLink>
            </li>
          </ul>
        </nav>

        {/* Routes */}
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/team" element={<TeamAnalysis />} />
          <Route path="/models" element={<ModelResults />} />
          <Route path="/predict" element={<Predictor />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
