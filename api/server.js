/**
 * EPL Match Prediction API Server
 * Serves pipeline outputs and provides REST API endpoints
 */

const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

// Initialize Express app
const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Configuration
const PORT = process.env.PORT || 5000;
const OUTPUT_DIR = path.join(__dirname, '..', 'pipeline', 'output');

// Log configuration on startup
console.log('='.repeat(60));
console.log('EPL MATCH PREDICTION API SERVER');
console.log('='.repeat(60));
console.log(`Output Directory: ${OUTPUT_DIR}`);
console.log('='.repeat(60));

// STEP 2: Static File Serving
// Serve static files from pipeline/output directory
app.use('/output', express.static(OUTPUT_DIR));
console.log('✓ Static file serving enabled at /output');

// STEP 3: API Routes

/**
 * GET /api/health
 * Health check endpoint
 */
app.get('/api/health', (req, res) => {
    console.log('GET /api/health');
    res.json({
        status: 'ok',
        timestamp: new Date(),
        message: 'EPL API Server is running'
    });
});

/**
 * GET /api/results
 * Returns model evaluation results (prioritizes CatBoost results if available)
 */
app.get('/api/results', (req, res) => {
    console.log('GET /api/results');
    
    try {
        // Try CatBoost results first (newer, better models)
        const catboostPath = path.join(OUTPUT_DIR, 'catboost_results.json');
        const baselinePath = path.join(OUTPUT_DIR, 'model_results.json');
        
        let filePath, resultsType;
        
        if (fs.existsSync(catboostPath)) {
            filePath = catboostPath;
            resultsType = 'CatBoost Enhanced Models';
            console.log(`  ✓ Using CatBoost results (52.67% accuracy)`);
        } else if (fs.existsSync(baselinePath)) {
            filePath = baselinePath;
            resultsType = 'Baseline Models';
            console.log(`  ✓ Using baseline results`);
        } else {
            console.log(`  ✗ No results files found`);
            return res.status(404).json({
                error: 'File not found',
                message: 'Please run pipeline/6_evaluate_catboost.py or pipeline/6_evaluate.py first'
            });
        }
        
        // Read and parse JSON file
        const fileContent = fs.readFileSync(filePath, 'utf8');
        const results = JSON.parse(fileContent);
        
        console.log(`  ✓ Successfully loaded results`);
        res.json({
            ...results,
            _metadata: {
                type: resultsType,
                source: path.basename(filePath)
            }
        });
        
    } catch (error) {
        console.error(`  ✗ Error reading results: ${error.message}`);
        res.status(500).json({
            error: 'Server error',
            message: error.message
        });
    }
});

/**
 * GET /api/features
 * Returns selected features from selected_features.txt
 */
app.get('/api/features', (req, res) => {
    console.log('GET /api/features');
    
    try {
        // Build file path using path.join
        const filePath = path.join(OUTPUT_DIR, 'selected_features.txt');
        
        // Check if file exists
        if (!fs.existsSync(filePath)) {
            console.log(`  ✗ File not found: ${filePath}`);
            return res.status(404).json({
                error: 'File not found',
                file: 'selected_features.txt',
                message: 'Please run pipeline/4_feature_selection.py first'
            });
        }
        
        // Read file
        const fileContent = fs.readFileSync(filePath, 'utf8');
        
        // Split by newline and remove empty lines
        const features = fileContent
            .split('\n')
            .map(line => line.trim())
            .filter(line => line.length > 0);
        
        console.log(`  ✓ Successfully loaded ${features.length} features`);
        res.json({
            features: features,
            count: features.length
        });
        
    } catch (error) {
        console.error(`  ✗ Error reading features: ${error.message}`);
        res.status(500).json({
            error: 'Server error',
            message: error.message
        });
    }
});

/**
 * GET /api/charts
 * Returns list of available chart files
 */
app.get('/api/charts', (req, res) => {
    console.log('GET /api/charts');
    
    try {
        // Check if output directory exists
        if (!fs.existsSync(OUTPUT_DIR)) {
            return res.status(404).json({
                error: 'Output directory not found',
                message: 'Please run the pipeline scripts first'
            });
        }
        
        // Read directory contents
        const files = fs.readdirSync(OUTPUT_DIR);
        
        // Filter for PNG files
        const charts = files
            .filter(file => file.endsWith('.png'))
            .map(file => ({
                name: file,
                url: `/output/${file}`
            }));
        
        console.log(`  ✓ Found ${charts.length} chart files`);
        res.json({
            charts: charts,
            count: charts.length
        });
        
    } catch (error) {
        console.error(`  ✗ Error listing charts: ${error.message}`);
        res.status(500).json({
            error: 'Server error',
            message: error.message
        });
    }
});

/**
 * GET /api/models
 * Returns list of trained model files
 */
app.get('/api/models', (req, res) => {
    console.log('GET /api/models');
    
    try {
        // Check if output directory exists
        if (!fs.existsSync(OUTPUT_DIR)) {
            return res.status(404).json({
                error: 'Output directory not found',
                message: 'Please run the pipeline scripts first'
            });
        }
        
        // Read directory contents
        const files = fs.readdirSync(OUTPUT_DIR);
        
        // Filter for model files (.pkl)
        const models = files
            .filter(file => file.startsWith('model_') && file.endsWith('.pkl'))
            .map(file => ({
                name: file.replace('model_', '').replace('.pkl', '').replace(/_/g, ' '),
                filename: file
            }));
        
        console.log(`  ✓ Found ${models.length} model files`);
        res.json({
            models: models,
            count: models.length
        });
        
    } catch (error) {
        console.error(`  ✗ Error listing models: ${error.message}`);
        res.status(500).json({
            error: 'Server error',
            message: error.message
        });
    }
});

/**
 * GET /
 * Root endpoint - API documentation
 */
app.get('/', (req, res) => {
    res.json({
        message: 'EPL Match Prediction API',
        version: '1.0.0',
        endpoints: {
            health: 'GET /api/health',
            results: 'GET /api/results',
            features: 'GET /api/features',
            charts: 'GET /api/charts',
            models: 'GET /api/models',
            staticFiles: 'GET /output/<filename>'
        },
        documentation: 'Visit each endpoint for data'
    });
});

// STEP 4: Error Handling
// 404 handler for undefined routes
app.use((req, res) => {
    console.log(`  ✗ 404 Not Found: ${req.method} ${req.url}`);
    res.status(404).json({
        error: 'Not Found',
        message: `Route ${req.method} ${req.url} does not exist`,
        availableEndpoints: [
            'GET /',
            'GET /api/health',
            'GET /api/results',
            'GET /api/features',
            'GET /api/charts',
            'GET /api/models',
            'GET /output/<filename>'
        ]
    });
});

// Global error handler
app.use((err, req, res, next) => {
    console.error('  ✗ Server Error:', err.message);
    res.status(500).json({
        error: 'Internal Server Error',
        message: err.message
    });
});

// STEP 5: Start Server
app.listen(PORT, () => {
    console.log('\n' + '='.repeat(60));
    console.log('✓ SERVER STARTED SUCCESSFULLY');
    console.log('='.repeat(60));
    console.log(`Server running on http://localhost:${PORT}`);
    console.log('\nAvailable Endpoints:');
    console.log(`  - http://localhost:${PORT}/`);
    console.log(`  - http://localhost:${PORT}/api/health`);
    console.log(`  - http://localhost:${PORT}/api/results`);
    console.log(`  - http://localhost:${PORT}/api/features`);
    console.log(`  - http://localhost:${PORT}/api/charts`);
    console.log(`  - http://localhost:${PORT}/api/models`);
    console.log(`  - http://localhost:${PORT}/output/<filename>`);
    console.log('\nExample Static File:');
    console.log(`  - http://localhost:${PORT}/output/model_comparison.png`);
    console.log('='.repeat(60) + '\n');
});
