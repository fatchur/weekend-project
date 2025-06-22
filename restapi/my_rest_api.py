from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
import io
import logging

app = Flask(__name__)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_statistics(data):
    stats = {}
    
    numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    for column in numerical_columns:
        clean_data = data[column].dropna()
        
        if len(clean_data) > 0:
            stats[column] = {
                'min': float(clean_data.min()),
                'max': float(clean_data.max()),
                'mean': float(clean_data.mean()),
                'std': float(clean_data.std()),
                'kurtosis': float(kurtosis(clean_data, fisher=True))  # Fisher=True for excess kurtosis
            }
        else:
            stats[column] = {
                'min': None,
                'max': None,
                'mean': None,
                'std': None,
                'kurtosis': None
            }
    
    return stats

@app.route('/generateReport', methods=['POST'])
def generate_report():
    """
    Endpoint to generate statistical report from uploaded CSV file
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'File must be a CSV file'}), 400
        
        try:
            file_content = file.read()
            csv_string = file_content.decode('utf-8')
            data = pd.read_csv(io.StringIO(csv_string))
            
            logger.info(f"Successfully loaded CSV with {len(data)} rows and {len(data.columns)} columns")
            logger.info(f"Columns: {list(data.columns)}")
            
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400
        
        if data.empty:
            return jsonify({'error': 'CSV file is empty'}), 400
        
        try:
            statistics = calculate_statistics(data)
            
            if not statistics:
                return jsonify({'error': 'No numerical columns found in the CSV file'}), 400
            
            response = {
                'status': 'success',
                'message': 'Report generated successfully',
                'total_rows': len(data),
                'numerical_columns': len(statistics),
                'statistics': statistics
            }
            
            logger.info(f"Successfully generated report for {len(statistics)} numerical columns")
            return jsonify(response), 200
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return jsonify({'error': f'Error calculating statistics: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'API is running'}), 200

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Log Analysis REST API',
        'version': '1.0.0',
        'endpoints': {
            '/generateReport': 'POST - Upload CSV file and get statistical report',
            '/health': 'GET - Health check',
            '/': 'GET - API information'
        }
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)