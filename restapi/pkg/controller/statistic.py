from flask import Flask, jsonify
from pkg.service.statistic import statistic_bp

app = Flask(__name__)
app.register_blueprint(statistic_bp)

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