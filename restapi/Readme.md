# Log Analysis REST API

A Flask-based REST API application that processes CSV log files and generates statistical reports. The application is containerized using Docker and can be deployed using Docker Compose.

## Features

- **CSV File Processing**: Upload CSV files containing machine log data
- **Statistical Analysis**: Calculates min, max, mean, standard deviation, and kurtosis for numerical columns
- **REST API**: Simple HTTP POST endpoint for file uploads
- **Dockerized**: Easy deployment using Docker and Docker Compose
- **Error Handling**: Comprehensive error handling and logging
- **Health Check**: Built-in health check endpoint

## Project Structure

```
restapi/
├── Dockerfile
├── docker-compose.yml
├── my_rest_api.py
├── requirements.txt
└── README.md
```

## API Endpoints

### POST /generateReport
- **Description**: Upload a CSV file and receive statistical analysis
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Parameters**: 
  - `file`: CSV file (form field name)
- **Response**: JSON containing statistical analysis

### GET /health
- **Description**: Health check endpoint
- **Method**: GET
- **Response**: JSON status message

### GET /
- **Description**: API information and available endpoints
- **Method**: GET
- **Response**: JSON with API details

## Statistical Metrics Calculated

For each numerical column in the uploaded CSV:
- **Min**: Minimum value
- **Max**: Maximum value  
- **Mean**: Average value
- **Standard Deviation**: Measure of data spread
- **Kurtosis**: Measure of data distribution shape (using Fisher's definition)

## Methodology

### Data Processing Pipeline
1. **File Validation**: Ensures uploaded file is a valid CSV
2. **Data Loading**: Uses pandas to read CSV into DataFrame
3. **Column Detection**: Automatically identifies numerical columns
4. **Statistical Calculation**: Computes statistics using pandas and scipy
5. **Response Formatting**: Returns results in structured JSON format

### Technology Stack
- **Flask**: Lightweight web framework for REST API
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **SciPy**: Statistical functions (kurtosis calculation)
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration

## Deployment

### Prerequisites
- Docker and Docker Compose installed
- Port 10000 available

### Quick Start

1. **Clone/Download the project files**

2. **Navigate to project directory**
   ```bash
   cd restapi
   ```

3. **Deploy using Docker Compose**
   ```bash
   docker-compose up -d
   ```

4. **API will be available at**: `http://localhost:10000`

### Alternative Deployment Methods

**Build and run manually:**
```bash
docker build -t log-analysis-api .

docker run -p 10000:10000 log-analysis-api
```

**See live container:**
```
docker ps - a
```
result example: 


## Usage Example

### Python Client
```python
import requests

url = 'http://localhost:10000/generateReport'
file_path = 'M6.csv'

# Upload file
with open(file_path, 'rb') as file:
    files = {'file': (file_path, file)}
    response = requests.post(url, files=files)

if response.status_code == 200:
    print("File uploaded successfully.")
    print(response.json())
else:
    print(f"Failed to upload file. Status code: {response.status_code}")
    print(response.text)
```

### cURL Example
```bash
curl -X POST -F "file=@M6.csv" http://localhost:10000/generateReport
```

## Expected Response Format

```json
{
  "status": "success",
  "message": "Report generated successfully",
  "total_rows": 23,
  "numerical_columns": 4,
  "statistics": {
    "Temperature": {
      "min": 55.0,
      "max": 65.0,
      "mean": 58.95652173913043,
      "std": 3.1234567890123456,
      "kurtosis": -0.5678901234567890
    },
    "CPU Usage": {
      "min": 10.38520516,
      "max": 35.04076087,
      "mean": 15.123456789012345,
      "std": 6.789012345678901,
      "kurtosis": 2.3456789012345678
    },
    "RAM Usage": {
      "min": 52.27745445,
      "max": 54.97228704,
      "mean": 53.456789012345678,
      "std": 0.789012345678901,
      "kurtosis": -1.2345678901234567
    },
    "Storage Usage": {
      "min": 3.0,
      "max": 7.0,
      "mean": 4.130434782608696,
      "std": 1.8612345678901234,
      "kurtosis": 0.1234567890123456
    }
  }
}
```

## Error Handling

The API includes comprehensive error handling for:
- Missing file in request
- Invalid file format (non-CSV)
- Empty files
- Corrupted CSV data
- No numerical columns found
- Statistical calculation errors

## Logging

The application includes structured logging for:
- File upload events
- Data processing steps
- Error tracking
- Performance monitoring

## Development Notes

- The API automatically detects numerical columns using pandas data types
- Kurtosis is calculated using Fisher's definition (excess kurtosis)
- NaN values are automatically handled in statistical calculations
- The application runs in production mode when deployed via Docker

## Health Monitoring

The application includes a health check endpoint that can be used for:
- Container orchestration health checks
- Load balancer health monitoring  
- Application status verification