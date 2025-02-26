# Model Serving with Flask

This module provides a Flask-based API for serving the Brent Oil Price Prediction model. It allows users to analyze time series data through HTTP requests.

## Overview
The Flask app handles CORS, logging, data loading, and serves analysis results via endpoints.

## File Structure
- **serve_model.py**: Main Flask app
  - Initializes Flask, enables CORS, and configures logging.
  - Loads Brent oil prices data and sets the 'Date' column as the index.
  - **/analyze**: Endpoint to analyze time series data and return components (moving average, trend, seasonality, residuals).

## Setup and Usage

### Prerequisites
- Python 3.x
- Flask, pandas, statsmodels, flask_cors

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/Brent-Oil-Price-Prediction.git
   cd Brent-Oil-Price-Prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Flask App
1. Navigate to the directory:
   ```bash
   cd api
   ```

2. Run the app:
   ```bash
   flask run
   ```

### Endpoints
- **GET /analyze**: Returns moving average, trend, seasonality, and residual components.

### Logging
Logs are stored in the `logs` directory, capturing request details.

## Example Request
```bash
curl -X GET "http://127.0.0.1:5000/analyze"
```

## Conclusion
This module enables easy integration of the Brent Oil Price Prediction model with other services through a Flask API.