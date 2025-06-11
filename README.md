# Credit Card Fraud Detection System

A machine learning-based fraud detection system that uses Random Forest and Decision Tree models to identify fraudulent credit card transactions.

## Project Structure

```
├── main.py                      # Core model training code
├── api.py                       # API implementation
├── test_api.py                  # API testing script
├── project_documentation.md     # Detailed project documentation
├── best_random_forest_random_model.joblib  # Best performing model
├── requirements.txt             # Project dependencies
└── README.md                    # Project overview
```

## Features

- **Model Training**:
  - Random Forest,Decision Tree models,Logestuc Regression, XGBoost and LightGBM
  - Multiple resampling techniques (SMOTE, ADASYN, Random)
  - Performance metrics tracking
  - Model comparison and selection

- **API Implementation**:
  - Real-time fraud detection
  - Risk assessment
  - Transaction analysis
  - Action recommendations

## Performance Metrics

### Best Model (Random Forest with Random Resampling)
- Accuracy: 99.94%
- Precision: 82.65%
- Recall: 82.65%
- F1-Score: 82.65%
- ROC AUC: 95.84%

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Running the API Server

1. Open a command prompt and navigate to the project directory:
   ```bash
   cd C:\Users\dell\OneDrive\Desktop\miniproject
   ```

2. Activate the virtual environment:
   ```bash
   venv\Scripts\activate
   ```

3. Start the API server:
   ```bash
   python api.py
   ```
   - The server will run on `http://0.0.0.0:8080`
   - Keep this command prompt window open

### Running the Test File

1. Open a new command prompt window
2. Navigate to the project directory:
   ```bash
   cd C:\Users\dell\OneDrive\Desktop\miniproject
   ```

3. Activate the virtual environment:
   ```bash
   venv\Scripts\activate
   ```

4. Run the test file:
   ```bash
   python test_api.py
   ```

### Important Notes:
- Make sure the API server is running before executing the test file
- Keep both command prompt windows open
- If you encounter port conflicts, you can modify the port in `api.py`
- Ensure all required packages are installed using `pip install -r requirements.txt`

## API Endpoints

### POST /detect
Analyzes a transaction for potential fraud.

**Input Format**:
```json
{
    "amount": float,
    "time": int,
    "location": string,
    "customer_location": string,
    "device_id": string,
    "customer_device_id": string,
    "ip_address": string,
    "customer_ip": string,
    "v1": float,
    "v2": float
}
```

**Output Format**:
```json
{
    "transaction_status": string,
    "risk_level": string,
    "detection_confidence": float,
    "analysis_result": {
        "customer_behavior": object,
        "location_analysis": object,
        "risk_factors": array,
        "suspicious_patterns": array,
        "time_analysis": object
    },
    "recommended_action": {
        "action": string,
        "notify_parties": array,
        "required_verification": array,
        "timeframe": string
    },
    "timestamp": string
}
```

## Dependencies

- Python 3.x
- Flask
- scikit-learn
- numpy
- joblib
- requests (for testing)

## Documentation

For detailed documentation, including:
- Model performance analysis
- API testing results
- System requirements
- Implementation details

Please refer to `project_documentation.md`

## Limitations

1. Model requires multiple risk factors to trigger high-risk alerts
2. Detection confidence is consistently 0.5
3. No authentication system implemented
4. No rate limiting for API
5. No caching mechanism

## Future Improvements

1. Implement authentication system
2. Add rate limiting
3. Implement caching
4. Add real-time model updates
5. Improve model calibration

## License

This project is licensed under the MIT License - see the LICENSE file for details. 