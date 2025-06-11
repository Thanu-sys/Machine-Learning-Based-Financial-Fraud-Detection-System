from flask import Flask, request, jsonify
import numpy as np
import joblib
from datetime import datetime
import logging
from typing import Dict, Any, List
import math

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
try:
    model = joblib.load('best_random_forest_random_model.joblib')  # Using Random Forest with Random Resampling as it's our best performer
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

def calculate_distance(loc1: str, loc2: str) -> float:
    """Calculate distance between two locations (simplified)"""
    # In real implementation, use geocoding API
    return 100.0  # Placeholder

def analyze_transaction(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze transaction for fraud detection"""
    
    # Amount Analysis
    amount = float(transaction['amount'])
    amount_risk = "HIGH" if amount > 10000 else "MEDIUM" if amount > 5000 else "LOW"
    
    # Time Analysis
    timestamp = int(transaction['time'])
    hour = datetime.fromtimestamp(timestamp).hour
    is_weekend = datetime.fromtimestamp(timestamp).weekday() >= 5
    time_risk = "HIGH" if (hour < 6 or hour > 22) else "LOW"
    
    # Location Analysis
    location = transaction['location']
    customer_location = transaction['customer_location']
    location_mismatch = location != customer_location
    distance = calculate_distance(location, customer_location)
    location_risk = "HIGH" if distance > 1000 else "MEDIUM" if distance > 100 else "LOW"
    
    # Device Analysis
    device_mismatch = transaction['device_id'] != transaction['customer_device_id']
    device_risk = "HIGH" if device_mismatch else "LOW"
    
    # IP Analysis
    ip_mismatch = transaction['ip_address'] != transaction['customer_ip']
    ip_risk = "HIGH" if ip_mismatch else "LOW"
    
    # Prepare features for model
    features = np.array([[
        amount,
        timestamp,
        float(transaction['v1']),
        float(transaction['v2'])
    ]])
    
    # Get model prediction
    try:
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
    except Exception as e:
        logger.error(f"Error in model prediction: {str(e)}")
        prediction = 0
        probability = 0.5
    
    # Determine risk level
    risk_factors = []
    if amount_risk == "HIGH":
        risk_factors.append({"factor": "amount", "value": amount, "threshold": 10000})
    if location_risk == "HIGH":
        risk_factors.append({"factor": "location", "value": f"{location} vs {customer_location}"})
    if device_risk == "HIGH":
        risk_factors.append({"factor": "device", "value": "Different from customer's usual device"})
    if ip_risk == "HIGH":
        risk_factors.append({"factor": "ip", "value": "Different from customer's usual IP"})
    
    # Calculate overall risk level
    risk_level = "HIGH" if (probability > 0.8 or len(risk_factors) >= 3) else \
                "MEDIUM" if (probability > 0.5 or len(risk_factors) >= 2) else "LOW"
    
    # Prepare suspicious patterns
    suspicious_patterns = []
    if amount_risk == "HIGH":
        suspicious_patterns.append("High transaction amount")
    if location_risk == "HIGH":
        suspicious_patterns.append("Location mismatch")
    if device_risk == "HIGH":
        suspicious_patterns.append("Device mismatch")
    if ip_risk == "HIGH":
        suspicious_patterns.append("IP address mismatch")
    
    # Determine recommended action
    if risk_level == "HIGH":
        action = "BLOCK_AND_NOTIFY"
        notify_parties = ["customer", "fraud_team", "security_team"]
        required_verification = ["customer_verification", "manual_review"]
        timeframe = "immediate"
    elif risk_level == "MEDIUM":
        action = "REVIEW"
        notify_parties = ["fraud_team"]
        required_verification = ["customer_verification"]
        timeframe = "within_24h"
    else:
        action = "PROCESS"
        notify_parties = []
        required_verification = []
        timeframe = "normal"
    
    return {
        "transaction_status": "FRAUD_DETECTED" if prediction == 1 else "LEGITIMATE",
        "risk_level": risk_level,
        "detection_confidence": float(probability),
        "analysis_result": {
            "suspicious_patterns": suspicious_patterns,
            "risk_factors": risk_factors,
            "customer_behavior": {
                "total_transactions_24h": 0,  # Would be calculated from database
                "total_amount_24h": 0.0,      # Would be calculated from database
                "average_transaction_amount": 0.0  # Would be calculated from database
            },
            "location_analysis": {
                "current_location": location,
                "customer_location": customer_location,
                "distance": distance
            },
            "time_analysis": {
                "hour": hour,
                "day": datetime.fromtimestamp(timestamp).weekday(),
                "is_weekend": is_weekend
            }
        },
        "recommended_action": {
            "action": action,
            "notify_parties": notify_parties,
            "required_verification": required_verification,
            "timeframe": timeframe
        },
        "timestamp": datetime.now().isoformat()
    }

@app.route('/detect', methods=['POST'])
def detect_fraud():
    """API endpoint for fraud detection"""
    try:
        transaction = request.json
        result = analyze_transaction(transaction)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0') 