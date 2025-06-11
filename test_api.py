import requests
import json

# API endpoint
url = "http://localhost:8080/detect"

# Test cases with different risk levels
test_cases = [
    {
        "name": "Normal Transaction",
        "data": {
            "amount": 1000,
            "time": 1620000000,
            "location": "New York",
            "customer_location": "Boston",
            "device_id": "device123",
            "customer_device_id": "device123",
            "ip_address": "192.168.1.1",
            "customer_ip": "192.168.1.1",
            "v1": 0.5,
            "v2": 0.3
        }
    },
    {
        "name": "High Amount Transaction",
        "data": {
            "amount": 15000,
            "time": 1620000000,
            "location": "New York",
            "customer_location": "Boston",
            "device_id": "device123",
            "customer_device_id": "device123",
            "ip_address": "192.168.1.1",
            "customer_ip": "192.168.1.1",
            "v1": 0.5,
            "v2": 0.3
        }
    },
    {
        "name": "Different Device Transaction",
        "data": {
            "amount": 1000,
            "time": 1620000000,
            "location": "New York",
            "customer_location": "Boston",
            "device_id": "device123",
            "customer_device_id": "device456",
            "ip_address": "192.168.1.1",
            "customer_ip": "192.168.1.1",
            "v1": 0.5,
            "v2": 0.3
        }
    },
    {
        "name": "Different IP Transaction",
        "data": {
            "amount": 1000,
            "time": 1620000000,
            "location": "New York",
            "customer_location": "Boston",
            "device_id": "device123",
            "customer_device_id": "device123",
            "ip_address": "192.168.1.1",
            "customer_ip": "10.0.0.1",
            "v1": 0.5,
            "v2": 0.3
        }
    },
    {
        "name": "Late Night Transaction",
        "data": {
            "amount": 1000,
            "time": 1620000000,  # This will be converted to 5 AM
            "location": "New York",
            "customer_location": "Boston",
            "device_id": "device123",
            "customer_device_id": "device123",
            "ip_address": "192.168.1.1",
            "customer_ip": "192.168.1.1",
            "v1": 0.5,
            "v2": 0.3
        }
    },
    {
        "name": "Multiple Risk Factors",
        "data": {
            "amount": 15000,
            "time": 1620000000,
            "location": "Tokyo",
            "customer_location": "New York",
            "device_id": "device123",
            "customer_device_id": "device456",
            "ip_address": "192.168.1.1",
            "customer_ip": "10.0.0.1",
            "v1": 0.5,
            "v2": 0.3
        }
    }
]

# Send POST requests for each test case
for test_case in test_cases:
    print(f"\n{'='*50}")
    print(f"Testing: {test_case['name']}")
    print(f"{'='*50}")
    
    try:
        response = requests.post(url, json=test_case['data'])
        print("\nStatus Code:", response.status_code)
        print("\nResponse:", json.dumps(response.json(), indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print(f"\n{'-'*50}") 