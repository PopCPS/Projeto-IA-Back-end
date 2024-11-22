import pytest
import json
import numpy as np
from app import app 

@pytest.fixture
def client():
    """Fixture to provide a test client for the Flask app."""
    with app.test_client() as client:
        yield client

def test_predict_valid_input(client):
    """Test the predict endpoint with valid input."""
    input_data = {
        "day_of_class": "Mon",
        "time_of_class": "AM",
        "category": "Yoga",
        "months_as_member": 12,
        "weight": 65.5,
        "days_before_class": 3
    }

    response = client.post('/predict',
                           data=json.dumps(input_data),
                           content_type='application/json')

    assert response.status_code == 200

    response_data = response.get_json()

    print("Response Data:", response_data)

    assert "prediction" in response_data
    assert isinstance(response_data["prediction"], int)

    assert "probabilities" in response_data
    assert isinstance(response_data["probabilities"], list)

    assert isinstance(response_data["probabilities"][0], list)  
    assert len(response_data["probabilities"][0]) == 2  

    print("Types in Probabilities:", [type(prob) for prob in response_data["probabilities"][0]])

    assert all(isinstance(prob, float) for prob in response_data["probabilities"][0])

    assert 0 <= response_data["prediction"] <= 1 
    assert 0 <= response_data["probabilities"][0][0] <= 1  
    assert 0 <= response_data["probabilities"][0][1] <= 1
    
def test_predict_invalid_input(client):
    """Test the predict endpoint with invalid input."""
    incomplete_input_data = {
        "day_of_class": "Mon",
        "time_of_class": "PM"
    }
    
    response = client.post('/predict', 
                           data=json.dumps(incomplete_input_data), 
                           content_type='application/json')
    
    assert response.status_code == 400
    response_data = response.get_json()
    
    assert "error" in response_data
    assert "Invalid input" in response_data["error"]

def test_predict_server_error(client, mocker):
    """Test the predict endpoint when a server error occurs."""
    input_data = {
        "day_of_class": "Mon",
        "time_of_class": "PM",
        "category": "Yoga",
        "months_as_member": 12,
        "weight": 65.5,
        "days_before_class": 3
    }

    mocker.patch('app.model.predict', side_effect=Exception("Mocked Exception"))

    response = client.post('/predict', 
                           data=json.dumps(input_data), 
                           content_type='application/json')
    
    assert response.status_code == 500
    response_data = response.get_json()
    
    assert "error" in response_data
    assert response_data["error"] == "Mocked Exception"
