from flask import Flask, request, jsonify
import numpy as np
import xgboost as xgb
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('xgboost_model.pkl')

one_hot_encoder = joblib.load('one_hot_encoder.pkl')

categorical_features = ['day_of_class', 'time_of_class', 'category']
numerical_features = ['months_as_member', 'weight', 'days_before_class']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        if not input_data or not all(key in input_data for key in (categorical_features + numerical_features)):
            return jsonify({"error": f"Invalid input. Please provide all the following features: {categorical_features + numerical_features}"}), 400
        
        data = pd.DataFrame([input_data])

        numeric_data = data[numerical_features].values
        categorical_data = one_hot_encoder.transform(data[categorical_features])

        processed_features = np.hstack((numeric_data, categorical_data))

        prediction = model.predict(processed_features)[0]
        probabilities = model.predict_proba(processed_features)

        return jsonify({
            "prediction": int(prediction > 0.5),
            "probabilities": probabilities.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
