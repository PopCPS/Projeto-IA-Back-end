from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np

app = Flask(__name__)

model = xgb.XGBClassifier()
model.load_model("xgboost_model.json")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        
        if not input_data or 'features' not in input_data:
            return jsonify({"error": "Invalid input. Please provide 'features' as a list of numbers."}), 400
        
        features = np.array(input_data['features']).reshape(1, -1) 
        
        if features.shape[1] != model.n_features_in_:
            return jsonify({"error": f"Expected {model.n_features_in_} features, but got {features.shape[1]}."}), 400
        
        prediction = model.predict(features)[0]
        prediction_probabilities = model.predict_proba(features).tolist()
        
        return jsonify({
            "prediction": int(prediction),
            "probabilities": prediction_probabilities
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
