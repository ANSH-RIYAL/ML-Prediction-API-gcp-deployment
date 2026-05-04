from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
 
from inference_lib import (
    get_predictions_from_user_dict,
    get_optimal_value_combination,
    transform_user_data
)
 
config_file_path = './configs/config.json'
with open(config_file_path) as f:
    config = json.load(f)
 
app = Flask(__name__)
CORS(app)
 
 
@app.route('/health', methods=['GET'])
@app.route('/test/', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'Lifespan Prediction API'})
 
 
@app.route('/prediction_model_test/', methods=['POST'])
def prediction_model_test():
    """
    Main endpoint called by the chatbot backend.
    Accepts JSON with health variables, returns lifespan + disease risk predictions.
    """
    user_data = request.json
    user_data = transform_user_data(user_data)
    preds = get_predictions_from_user_dict(user_data)
    predictions_dict = preds
    return jsonify({k: float(v) for k, v in predictions_dict.items()})
 
 
@app.route('/optimization_model_test/', methods=['POST'])
def optimization_model_test():
    """
    Returns optimal variable values for maximum lifespan.
    Accepts JSON with current user data + optional constraints.
    """
    user_data = request.json
 
    all_combos = list(user_data.items())
    for k, v in all_combos:
        if type(user_data[k]) == type(None):
            del user_data[k]
 
    age = user_data['age']
    gender = user_data['gender']
 
    gender_str_to_dict = config['gender_str_to_dict']
 
    if gender in ['male', 'female']:
        gender = gender_str_to_dict[gender]
 
    optimal_values = get_optimal_value_combination(age, gender)
 
    optimal_values['smoking_status'] = 0
    optimal_values['smoking_frequency'] = 0
    optimal_values['calorie_restriction'] = 1
 
    if user_data.get('smoking_status', 0) != 0:
        optimal_values['smoking_status'] = 1
 
    preds = get_predictions_from_user_dict(optimal_values)
 
    for k, v in preds.items():
        if k not in ['smoking_status', 'smoking_frequency']:
            optimal_values[k] = v
 
    if 'stress_level' in optimal_values:
        del optimal_values['stress_level']
 
    return jsonify(optimal_values)
 
 
@app.route('/lifespan_calculator_test/', methods=['POST'])
def lifespan_calculator_test():
    """
    Returns lifespan prediction + optimal lifespan for comparison.
    Accepts JSON with user health variables.
    """
    user_data = request.json
    user_data = transform_user_data(user_data)
 
    preds = get_predictions_from_user_dict(user_data)
    predictions_dict = preds
 
    age = user_data['age']
    gender = user_data['gender']
 
    gender_str_to_dict = config['gender_str_to_dict']
 
    if gender in ['male', 'female']:
        gender = gender_str_to_dict[gender]
 
    optimal_values = get_optimal_value_combination(age, gender)
 
    optimal_values['smoking_status'] = 0
    optimal_values['smoking_frequency'] = 0
    optimal_values['calorie_restriction'] = 1
 
    if user_data.get('smoking_status', 0) != 0:
        optimal_values['smoking_status'] = 1
 
    preds = get_predictions_from_user_dict(optimal_values)
    predictions_dict['optimal_lifespan'] = preds['all_cause_mortality_predicted_lifespan']
 
    return jsonify({k: float(v) for k, v in predictions_dict.items()})
 
 
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
 