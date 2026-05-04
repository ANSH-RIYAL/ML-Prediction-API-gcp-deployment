import numpy as np
import pandas as pd
import json
from single_model_lib import *

import os
import contextlib
@contextlib.contextmanager
def suppress_print():
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        yield
with suppress_print():
    print('THIS WILL NOT BE PRINTED, USE THIS FOR SETTING PRINT STATEMENTS INVISIBLE')

config_file_path = './configs/config.json'
with open(config_file_path) as f:
    config = json.load(f)
    
dv_names = config['dv_names']
all_optimal_values = {}
base_dir = './scalars/'
for dv_name in dv_names:    
    with open(base_dir + dv_name + '_data_config/optimal_xs.pkl', 'rb') as f:
        x = pickle.load(f)
        all_optimal_values[dv_name] = x
       
        
def optimal_value_sorter(all_dv_optimals, priority_order = 'all_cause_mortality > cancer > stroke > cardio_vascular_disease > diabetes > depression'):
    
    print('\n\n\n\n\nNOW FINDING OPTIMAL COMBINATION\n\n\n')
    global config
    
    allowed_drop = config['allowed_drop'] # How much are the dvs allowed to dip
    
    best_optimal_plan = config['best_optimal_plan']
    print(best_optimal_plan)
    
    pred_func_name = config['pred_func_name']
    priority_order = priority_order.split(' > ')
    
    optimal_combination_final = {'age': all_dv_optimals['age'], 'gender': all_dv_optimals['gender']}
#     optimal_combination_dv_name = {}
    optimal_combination_dv_name = {k:v for k,v in best_optimal_plan.items()}
    
    
    for dv_name in priority_order:
        for iv_name in all_dv_optimals[dv_name]:
            
            if 'smoking' in iv_name:
                continue
                print('HERE')
                print(all_dv_optimals[dv_name][iv_name])
                if (all_dv_optimals[dv_name][iv_name] - 1.0734455)**2 <= 0.0001:
                    print('HERE')
                    all_dv_optimals[dv_name][iv_name] = 1
            if iv_name not in optimal_combination_final:
                optimal_combination_final[iv_name] = all_dv_optimals[dv_name][iv_name]
                optimal_combination_dv_name[iv_name] = dv_name
            elif priority_order.index(dv_name) == priority_order.index(optimal_combination_dv_name[iv_name]) + 1:
                high_priority_dv_name = optimal_combination_dv_name[iv_name]
                original_best_combo = {k:v for k,v in optimal_combination_final.items()}
                
                new_contender = {k:v for k,v in optimal_combination_final.items()}
                new_contender[iv_name] = all_dv_optimals[dv_name][iv_name]
                
                with suppress_print():
                    original_score = get_predictions_from_user_dict(original_best_combo)
                    new_score = get_predictions_from_user_dict(new_contender)
                    
                drop = original_score[pred_func_name[high_priority_dv_name]] - new_score[pred_func_name[high_priority_dv_name]]
                rise = new_score[pred_func_name[dv_name]] - original_score[pred_func_name[dv_name]]
                
                drop = drop/original_score[pred_func_name[high_priority_dv_name]]
                rise = rise/ original_score[pred_func_name[dv_name]]
                
                if high_priority_dv_name == 'all_cause_mortality':
                    drop *= 10
                
                if drop <= allowed_drop[high_priority_dv_name] and rise > drop:
                    optimal_combination_final[iv_name] = all_dv_optimals[dv_name][iv_name]
    
    return optimal_combination_final


def get_optimal_value_combination(age, gender):
    global all_optimal_values, config
    
    priority_order = config['priority_order']
    all_dv_optimals = {}
    for dv_name in all_optimal_values:
        all_dv_optimals[dv_name] = {}
        for iv_name in all_optimal_values[dv_name]:
            if gender not in all_optimal_values[dv_name][iv_name]:
                all_dv_optimals[dv_name][iv_name] = float(all_optimal_values[dv_name][iv_name][0])
            else:
                all_dv_optimals[dv_name][iv_name] = float(all_optimal_values[dv_name][iv_name][gender])
    all_dv_optimals['age'] = age
    all_dv_optimals['gender'] = gender
    return optimal_value_sorter(all_dv_optimals, priority_order)


def get_age_from_ordered_acm_values(gender = 1, current_age = 40, acm_values = [1], common_variables_in_order = None):
    
    # Making sure there are no absurb RR values:
#     print(acm_values)
    for i in range(len(acm_values)):
        temp = min(3, max(0.1, acm_values[i]))
        if temp != acm_values[i]:
            print(f'\n\nACM/RR value found at {i} - ({acm_values[i]}) outside of 0.1-3; looks suspicious, ignoring effect, please investigate.\n')
            acm_values[i] = 1
    if type(common_variables_in_order) == type(None):
        return acm_to_lifespan(gender, int(current_age), acm_values[0]), None
        
    df = {}
    for var_name in ['age','gender','next_variable','predicted_acm', 'predicted_age']:
        df[var_name] = []

    df['age'].append(current_age)
    original_age = int(current_age)
    df['gender'].append(gender)
    df['next_variable'].append('None')
    df['predicted_acm'].append(1)
    current_age = acm_to_lifespan(gender, int(current_age), 1)
    without_intervention_lifespan = float(current_age)
    df['predicted_age'].append(current_age)

    for i in range(len(acm_values)):
        acm = acm_values[i]
        df['age'].append(original_age + current_age - without_intervention_lifespan)
        
        temp = acm_to_lifespan(gender, int(current_age), acm)
        standard = acm_to_lifespan(gender, int(current_age), 1)
        current_age += temp-standard
        
        df['gender'].append(gender)
        df['next_variable'].append(common_variables_in_order[i])
        df['predicted_acm'].append(acm)
        df['predicted_age'].append(current_age)
        
        
    return current_age, pd.DataFrame(df)


def get_predictions_from_user_dict(user_data, lit_review_config_file = None, target_name = None, model_family = None):
    global config
#     print('\n\n\n\nDATA FROM VIEW:\n\n\n', user_data, '\n\n\n')
    
    
    user_var_preds = {}    
    dv_names = config['dv_names']
    l_or_rr = config['l_or_rr']

    for dv_name in dv_names:
        with open('./scalars/all_model_families.json') as f:
            model_family = json.load(f)[dv_name]['model_family']

        ensemble_algo_results = get_return_dictionary(user_data, model_family, dv_name)

        user_var_preds[f'{dv_name}_{l_or_rr[dv_name]}'] = ensemble_algo_results[l_or_rr[dv_name]]

    return user_var_preds


def get_predictions(user_data, model_family, dv_name):
    global config
    analyzer = SingleVariableAnalyzer(f'./scalars/{dv_name}_data_config/{dv_name}_data_config.json', dv_name, model_family)
    analyzer.load_variables()  # Load saved variables
    
    user_var_preds = {}

    user_var_preds['age'] = user_data['age']
    user_var_preds['gender'] = user_data['gender']
    
    # EFFECT OF SMOKING - Defined by a combo of smoking_status (implemented here) and smoking_frequency (added in json file)
    former_smoker_data = config['former_smoker_data']
    if 'smoking_status' in user_data and dv_name in former_smoker_data:
        if user_data['smoking_status'] == 0:
            user_var_preds['smoking'] = 1
        elif user_data['smoking_status'] == 1:
#             print(former_smoker_data)
            user_var_preds['smoking'] = former_smoker_data[dv_name][str(user_data['gender'])]
        else:
            user_data['smoking'] = user_data['smoking_frequency'] 
        
#         del user_data['smoking_frequency']
#         del user_data['smoking_status']
    
    
    user_data = transform_user_data(user_data)
        
        
    variables = config['all_variable_names']

    for var_name in variables:
        if var_name in user_data.keys() and var_name in analyzer.variable_models:
            print(f'\n{var_name} \nrelevant to {dv_name}, is present in user_data.\n')
#             print(user_data[var_name])
            if 1 in analyzer.variable_models[var_name]:
#                 print(analyzer.variable_models)
                inp = {
                    'x_name' : var_name,
                    'model' : analyzer.variable_models[var_name][int(user_data['gender'])],
                    'x_values' : np.array([user_data[var_name]])
                }
            else:                
                inp = {
                    'x_name' : var_name,
                    'model' : analyzer.variable_models[var_name][0],
                    'x_values' : np.array([user_data[var_name]])
                }
#             print(analyzer.variable_models)
#             print(inp["x_values"], inp["x_name"], inp["model"])
    
            preds = analyzer.predict_dv_from_var(inp["x_values"], inp["x_name"], inp["model"])
            user_var_preds[var_name] = float(preds)
            print(var_name, preds)
        elif var_name not in user_data.keys() and var_name in analyzer.variable_models:
            print(f'\n{var_name} \nrelevant to {dv_name}, but not present in user_data.\n')
            
        elif var_name in user_data.keys() and var_name not in analyzer.variable_models:
            print(f'\n{var_name} \nnot relevant to {dv_name}\n')
    return user_var_preds


def get_age_from_user_values(user_var_preds):
    global config
    variables = config['all_variable_names']
    
    flip_order = False # Order of input variables is in inc order of their pred acm by default

    user_logs = {}
    common_variables_in_order = []
    acm_values = []
    for var_name in variables:
        if var_name in user_var_preds:
            common_variables_in_order.append(var_name)
            acm_value = user_var_preds[var_name]
#             if acm_value >1.5 or acm_value < 0.75: # For controlling extreme values
#                 acm_value = 0.5 + acm_value/2.0
            acm_values.append(acm_value)

    acm_values = np.asarray(acm_values)
    common_variables_in_order = np.asarray(common_variables_in_order)
    order = np.argsort(acm_values)
    if flip_order:
        order = order[::-1]
    common_variables_in_order = common_variables_in_order[order]

    acm_values = acm_values[order]
    
    acm_values = 1 + (acm_values - 1)
    final_age, df = get_age_from_ordered_acm_values(gender = user_var_preds['gender'], current_age = user_var_preds['age'], acm_values = acm_values, common_variables_in_order = common_variables_in_order)
        
    return final_age, df
    
    
def get_return_dictionary(user_data, model_family, dv_name):
    
    
    
#     with suppress_print():
    user_var_preds = get_predictions(user_data, model_family, dv_name)
    print(f'dv: {dv_name}, \n\n\nuser_data:\n{user_data}\nuser_var_preds:\n{user_var_preds}\n')
    final_age, df = get_age_from_user_values(user_var_preds)
    no_intervention_age, _ = get_age_from_ordered_acm_values(gender = user_var_preds['gender'], current_age = user_var_preds['age'])
    diff = 1
    if (final_age - no_intervention_age)**2 < 0.01:
        pass
#         print('\n\nNo effect of interventions')
    elif final_age < no_intervention_age:
#         print('\n\nInterventions led to decrease in predicted lifespan')
        diff = 1
    else:
#         print('\n\nInterventions led to increase in predicted lifespan ')
        diff = -1

    
    print('\n\nNOW we are finding an approx rr which is close to the ensemble predicted rr through lifespan method\n\n')
    diff *= 0.01
    all_rrs = np.arange(1,1 + diff*101, diff)
    min_diff = 100
    min_rr = 1
    min_rr_age = no_intervention_age
    for rr in all_rrs:
        with suppress_print():
            rr_age, _ = get_age_from_ordered_acm_values(gender = user_var_preds['gender'], current_age = user_var_preds['age'], acm_values = [rr])
        if (rr_age - final_age)**2 <= min_diff:
            min_diff = (rr_age - final_age)**2
            min_rr = rr
            min_rr_age = rr_age

#     print(f'\n\nFor Risk ratio of {min_rr}, we get a predicted age of {min_rr_age} which is close to final predicted age (with interventions) i.e. {final_age}; compared to no intervention predicted age {round(no_intervention_age, 4)}')

    
    # For adding effect of calorie_restriction:
    if dv_name == 'all_cause_mortality':
        if 'calorie_restriction' in user_var_preds and float(user_var_preds['calorie_restriction']) == 1.0:
            if user_var_preds['age'] <= 40:
                final_age += 5
            elif user_var_preds['age'] <= 50:
                final_age += 4
            elif user_var_preds['age'] <= 60:
                final_age += 3
            elif user_var_preds['age'] <= 70:
                final_age += 2
            else:
                final_age += 1
        
        return {'predicted_lifespan': final_age, 'df' : df}
    return {'predicted_rr' : min_rr, 'df' : df}


def transform_user_data(user_data):
    global config
#     progress_weight_values = [i['weight_progress'] for i in user_data]
    if type(user_data) == list:
        user_data = user_data[0]
        
        
    all_combos = list(user_data.items())
    for k,v in all_combos:
        if type(user_data[k]) == type(None):
            del user_data[k]

    gender_str_to_dict = config['gender_str_to_dict']
    
    if str(user_data['gender']).lower() in ['male', 'female']:
        user_data['gender'] = gender_str_to_dict[user_data['gender']]

    if 'sleep_quality_score_lifestyle' in user_data:
        sleep_quality = user_data['sleep_quality_score_lifestyle']

        if sleep_quality in range(0,8):
            user_data['sleep_quality'] = 2
        elif sleep_quality in range(8,15):
            user_data['sleep_quality'] = 1
        else: # 15-21
            user_data['sleep_quality'] = 0
        
#     print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nUSER_DATA before stress:\n\n\n\n\n\n', user_data)
    if 'stress_level_score' in user_data:
        stress_level = user_data['stress_level_score']
        if stress_level in range(0,14):
            user_data['stress_quality'] = 2
        elif stress_level in range(14,28):
            user_data['stress_quality'] = 1
        else: #28-40
            user_data['stress_quality'] = 0
#     print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nUSER_DATA after stress:\n\n\n\n\n\n', user_data)
        
#     weight, height = user_data['weight'], user_data['height']
    
#     if weight == None:
#         for i in (progress_weight_values + [60]):
#             if i != None:
#                 weight = i
#                 break
#     if height == None:
#         height = 172
#     bmi = weight/((height*0.01)**2)
#     user_data['BMI'] = bmi
    
    return user_data


