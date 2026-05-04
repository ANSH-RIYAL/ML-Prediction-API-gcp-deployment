import json
import os
import numpy as np

base_dir = './scalars/'
s = []
s_v = []
for file in ['all_cause_mortality_data_config', 'depression_data_config', 'diabetes_data_config', 'stroke_data_config', 'cancer_data_config', 'cardio_vascular_disease_data_config']:
    with open(base_dir + file + '/' + file + '.json') as f:

        print(file, '\n')
        data = json.load(f)
        var_name = str(list(data.keys())[0])
        intervention_names = list(data[var_name]['interventions'].keys())
        d = {i: {'numerical': 'linear', 'categorical': 'categorical'}[data[var_name]['interventions'][i]['type']] for i in intervention_names}
        print(d, '\n\n')
        
        for i in intervention_names:
            if i not in s:
                s_v += [(i,var_name)]
                s += [i]
        


# np.asarray(s_v)[np.argsort([i[0] for i in s_v])]



all_model_families = {}


all_model_families['all_cause_mortality'] = {
    'lit_review_file_path' : base_dir + 'all_cause_mortality_data_config/all_cause_mortality_data_config.json',
    'target_name' : 'all_cause_mortality',
    'model_family':{'alcohol': 'quadratic', # comments from email
                    'grain_unrefined': 'exponential', 
                    'grain_refined': 'linear', 
                    'meat_unprocessed': 'exponential', 
                    'meat_processed': 'exponential', 
                    'fruits_and_veggies': 'exponential', 
                    'water': 'quadratic', 
                    'refined_sugar': 'linear', 
                    'artificial_sweetener': 'linear', 
                    'cardio': 'exponential', 
                    'strength_training': 'quadratic', 
                    'sleep_duration': 'quadratic', 
                    'sleep_quality': 'categorical', 
                    'sauna_frequency': 'quadratic', 
                    'stress_level': 'categorical', 
                    'calcium': 'quadratic', 
                    'fish_oil_omega_3': 'quadratic', 
                    'green_tea': 'quadratic', 
                    'legumes': 'exponential', 
                    'fat_trans': 'exponential',
                    'smoking': 'linear'} 
}



all_model_families['depression'] = {
    'lit_review_file_path' : base_dir + 'depression_data_config/depression_data_config.json',
    'target_name' : 'depression',
    'model_family':{'fish_oil_omega_3': 'exponential'}
}

all_model_families['diabetes'] = {
    'lit_review_file_path' : base_dir + 'diabetes_data_config/diabetes_data_config.json',
    'target_name' : 'diabetes',
    'model_family': {'olive_oil': 'exponential',
                     'meat_unprocessed': 'linear',
                     'meat_processed': 'exponential',
                     'refined_sugar': 'exponential',
                     'artificial_sweetener': 'exponential',
                     'grain_unrefined': 'exponential',
                     'grain_refined': 'linear',
                     'dairy': 'linear',
                     'fruits_and_veggies': 'exponential',
                     'cardio': 'quadratic',
                     'sleep_duration': 'quadratic'}
}

all_model_families['stroke'] = {
    'lit_review_file_path' : base_dir + 'stroke_data_config/stroke_data_config.json',
    'target_name' : 'stroke',
    "model_family": {
        "olive_oil": "linear",
        "dairy": "linear",
        "grain_unrefined": "exponential",
        "cardio": "quadratic",
        "sleep_duration": "quadratic",
        "sauna_frequency": "quadratic",
        "stress_level": "categorical",
        "dietary_fiber": "linear",
        "legumes": "exponential"
    }
}



all_model_families['cancer'] = {
    'lit_review_file_path' : base_dir + 'cancer_data_config/cancer_data_config.json',
    'target_name' : 'cancer',
    "model_family": {"olive_oil": "linear",
                     "alcohol": "exponential",
                     "fruits_and_veggies": "linear",
                     "dietary_fiber": "exponential",
                     "sleep_duration": "quadratic",
                     "cardio": "quadratic",
                     "grain_unrefined": "exponential",
                     "refined_sugar": "exponential",
                     "calcium": "quadratic",
                     "strength_training": "exponential",
                     "meat_unprocessed": "exponential",
                     "meat_processed": "exponential",
                     "smoking": "linear"
    }
}


all_model_families['cardio_vascular_disease'] = {
    'lit_review_file_path' : base_dir + 'cardio_vascular_disease_data_config/cardio_vascular_disease_data_config.json',
    'target_name' : 'cardio_vascular_disease',
    'model_family': {'olive_oil': 'exponential', 
                     'multi_vitamins': 'categorical', 
                     'vitamin_e': 'linear', 
                     'water': 'linear',   
                     'fruits_and_veggies': 'exponential',   
                     'refined_sugar': 'linear',   
                     'calcium': 'quadratic',   
                     'strength_training': 'quadratic',   
                     'sauna_duration': 'linear', 
                     'sauna_frequency': 'exponential',   
                     'sleep_duration': 'quadratic',   
                     'cardio': 'exponential',   
                     'alcohol': 'quadratic', 
                     'fat_trans': 'linear',   
                     'grain_unrefined': 'linear',   
                     'grain_refined': 'linear', 
                     'stress_level': 'categorical',
                     'meat_processed' : 'exponential',
                     'meat_unprocessed' : 'exponential',
                     'meat_poultry' : 'exponential',
                     'smoking' : 'exponential'
                     } 
}


with open ('./scalars/all_model_families.json', 'w') as f:
    json.dump(all_model_families, f)

with open('./scalars/all_model_families.json') as f:
    all_model_families = json.load(f)
    for var in all_model_families:
        print(all_model_families[var])


config_dict = {}

config_dict['all_variable_names'] = ['alcohol',
 'grain_unrefined',
 'grain_refined',
 'meat_unprocessed',
 'meat_processed',
 'meat_poultry',
 'fruits_and_veggies',
 'water',
 'refined_sugar',
 'artificial_sweetener',
 'cardio',
 'strength_training',
 'sleep_duration',
 'sleep_quality',
 'sauna_frequency',
 'stress_quality',
 'calcium',
 'fish_oil_omega_3',
 'green_tea',
 'legumes',
 'fat_trans',
 'smoking',
 'olive_oil',
 'dairy',
 'dietary_fiber',
 'multi_vitamins',
 'vitamin_e',
 'sauna_duration']

# config_dict['sql_query_paid'] = 'SELECT * FROM mmlmodel_updated WHERE mmlmodel_updated.user_id = "{}" ORDER BY submission_date DESC'
config_dict['sql_query_paid'] = 'SELECT * FROM mmlmodel WHERE mmlmodel.user_id = "{}"'

config_dict['sql_query_free'] = 'SELECT * FROM calculatorTable WHERE calculatorTable.user_id = "{}"'


config_dict['vars_starting_1'] = ['alcohol', 'meat_unprocessed', 'meat_processed', 'meat_poultry', 'smoking', 'artificial_sweetener', 'cardio', 'strength_training', '', 'stress_level', 'calcium', '', 'legumes', 'dairy', 'dietary_fiber', '', 'vitamin_e', '', 'olive_oil', 'dairy']


config_dict['vars_at_most_1'] = ['dietary_fiber', 'green_tea', 'olive_oil', 'grain_unrefined', 'sauna_frequency', 'fruits_and_veggies', '', 'sauna_duration', 'sauna_frequency', 'cardio']


config_dict['vars_at_least_1'] = ['grain_refined', 'smoking', 'fat_trans', 'refined_sugar', 'meat_unprocessed', 'meat_unprocessed'] 


config_dict['vars_fix_at_1_till'] = {'water' : 1, 'fish_oil_omega_3' : 0.3}


config_dict['npy_file_location'] = './scalars/letbcaag-ss.npy'

config_dict['dv_names'] = ['all_cause_mortality', 'diabetes', 'stroke', 'cancer', 'cardio_vascular_disease', 'depression']


config_dict['l_or_rr'] = {'all_cause_mortality' : 'predicted_lifespan', 
               'diabetes' : 'predicted_rr', 
               'stroke' : 'predicted_rr', 
               'cancer' : 'predicted_rr', 
               'cardio_vascular_disease' :'predicted_rr',
              'depression': 'predicted_rr'}

config_dict['former_smoker_data'] = {"all_cause_mortality" : {0 : np.mean([1.00,1.11]), 1 : np.mean([.87, 1.07])},
                                  "cancer" : {0 : np.mean([1.16,1.37]), 1 : np.mean([1.16,1.37])}}


config_dict['directory_structure'] = ['./scalars', './scalars/all_cause_mortality_data_config', './scalars/heart_disease_data_config', './scalars/stroke_data_config', './scalars/cancer_data_config', './scalars/diabetes_data_config', './scalars/depression_data_config']


config_dict['gender_str_to_dict'] = {'Male': 0, 'male' : 0, 'Female' : 0, 'female' : 0}

config_dict['priority_order'] = 'all_cause_mortality > cardio_vascular_disease > stroke > cancer > diabetes > depression'

config_dict['allowed_drop'] = {'all_cause_mortality' : -0.2, # we multiple the actual drop by 10 to get this
                   'cancer' : 0.1,
                   'stroke' : 0.1,
                   'cardio_vascular_disease' : 0.1,
                   'diabetes' : 0.1,
                   'depression' : 0.1,
                   }
    
config_dict['pred_func_name'] = {
    'all_cause_mortality' : 'all_cause_mortality_predicted_lifespan',
    'cancer' : 'cancer_predicted_rr',
    'stroke' : 'stroke_predicted_rr',
    'cardio_vascular_disease' : 'cardio_vascular_disease_predicted_rr',
    'diabetes' : 'diabetes_predicted_rr',
    'depression' : 'depression_predicted_rr'
}

config_dict['best_optimal_plan'] = {'alcohol': 0.0, 'artificial_sweetener': 0.0, 'calcium': 0.0, 'calorie_restriction': 1, 'cardio': 35.4955, 'dairy': 0.0, 'dietary_fiber': 60.0, 'fat_trans': 0.0, 'fish_oil_omega_3': 1.8869, 'fruits_and_veggies': 8.4319, 'gender': 0, 'grain_refined': 0.0, 'grain_unrefined': 2.8861, 'green_tea': 1116.753, 'legumes': 6.0511, 'meat_poultry': 0.0, 'meat_processed': 0.0, 'meat_unprocessed': 0.0, 'multi_vitamins': 1.0, 'olive_oil': 50.0, 'refined_sugar': 0.0, 'sauna_duration': 38.2, 'sauna_frequency': 3.034, 'sleep_duration': 7.0, 'strength_training': 96.4597, 'vitamin_e': 1000.0, 'water': 1901.2292}

with open('./configs/config.json', 'w') as f:
    json.dump(config_dict, f)
    

print('\n\n\nJSON FILES CREATED.\n\n\n')