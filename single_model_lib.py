import numpy as np
import json
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import pickle
import os

with open('./configs/config.json') as f:
    var_config = json.load(f)
    

npy_file_location = var_config['npy_file_location']

x = np.load(npy_file_location, allow_pickle=True)


def acm_to_lifespan(gender,age,all_cause_mortality):
    # Modification of Hrithik's code for usage as a function to get lifespan from all cause mortality ratio, gender and age
    
    # letbcaag-ss.npy : life expectancy table values by current age and gender- social security.xlsx
    # contains expected lifespan (without extension based on all cause mortality ratio)
    
    # Saving and using a numpy array for faster loading
    global x
    
    
    # Right now, x is an array of shape (120,2)
    # 120 years from 0 - 119 and 2 values for genders (Male 0; Female 1)
    # Accesing it is similar to original code
    # Original: x['Male'][50] ; New: x[50,0]
        
    number_of_lives = 1000000
    total = 0
    
    for i in range(age,120):
        probability = x[i][int(gender)]*(all_cause_mortality)
        number_of_lives *= (1-probability)
        total += number_of_lives
    return age+(total/1000000)+0.5

class CategoricalModel:
    def __init__(self, variable_data):
        self.intervention_values = variable_data['intervention_values']
        self.prediction_values = variable_data['prediction_values']
        self.value_to_dv = {}

        for i in range(len(self.intervention_values)):
            key = self.intervention_values[i]
            if type(key) == list:
                if key[1] == 'inf':
                    key[1] = key[0] * 2
                    
                key = np.mean(key)
                
            self.value_to_dv[key] = self.prediction_values[i]
            
    def predict(self, x_value):
        if x_value in self.intervention_values:
            if type(x_value) == list:
                x_value = np.mean(x_value)
            return self.value_to_dv[x_value]
        else:
            print(x_value, ' not found in intervention values, invalid class')
            return 1.0

class SingleVariableAnalyzer:
    def __init__(self, config_file_path, target_name, model_family):
        global var_config
        self.config_file_path = config_file_path
        with open(self.config_file_path) as f:
            self.config = json.load(f)
            self.lit_review_data = self.config[target_name]['interventions']
        self.all_input_variables = list(self.lit_review_data.keys())
        self.target_name = target_name
        self.model_family = model_family
        
        self.vars_starting_1 = var_config['vars_starting_1']
        
        self.vars_at_most_1 = var_config['vars_at_most_1'] 
        self.vars_at_least_1 = var_config['vars_at_least_1']
        
        self.fix_at_1_till = var_config['vars_fix_at_1_till']
        
    def modify_y_pred(self, y_pred, x_values, x_name):
        if x_name in self.vars_at_most_1:
            for i in range(len(y_pred)):
                if y_pred[i] > 1:
                    y_pred[i] = 1.0            
                
        if x_name in self.vars_at_least_1:
            for i in range(len(y_pred)):
                if y_pred[i] < 1:
                    y_pred[i] = 1.0

        if x_name in self.vars_starting_1:
            if len(y_pred) != 1:
                y_pred[0] = 1.0
            else:
                if x_values[0] == 0:
                    y_pred[0] = 1.0
            
        if x_name in self.fix_at_1_till:
            for i in range(len(y_pred)):
                if x_values[i] < self.fix_at_1_till[x_name]:
                    y_pred[i] = 1.0

        if x_name == 'sleep_duration':
            within_7_9 = (np.asarray(x_values) >= 7) * (np.asarray(x_values) <= 9)
            for i in range(y_pred.shape[0]):
                if within_7_9[i]:
                    y_pred[i] = 1.0
                    
        return y_pred

        
    def get_datasets(self, input_variable_name, variable_data=None):
        if variable_data is None:
            variable_data = self.lit_review_data[input_variable_name]['distribution']
        values_inp = variable_data['intervention_values']
        values_rr = variable_data['prediction_values']

        print(f'Input Variable Range   \t|\t{self.target_name} (RR) Range')
        for i in range(len(values_inp)):
            print(f'{values_inp[i]}\t\t|\t{values_rr[i]}')

        if values_inp[-1][1] == 'inf':
            values_inp[-1][1] = values_inp[-1][0] * 2

        if np.mean(values_rr[-1]) > np.mean(values_rr[0]):
            if min(values_inp[-1]) > min(values_inp[0]):
                corr = 'positive'
            else:
                corr = 'negative'
                print('check data')
        else:
            if min(values_inp[-1]) > min(values_inp[0]):
                corr = 'negative'
            else:
                corr = 'positive'
                print('check data')

        dataset_method_2 = []  # [start, 25th %tile, 50th %tile (mid), 75th %tile, end]

        min_point_ind = np.argmin(np.asarray([np.mean(i) for i in values_rr]))
#         print(values_inp[min_point_ind//2], values_inp[min_point_ind//2])

        var_model_family = self.model_family[input_variable_name]
        
        for i in range(len(values_inp)):

            input_values = values_inp[i]
            risk_ratios = values_rr[i]

            if var_model_family == 'linear':
                if corr == 'positive':
                    range_start_inp, range_end_inp = min(input_values), max(input_values)
                    range_start_rr, range_end_rr = min(risk_ratios), max(risk_ratios)
#                     print('linear, positive, ', (range_end_rr-range_start_rr)/(range_end_inp-range_start_inp))
                else:
                    range_start_inp, range_end_inp = max(input_values), min(input_values)
                    range_start_rr, range_end_rr = min(risk_ratios), max(risk_ratios)
#                     print('linear, negative, ', (range_end_rr-range_start_rr)/(range_end_inp-range_start_inp))
            
            elif var_model_family == 'exponential':
                if corr == 'positive':
                    range_start_inp, range_end_inp = min(input_values), max(input_values)
                    range_start_rr, range_end_rr = min(risk_ratios), max(risk_ratios)
#                     print('exponential, positive, ', (range_end_rr-range_start_rr)/(range_end_inp-range_start_inp))
                else:
                    range_start_inp, range_end_inp = max(input_values), min(input_values)
                    range_start_rr, range_end_rr = min(risk_ratios), max(risk_ratios)
#                     print('exponential, negative, ', (range_end_rr-range_start_rr)/(range_end_inp-range_start_inp))
                    
            elif var_model_family == 'quadratic':
#                 print('\n\n', values_inp[i], values_inp[i])
                if i > min_point_ind:
                    range_start_inp, range_end_inp = min(input_values), max(input_values)
                    range_start_rr, range_end_rr = min(risk_ratios), max(risk_ratios[0], risk_ratios[1])
#                     print('quadratic, positive, ', (range_end_rr-range_start_rr)/(range_end_inp-range_start_inp))

                else:
#                     print(f'\n\n\n\n\n{input_values}\n\n{risk_ratios}\n\n')
                    range_start_inp, range_end_inp = max(input_values), min(input_values)
                    range_start_rr, range_end_rr = min(risk_ratios), max(risk_ratios)
#                     print('quadratic, negative, ', (range_end_rr-range_start_rr)/(range_end_inp-range_start_inp))

            mid_point_inp = (range_start_inp + range_end_inp) / 2
            mid_point_rr = (range_start_rr + range_end_rr) / 2

            # Method 2

            first_quart_inp = (range_start_inp + mid_point_inp) / 2
            first_quart_rr = (range_start_rr + mid_point_rr) / 2

            second_quart_inp = (mid_point_inp + range_end_inp) / 2
            second_quart_rr = (mid_point_rr + range_end_rr) / 2
            
            inc_i = np.abs(values_inp[i][0] - values_inp[i][1])

            if i >= len(values_inp)-2:
                inc_i_p_1 = inc_i
            else:
                inc_i_p_1 = np.abs(values_inp[i+1][0] - values_inp[i+1][1])

            if inc_i == 0:
                dataset_method_2.append((range_start_inp, range_start_rr))
                
            elif inc_i in [0.1,0.5] and inc_i <= inc_i_p_1/5: # for alcohol graph, change this to inc_i <=1 and run
                dataset_method_2.append((range_start_inp, range_start_rr))
#                 dataset_method_2.append((mid_point_inp, mid_point_rr))
                dataset_method_2.append((range_end_inp, range_end_rr))

            else:
                dataset_method_2.append((range_start_inp, range_start_rr))
                dataset_method_2.append((first_quart_inp, first_quart_rr))
                dataset_method_2.append((mid_point_inp, mid_point_rr))
                dataset_method_2.append((second_quart_inp, second_quart_rr))
                dataset_method_2.append((range_end_inp, range_end_rr))

        return dataset_method_2

    def get_polynomial_model(self, dataset_method, degree):
        x = np.asarray([i[0] for i in dataset_method])
        y = np.asarray([i[1] for i in dataset_method])

        x = x.reshape(-1, 1)

        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(x)

        # Fit polynomial regression model
        model = LinearRegression()
        model.fit(X_poly, y)

        return model, poly, x, y

    def predict_polynomial(self, x_values, x_name, model):
#         print(model)
        model, poly = model['model'], model['poly']
        # Generate polynomial features for x values
        x_values_poly = poly.transform(x_values.reshape(-1, 1))
#         print('Polynomial, model, ', model, '\nx_values:\n\n',x_values,'\n\n\n')
        y_pred = np.asarray(model.predict(x_values_poly))
        
        y_pred = self.modify_y_pred(y_pred, x_values, x_name)

        return y_pred

    def predict_and_plot_polynomial(self, x, y, model_instance, poly_instance, x_name, degree, plot_flag=False):
        # Generate x values for plotting the polynomial function
        x_values = np.linspace(min(x), max(x), 1000)

        
        
        
        # THIS IS NOT GETTING TRANSMITTED
        model_dict = {'model': model_instance, 'poly': poly_instance}
#         print('\n\n\n\n\n\n\n\n\n\n\n\n\npredict_and_plot_polynomial\n', model_dict)
        y_pred = self.predict_dv_from_var(x_values, x_name, model_dict)
        min_y_ind = np.argmin(y_pred)

        if plot_flag:
            print('variable found: ', x_name)
            if degree == 1:
                label = 'Linear Model'
            else:
                label = 'Quadratic Model'

            plt.scatter(x, y, label='Data Points')
            plt.plot(x_values, y_pred, color='red', label=label)
            plt.xlabel(x_name)
            plt.ylabel(self.target_name)
            plt.title(label)
            plt.legend()
            plt.grid(True)
            plt.show()

        return np.round(x_values[min_y_ind],4), np.round(y_pred[min_y_ind],4)
    
    def predict_categorical(self, x_values, x_name, model):
#         for i in model:
#             print(i,model[i])
        y_pred = model['model'].predict(float(x_values))
        return y_pred

    def predict_dv_from_var(self, x_values, x_name, model_dict):
        if self.model_family[x_name] == 'exponential':
            return self.predict_exponential(x_values, x_name, model_dict)
        elif self.model_family[x_name] == 'categorical':
            return self.predict_categorical(x_values, x_name, model_dict)
        else:
#             print('predict_dv_from_var: ', model_dict)
            return self.predict_polynomial(x_values, x_name, model_dict)

    def exponential_decay_function(self, x, a, b, c):
        return a * np.exp(-b * x) + c

    def predict_exponential(self, x_values, x_name, model):
        model = model['model']
        y_pred = self.exponential_decay_function(x_values, *model[0])
        y_pred = self.modify_y_pred(y_pred, x_values, x_name)
            
        return y_pred
    

    def predict_and_plot_exponential(self, dataset_method, model, x_name, plot_flag=False):
        

        x = np.asarray([i[0] for i in dataset_method])
        y = np.asarray([i[1] for i in dataset_method])

        
        # Generate x values for plotting the exponential function
        x_values = np.linspace(min(x), max(x), 1000)

        # Calculate predicted y values using the exponential function
        y_pred = self.predict_dv_from_var(x_values, x_name, model)

        if plot_flag:
            plt.scatter(x, y, label='Data Points')
            # Plot the exponential function
            plt.plot(x_values, y_pred, color='red', label='Exponential Regression')

            plt.xlabel(x_name)
            plt.ylabel(self.target_name)
            plt.title('Exponential Regression')
            plt.legend()
            plt.grid(True)
            plt.show()

        min_y_ind = np.argmin(y_pred)
        
        min_y = y_pred[0]
        min_x = x_values[0]
        
        threshold = (np.max(y_pred) - np.min(y_pred))/50 # benefit of increasing variable should be atleast threshold
#         threshold = max(0.01, (np.max(y_pred) - np.min(y_pred))/1000.0) # benefit of increasing variable should be atleast threshold
#         print(x_name, ' threshold ', threshold, '\ny_pred\n', y_pred)
        for i in range(len(y_pred)):
            if y_pred[i] < min_y and (np.abs(min_y-y_pred[i]) >= threshold):
                min_y = y_pred[i]
                min_x = x_values[i]
#                 print('min_y updated to: ', min_y)

        return np.round(min_x,4), np.round(min_y, 4)

    def fit_all_models(self):
        variable_models = {}
        optimal_xs = {}
        optimal_ys = {}
        lit_review_data_copy = {k: v for k, v in self.lit_review_data.items()}

        for variable_ind in range(len(self.all_input_variables)):
            variable = self.all_input_variables[variable_ind]

            print('\n', variable, '\n')

            if variable not in self.model_family:
                print(f'Variable not found in {self.config_file_path}\n\n')
                continue

            variable_data_s = lit_review_data_copy[variable]['distribution']

            print(f'\n\n\n\nVariable Name: {variable}\n')

            if 'male' not in variable_data_s:
                variable_data_s = {'gender_unbiased': variable_data_s}
            else:
                print("This variable's distribution is split on gender\n\n")

                
            variable_models[variable], optimal_xs[variable], optimal_ys[variable] = {}, {}, {}
            for gender in variable_data_s:
                if gender in ['male', 'gender_unbiased']:
                    gender_ind = 0
                elif gender =='female':
                    print('\n\n' + '-' * 115 + '\n* Now for next gender *\n' + '-' * 115)
                    gender_ind = 1
                    
                print('\ngender: ', gender)
                variable_data = variable_data_s[gender]
                
                    
                if self.model_family[variable] == 'categorical':
                    print(f'\n{variable} type = categorical\n')
                    print(f"input values:\n{variable_data['intervention_values']}\n{self.target_name} (RR) range:\n{variable_data['prediction_values']}\n\n")
                    intervention_values, prediction_ranges = variable_data['intervention_values'], variable_data['prediction_values']

                    if type(prediction_ranges[0]) == list:
                        prediction_ranges = [np.mean(i) for i in prediction_ranges]

                    best_class_ind = np.argmin(prediction_ranges)
                    best_class_option = intervention_values[best_class_ind]
                    best_class_range = prediction_ranges[best_class_ind]
#                     print(prediction_ranges)
#                     print(np.min(prediction_ranges))
#                     print(best_class_ind)
#                     print(best_class_range)
                    print(f'From the distribution, class {best_class_option} has the best {self.target_name} rating and thus optimal')

                    cat_var_inf = {'intervention_values' : intervention_values, 'prediction_values' : prediction_ranges}
                    model = CategoricalModel(cat_var_inf)

                    variable_models[variable][gender_ind] = {'model' :model}
                    least_y_ind = np.argmin(prediction_ranges)
                    optimal_x = intervention_values[least_y_ind]
                    least_y = prediction_ranges[least_y_ind]
                    optimal_xs[variable][gender_ind] =  optimal_x
                    optimal_ys[variable][gender_ind] = least_y
                    


                elif self.model_family[variable] == 'linear':
                    degree = 1
                    data = self.get_datasets(variable_data=variable_data, input_variable_name=variable)
                    model, poly, x, y = self.get_polynomial_model(data, degree)
                    optimal_x, least_y = self.predict_and_plot_polynomial(x, y, model, poly, x_name=variable, degree=degree)
                    print(f'According to the graph, {optimal_x} gives us least polynomial model predicted {self.target_name}(RR) value {least_y}')

                    variable_models[variable][gender_ind] = {'model': model, 'poly': poly}
                    optimal_xs[variable][gender_ind] = optimal_x
                    optimal_ys[variable][gender_ind] = least_y

                elif self.model_family[variable] == 'quadratic':
                    degree = 2
                    data = self.get_datasets(variable_data=variable_data, input_variable_name=variable)
                    model, poly, x, y = self.get_polynomial_model(data, degree)
                    optimal_x, least_y = self.predict_and_plot_polynomial(x, y, model, poly, x_name=variable, degree=degree)
                    print(f'According to the graph, {optimal_x} gives us least polynomial model predicted {self.target_name}(RR) value {least_y}')

                    variable_models[variable][gender_ind] = {'model': model, 'poly': poly}
                    optimal_xs[variable][gender_ind] = optimal_x
                    optimal_ys[variable][gender_ind] = least_y

                elif self.model_family[variable] == 'exponential':
                    data = self.get_datasets(variable_data=variable_data, input_variable_name=variable)
#                     print("variable_data", variable, data)
                    model = curve_fit(self.exponential_decay_function, [i[0] for i in data], [i[1] for i in data], maxfev = 10000)
                    optimal_x, least_y = self.predict_and_plot_exponential(data, {'model':model}, x_name=variable)
                    print(f'According to the graph, {optimal_x} gives us least exponential model predicted {self.target_name}(RR) value {least_y}')

                    variable_models[variable][gender_ind] = {'model' :model}
                    optimal_xs[variable][gender_ind] = optimal_x
                    optimal_ys[variable][gender_ind] = least_y
                    
            print('\n\n' + '-' * 230 + '\n* END OF VARIABLE *\n' + '-' * 230)
            
        self.variable_models, self.optimal_xs, self.optimal_ys = variable_models, optimal_xs, optimal_ys
        return variable_models, optimal_xs, optimal_ys
    
    def save_variables(self):
        global var_config
        directory_structure = var_config['directory_structure']
        for dir_name in directory_structure:
            os.makedirs(dir_name, exist_ok=True)
            
        print('\n\n\n\n\n\n\nSAVING MODELS\n')
        with open(os.path.join('/'.join(self.config_file_path.split('/')[:-1]) , 'variable_models.pkl'), 'wb') as f:
            pickle.dump(self.variable_models, f)
        with open(os.path.join('/'.join(self.config_file_path.split('/')[:-1]) , 'optimal_xs.pkl'), 'wb') as f:
            pickle.dump(self.optimal_xs, f)
        with open(os.path.join('/'.join(self.config_file_path.split('/')[:-1]) , 'optimal_ys.pkl'), 'wb') as f:
            pickle.dump(self.optimal_ys, f)

    def load_variables(self):
        print('\n\n\n\n\n\n\nLOADING MODELS\n',os.path.join('/'.join(self.config_file_path.split('/')[:-1]) , 'variable_models.pkl'))
        with open(os.path.join('/'.join(self.config_file_path.split('/')[:-1]) , 'variable_models.pkl'), 'rb') as f:
            self.variable_models = pickle.load(f)
        with open(os.path.join('/'.join(self.config_file_path.split('/')[:-1]) , 'optimal_xs.pkl'), 'rb') as f:
            self.optimal_xs = pickle.load(f)
        with open(os.path.join('/'.join(self.config_file_path.split('/')[:-1]) , 'optimal_ys.pkl'), 'rb') as f:
            self.optimal_ys = pickle.load(f)


def main():
    with open('./scalars/all_model_families.json') as f:
        all_model_families = json.load(f)
        for target_variable in all_model_families:

            print('Starting with dependent variable: ', target_variable, '\n\n\n')
            var_info = all_model_families[target_variable]

            config_file_path = var_info['lit_review_file_path']
            target_name = var_info['target_name']
            model_family = var_info['model_family']

            analyzer = SingleVariableAnalyzer(config_file_path, target_name, model_family)
            variable_models, optimal_xs, optimal_ys = analyzer.fit_all_models()
            analyzer.save_variables()

if __name__ == "__main__":
    main()
