import os
import yaml
import pandas as pd
import csv

def save_best_parameters(base_name, model_type, best_params, best_rmse):
    file_path = os.path.join('config', 'bestparams.yaml')
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            all_params = yaml.safe_load(file)
    else:
        all_params = {}
    
    if not all_params:
        all_params = {}
    
    if base_name not in all_params:
        all_params[base_name] = {}
    
    if model_type not in all_params[base_name]:
        all_params[base_name][model_type] = {}
    
    all_params[base_name][model_type]['BestParameters'] = best_params
    all_params[base_name][model_type]['BestRMSE'] = best_rmse

    with open(file_path, 'w') as file:
        yaml.safe_dump(all_params, file)

def save_best_results_csv(results_dict, model_type, base_name):
    if not os.path.exists('results'):
        os.makedirs('results')

    file_name = f'results/{base_name}_{model_type}_results.csv'

    df = pd.DataFrame(results_dict.items(), columns=['Metric', 'Value'])
    df.to_csv(file_name, index=False)

    print(f'Resultados salvos em {file_name}')


def save_hyperparams_results(results_list, model_type, base_name, hyperparams_config):
    if not os.path.exists('results'):
        os.makedirs('results')

    file_name = f'results/{base_name}_{model_type}_all_results.csv'

    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)

        header = list(hyperparams_config.keys()) + ['rmse']
        writer.writerow(header)

        for entry in results_list:
            params = entry['params']
            results = entry['results']

            row = [params.get(hp,'N/A') for hp in hyperparams_config.keys()] + [results['rmse']]
            writer.writerow(row)