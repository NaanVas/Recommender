import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, precision_score, recall_score
import yaml
from tqdm import tqdm
import importlib
import random
import numpy as np
import pandas as pd
from metrics import rmse, f1score, ndcg
import csv

def set_seed(seed, use_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu:
        torch.cuda.manual_seed_all(seed)

def load_hyperparameters(model_type):
    file_path = os.path.join('config', 'hyperparams.yaml')
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    model_hyperparams = config['GridSearch'].get(model_type)
    if model_hyperparams is None:
        raise ValueError(f'Hyperparameters for {model_type} not found in {file_path}')
    
    return model_hyperparams
    

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

def load_model(model_type, num_users, num_itens, params):
    model_file = os.path.join('models', f'{model_type}.py')

    if not os.path.isfile(model_file):
        raise ImportError(f'Arquivo do modelo "{model_type}" não encontrado em "models.')
    
    module = importlib.import_module(f'models.{model_type}')
    model_class = getattr(module, model_type)

    if model_type == 'MatrixFactorization':
        if params is None or 'embedding_dim' not in params:
            print('Parâmetro "Embedding_dim" necessário para Matrix factorization não encontrado.')
            return model_class(num_users, num_itens, embedding_dim = 20)
    
        return model_class(num_users, num_itens, params['embedding_dim'])
    
    if model_type == 'NMF':
        if params is None or 'latent_factors' not in params:
            print('Parâmetro "latent_factors" necessário para NMF não encontrado.')
            return model_class(num_users, num_itens, latent_factors = 20)

        return model_class(num_users, num_itens, params['latent_factors'])
    
    return 0

def load_stats(base_name):
    stats_file = os.path.join('config', 'stats.yaml')
    with open(stats_file, 'r') as file:
        stats = yaml.safe_load(file)

    #Verificar se a base de dados ta no stats
    if base_name in stats:
        num_users = stats[base_name]['Usuarios']
        num_itens = stats[base_name]['Itens']
    else:
        raise ValueError(f'Estatísticas não encontradas para a base de dados {base_name}')

    return num_users, num_itens

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, lr, epochs, weight_decay, device):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for users, itens, ratings in train_loader:
            users, itens, ratings = users.to(device), itens.to(device), ratings.to(device)
            optimizer.zero_grad()
            predictions = model(users, itens)
            loss = loss_fn(predictions, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for users, itens, ratings in val_loader:
                users,itens,ratings = users.to(device), itens.to(device), ratings.to(device)
                predictions = model(users, itens)
                loss = loss_fn(predictions, ratings)
                val_loss += loss.item()
        
        avg_train_loss = total_loss/len(train_loader)
        avg_val_loss = total_loss/len(val_loader)

        #print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
        
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model

def load_metrics():
    metrics_path = os.path.join('config','metrics.yaml')
    with open(metrics_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['metrics']

def evaluate_model(model, test_loader, device):
    metrics = load_metrics()
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for users, itens, ratings in test_loader:
            users, itens, ratings = users.to(device), itens.to(device), ratings.to(device)
            preds = model(users, itens)
            predictions.extend(preds.cpu().numpy())
            actuals.extend(ratings.cpu().numpy())

    results = {
        'rmse': rmse.rmse(actuals, predictions)
    }

    for metric in metrics:
        if metric == 'mse':
            results[metric] = mean_squared_error(actuals, predictions)
        if metric == 'precision':
            results[metric] = precision_score(np.around(actuals), np.around(predictions), average="macro", zero_division=np.nan)
        if metric == 'recall':
            results[metric] = recall_score(np.around(actuals), np.around(predictions), average="macro", zero_division=np.nan)
        if metric == 'f1score':
            results[metric] = f1score.f1_score(actuals, predictions)
        if metric == 'ndcg':
            results[metric] = ndcg.ndcg(actuals, predictions, k=10)

    return results

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

def grid_search(model, model_type, train_dataset, val_dataset, test_dataset, base_name, device):
    best_rmse = float('inf')
    best_params = None
    best_results = None
    results_list = []

    hyperparams = load_hyperparameters(model_type)
    num_users, num_itens = load_stats(base_name)

    for params in tqdm(ParameterGrid(hyperparams), desc='Grid Search Progress'):
        #Reinicializar modelo para cada conjunto de hiperparametros
        model = load_model(model_type, num_users, num_itens, params).to(device)
        train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset, params['batch_size'])
        
        model = train_model(model, train_loader, val_loader, lr=params['learning_rate'], epochs=params['epochs'], weight_decay=params['weight_decay'],device=device)
        results = evaluate_model(model, test_loader, device)

        results_list.append({'params': params, 'results': results})

        if results['rmse'] < best_rmse:
            best_rmse = results['rmse']
            best_params = params
            best_results = results

    save_best_parameters(base_name, model_type, best_params, best_rmse)
    save_best_results_csv(best_results, model_type, base_name)
    save_hyperparams_results(results_list, model_type, base_name, hyperparams)

    print(f'Melhores parametros encontrados: {best_params}')
    print(f'RMSE: {best_rmse}')