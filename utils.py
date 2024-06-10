import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
from math import sqrt
import yaml
from tqdm import tqdm
import importlib
import random
import numpy as np
import pandas as pd
from models.MatrixFactorization import MatrixFactorization
from dataset.dataset import RatingsDataset
from metrics import rmse, mse, precision, recall, f1score, ndcg

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

def load_model(model_type, num_users, num_itens):
    model_file = os.path.join('models', f'{model_type}.py')

    if not os.path.isfile(model_file):
        raise ImportError(f'Arquivo do modelo "{model_type}" não encontrado em "models.')
    
    module = importlib.import_module(f'models.{model_type}')
    model_class = getattr(module, model_type)
    return model_class(num_users, num_itens)

def load_stats_and_data(base_name, train_data, test_data):
    stats_file = os.path.join('config', 'stats.yaml')
    with open(stats_file, 'r') as file:
        stats = yaml.safe_load(file)

    #Verificar se a base de dados ta no stats
    if base_name in stats:
        num_users = stats[base_name]['Usuarios']
        num_itens = stats[base_name]['Itens']
    else:
        raise ValueError(f'Estatísticas não encontradas para a base de dados {base_name}')

    train_dataset = RatingsDataset(train_data)
    test_dataset = RatingsDataset(test_data)

    return num_users, num_itens, train_dataset, test_dataset

def create_dataloaders(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def train_model(model, train_loader, lr, epochs, weight_decay, device):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()

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
            results[metric] = mse.mse(actuals, predictions)
        if metric == 'precision':
            results[metric] = precision.precision(actuals, predictions, k=10)
        if metric == 'recall':
            results[metric] = recall.recall(actuals, predictions, k=10)
        if metric == 'f1score':
            results[metric] = f1score.f1_score(actuals, predictions, k=10)
        if metric == 'ndcg':
            results[metric] = ndcg.ndcg(actuals, predictions, k=10)

    return results

def save_results_csv(results, model_type, base_name):
    if not os.path.exists('results'):
        os.makedirs('results')

    file_name = f'results/{base_name}_{model_type}_results.csv'

    df = pd.DataFrame(results.items(), columns=['Metric', 'Value'])
    df.to_csv(file_name, index=False)

    print(f'Resultados salvos em {file_name}')

def grid_search(model, model_type, train_dataset, test_dataset, base_name, device):
    best_rmse = float('inf')
    best_params = None

    hyperparams = load_hyperparameters(model_type)

    for params in tqdm(ParameterGrid(hyperparams), desc='Grid Search Progress'):
        #print(f'Testando parâmetros: {params}')

        train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, params['batch_size'])
        
        model = train_model(model, train_loader, lr=params['learning_rate'], epochs=params['epochs'], weight_decay=params['weight_decay'],device=device)
        results = evaluate_model(model, test_loader, device)

        if results['rmse'] < best_rmse:
            best_rmse = results['rmse']
            best_params = params

    save_best_parameters(base_name, model_type, best_params, best_rmse)
    save_results_csv(results, model_type, base_name)

    print(f'Melhores parametros encontrados: {best_params}')
    print(f'RMSE: {best_rmse}')