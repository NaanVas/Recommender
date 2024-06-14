import os
import yaml
import torch
from sklearn.metrics import mean_squared_error, precision_score, recall_score
import numpy as np
from metrics import rmse, f1score, ndcg

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