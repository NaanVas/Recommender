from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from utils.loads import load_hyperparameters, load_stats
from utils.model import load_model, create_dataloaders, train_model
from utils.metrics import evaluate_model
from utils.save import save_best_parameters, save_best_results_csv, save_hyperparams_results


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
        rmse = results['rmse']
        print(f'Results {rmse}')

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