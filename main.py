import os
import torch
import argparse
import yaml
import pandas as pd
from dataset.dataset import RatingsDataset
from utils import train_model, evaluate_model, grid_search, load_model, load_stats, create_dataloaders, set_seed, save_best_results_csv

def main(base_name, model_type, grid_search_flag, use_gpu):
    set_seed(seed=11, use_gpu=use_gpu)

    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    print(f'Dispositivo: {device}')

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', '.data', f'{base_name}')
    train_data_patch = os.path.join(data_dir, 'train_data.csv')
    val_data_patch = os.path.join(data_dir, 'val_data.csv')
    test_data_patch = os.path.join(data_dir, 'test_data.csv')

    if not os.path.exists(train_data_patch) or not os.path.exists(val_data_patch) or not os.path.exists(test_data_patch):
        raise FileNotFoundError('Dados preprocessados não encontrados. Execute o script preprocess.py primeiro')
    
    train_data = pd.read_csv(train_data_patch)
    val_data = pd.read_csv(val_data_patch)
    test_data = pd.read_csv(test_data_patch)

    num_users, num_itens = load_stats(base_name)
    train_dataset = RatingsDataset(train_data)
    val_dataset = RatingsDataset(val_data)
    test_dataset = RatingsDataset(test_data)

    if grid_search_flag:
        params = {}
        model = load_model(model_type, num_users, num_itens, params).to(device)
        grid_search(model, model_type, train_dataset, val_dataset, test_dataset, base_name, device=device)

    else:
        best_params_path = os.path.join('config', 'bestparams.yaml')
        
        with open(best_params_path, 'r') as file:
            best_params = yaml.safe_load(file)
        
        if base_name not in best_params or model_type not in best_params[base_name]:
            raise ValueError(f'Parametros para {model_type} na base {base_name} não encontrados')
        
        model_params = best_params[base_name][model_type]['BestParameters']
        model = load_model(model_type, num_users, num_itens, model_params).to(device)

        train_loader, val_loader ,test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset, model_params['batch_size'])

        model = train_model(model, train_loader, val_loader,lr=model_params['learning_rate'], epochs=model_params['epochs'], weight_decay=model_params['weight_decay'], device=device)
        results = evaluate_model(model, test_loader, device)

        save_best_results_csv(results, model_type, base_name)


        print(f'Modelo treinado com melhores parametros.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recommender System training and evaluation')
    parser.add_argument('--base', type=str,required=True ,help='Nome da base de dados')
    parser.add_argument('--model', type=str,required=True ,help='Tipo do modelo')
    parser.add_argument('--gs', type=int, default=1, help='Executar GridSearch: 1 para sim e 0 para não')
    parser.add_argument('--gpu', type=int, default=1, help='Usar GPU se disponível (1 para sim e 0 para não)')
    args = parser.parse_args()

    main(args.base, args.model, args.gs, args.gpu)