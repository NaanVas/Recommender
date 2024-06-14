import os
import yaml

def load_hyperparameters(model_type):
    file_path = os.path.join('config', 'hyperparams.yaml')
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    model_hyperparams = config['GridSearch'].get(model_type)
    if model_hyperparams is None:
        raise ValueError(f'Hyperparameters for {model_type} not found in {file_path}')
    
    return model_hyperparams

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