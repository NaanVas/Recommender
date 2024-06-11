import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'datasets.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_data_dir(base_name):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data', base_name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir


def load_data(base_name):
    config = load_config()
    if base_name not in config['datasets']:
        raise ValueError(f'Configuração para {base_name} não encontrada.')
    
    data_config = config['datasets'][base_name]
    data_dir = get_data_dir(base_name)
    file_path = os.path.join(data_dir, data_config['file_name'])

    if os.path.exists(file_path):
        print(f'Carregando dados de {file_path}')
        if data_config['file_format'] == 'csv':
            delimiter = ',' if data_config['delimiter'] == ',' else '\t'
            df = pd.read_csv(file_path, names=data_config['column_names'], delimiter=delimiter)
        elif data_config['file_format'] == 'json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f'Formato de arquivo não suportado: {data_config["file_format"]}')
        
    else:
        raise FileNotFoundError(f'Arquivo {file_path} não enontrado na pasta .data')
    
    return df

def preprocess_data(df, base_name):
    config = load_config()
    data_config = config['datasets'][base_name]

    df = df.drop('timestamp', axis=1)

    if df[data_config['column_names'][0]].dtype != 'int' or df[data_config['column_names'][1]].dtype != 'int':
        # Remapeia os ids para números de 0 a n-1
        df[data_config['column_names'][0]] = pd.factorize(df[data_config['column_names'][0]])[0]
        df[data_config['column_names'][1]] = pd.factorize(df[data_config['column_names'][1]])[0]
    else:
        df = ordenar_df(df, data_config)

    train_data, temp_data = train_test_split(df, test_size=0.40)
    val_data, test_data = train_test_split(temp_data, test_size=0.25)
    return train_data, val_data, test_data, df

def ordenar_df(df, data_config):

    df.sort_values(by=[data_config['column_names'][0], data_config['column_names'][1]], inplace=True)
    df.reset_index(drop=True, inplace=True)
            
    # Verificar se o user_id e o item_id não começam com 0
    if df[data_config['column_names'][0]].min() != 0 or df[data_config['column_names'][1]].min() != 0:
        df[data_config['column_names'][0]] = pd.factorize(df[data_config['column_names'][0]])[0]
        df[data_config['column_names'][1]] = pd.factorize(df[data_config['column_names'][1]])[0]
    
    return df