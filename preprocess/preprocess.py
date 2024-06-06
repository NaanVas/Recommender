import sys
import os
import yaml
from utils import preprocess_data, load_data

def main(base_name):
    df = load_data()
    train_data, test_data = preprocess_data(df)

    #Salvar dados preprocessados
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','dataset' ,'.data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    train_data.to_csv(os.path.join(data_dir, 'train_data.csv'), index=False)
    test_data.to_csv(os.path.join(data_dir, 'test_data.csv'), index=False)

    #Salvar estatísticas
    stats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config')
    stats_file = os.path.join(stats_dir, 'stats.yaml')

    num_users = df['user_id'].nunique()
    num_items = df['item_id'].nunique()

    stats = {
        'Usuarios': num_users,
        'Itens': num_items,
        'Rating': len(df) 
    }

    data = {}

    if os.path.exists(stats_file):
        with open(stats_file, 'r') as existing_file:
            data = yaml.safe_load(existing_file) or {}

    data[base_name] = stats

    with open(stats_file, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False)

    print("Dados carregados e preprocessados com sucesso.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Uso: python3 preprocess.py base_name")
        sys.exit(1)
    
    base_name = sys.argv[1]
    main(base_name)