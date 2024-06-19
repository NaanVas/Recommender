import os
import torch
from torch.utils.data import DataLoader
import importlib

def load_model(model_type, num_users, num_itens, params):
    model_file = os.path.join('models', f'{model_type}.py')

    if not os.path.isfile(model_file):
        raise ImportError(f'Arquivo do modelo "{model_type}" não encontrado em "models.')
    
    module = importlib.import_module(f'models.{model_type}')
    model_class = getattr(module, model_type)

    if model_type in ['MatrixFactorization', 'SVD']:
        if params is None or 'embedding_dim' not in params:
            #print(f'Parâmetro "Embedding_dim" necessário para {model_type} não encontrado.')
            return model_class(num_users, num_itens, embedding_dim = 20)
    
        return model_class(num_users, num_itens, params['embedding_dim'])
    
    if model_type == 'NMF':
        if params is None or 'latent_factors' not in params:
            #print('Parâmetro "latent_factors" necessário para NMF não encontrado.')
            return model_class(num_users, num_itens, latent_factors = 20)

        return model_class(num_users, num_itens, params['latent_factors'])
    
    if model_type == 'AutoEncoder':
        if params is None or 'embedding_dim' not in params or 'hidden_dim' not in params:
            return model_class(num_users, num_itens, embedding_dim =20 , hidden_dim=64)
    
        return model_class(num_users, num_itens, params['embedding_dim'], params['hidden_dim'])

    if model_type == 'MLP':
        if params is None or 'embedding_dim' not in params or 'hidden_dims' not in params:
            return model_class(num_users, num_itens, embedding_dim = 20, hidden_dims=[64,32])
    
        return model_class(num_users, num_itens, params['embedding_dim'], params['hidden_dims'])

    if model_type == 'RNN':
        if params is None or 'embedding_dim' not in params or 'hidden_dim' not in params or 'num_layers' not in params:
            return model_class(num_users, num_itens, embedding_dim = 20, hidden_dim=50, num_layers=1)
    
        return model_class(num_users, num_itens, params['embedding_dim'], params['hidden_dim'], params['num_layers'])
    return 0

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
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

        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
        
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model
