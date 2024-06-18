import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_users, num_itens, embedding_dim=20, hidden_dims=[64, 32]):
        super(MLP, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_itens, embedding_dim)

        input_size = embedding_dim * 2
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_size, hidden_dim))
            layers.append(nn.ReLU())
            input_size = hidden_dim

        layers.append(nn.Linear(input_size, 1))

        self.fc_layers = nn.Sequential(*layers)

        '''
        self.fc_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        '''

    def forward(self, user, item):
        user_embedded = self.user_embedding(user)
        item_embedded = self.item_embedding(item)

        x = torch.cat([user_embedded, item_embedded], dim=1)
        output = self.fc_layers(x).squeeze()
        return output