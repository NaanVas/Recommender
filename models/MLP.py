import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_users, num_itens, embedding_dim=20, hidden_dim=64):
        super(MLP, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_itens, embedding_dim)

        self.fc_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, user, item):
        user_embedded = self.user_embedding(user)
        item_embedded = self.item_embedding(item)

        x = torch.cat([user_embedded, item_embedded], dim=1)
        output = self.fc_layer(x).squeeze()
        return output