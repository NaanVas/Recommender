import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, num_users, num_itens, embedding_dim=20, hidden_dim=512):
        super(AutoEncoder, self).__init__()
        self.user_embeding = nn.Embedding(num_users, embedding_dim)
        self.item_embeding = nn.Embedding(num_itens, embedding_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, users_ids, itens_ids):
        user_embedded = self.user_embeding(users_ids)
        item_embedded = self.item_embeding(itens_ids)
        x = torch.cat([user_embedded, item_embedded], dim=1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.squeeze()