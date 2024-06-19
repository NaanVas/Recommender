import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, num_users, num_itens, embedding_dim=20, hidden_dim=50, num_layers=1):
        super(RNN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_itens, embedding_dim)
        self.rnn = nn.RNN(embedding_dim * 2, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, user, item):
        user_embedded = self.user_embedding(user)
        item_embedded = self.item_embedding(item)
        x = torch.cat([user_embedded, item_embedded], dim=1).unsqueeze(1)
        output, _ = self.rnn(x)
        output = output[:, -1, :]
        output = self.fc(output).squeeze()
        return output
    
    