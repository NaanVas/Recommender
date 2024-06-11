import torch
import torch.nn as nn
import torch.optim as optim

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=20):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user, item):
        user_embedded = self.user_embedding(user)
        item_embedded = self.item_embedding(item)
        user_bias = self.user_bias(user).squeeze()
        item_bias = self.item_bias(item).squeeze()
        dot_product = (user_embedded * item_embedded).sum(1)
        prediction = dot_product + user_bias + item_bias
        return prediction
