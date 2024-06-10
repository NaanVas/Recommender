import torch
import torch.nn as nn

class NMF(nn.Module):
    def __init__(self, num_users, num_itens, latent_factors):
        super(NMF, self).__init__()
        self.user_factors = nn.Parameter(torch.rand(num_users, latent_factors))
        self.item_factors = nn.Parameter(torch.rand(num_itens, latent_factors))

    def forward(self, user_ids, item_ids):
        user_embeddings = self.user_factors[user_ids]
        item_embeddings = self.item_factors[item_ids]
        return torch.sum(user_embeddings * item_embeddings, dim=1)