import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_users, num_itens, embedding_dim, num_filters):
        super(CNN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_itens, embedding_dim)
        self.conv = nn.Conv2d(1, num_filters, (1, embedding_dim * 2)) #Como esta lidando com dados de usuario e item como canais separados, o numero de canais tem que ser 1, para o modelo aprender as representações distinstas para o usuario e item durante a convolução, se fosse imagens por exemplo seria 3 canais (RGB) [1 canal de entrada, num_filters filtros, (altura_kernel=1, largura_kernel=embedding_dim * 2)]
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, user, item):
        user_embedded = self.user_embedding(user).unsqueeze(1) #[batch_size, 1, embedding_dim]
        item_embedded = self.item_embedding(item).unsqueeze(1) #[batch_size, 1, embedding_dim]
        x = torch.cat([user_embedded, item_embedded], dim=2) #[batch_size, 1, 2 * embedding_dim]
        x = x.unsqueeze(1) #[batch_size, 1, 1, 2 * embedding_dim]

        x = self.conv(x) #[batch_size, num_filters, 1, novo_width]
        x = torch.relu(x) 
        x = x.squeeze(3) #[batch_size, num_filters, 1]
        x = nn.functional.max_pool1d(x, x.size(2)).squeeze(2) #[batch_size, num_filters]
        x = self.fc(x) #[batch_size, 1]
        x = x.squeeze(1) #[batch_size]
        return x