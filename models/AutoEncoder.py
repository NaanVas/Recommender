import torch
import torch.nn as nn

'''
Objetivo principal desse modelo é prever a avaliação que um usuário daria a um item específico, com base
nos padrões aprendidos a partir de dados de avaliações passadas
'''
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
    
'''
Embeddings: são representações compactas de usuários e itens. A classe cria uma representação única para cada usuário e cada item. Ele converte ususarios e itens em conjuntos de números que a rede neural pode entender
Encoder: Esta parte da rede pega as representações dos usuarios e itens e combina essas informações em um novo vetor que representa a relação entre o usuario e item
    Ao concatenar o vetor de usuario e itens, fica um vetor de embedding_dim * 2. A primeira camada linear recebe o vetor de embedding_dim*2 e tranforma em um vetor de hidden_dim. Depois o vetor passa pela funcao de ativacao ReLU introduzindo a não linearidade no modelo. Dps o vetor de hidden_dim é transformado novamente em um vetor de embedding_dim*2  dimensoes
Decoder: Esta parte da rede pega a combinacoa de informações do encoder e usa para prever a avaliacao do usuario para o item
    Acontece a mesma coisa que no Encoder so que ao inves de trasnformar o vetor para embedding_dim * 2 de novo, ele transforma em um vetor de 1 dimensao, que seria a avaliação prevista
Previsão: Os ids do usuarios e itens sao convertido em vetores, esses vetores sao concatenados e preprocessador pelo encoder que gera uma nova representação, essa nova representação é entao processada pelo decoder para gerar a avaliacao prevista
'''