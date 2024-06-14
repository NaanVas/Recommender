import torch
import torch.nn as nn
import torch.optim as optim

'''
A tecnica SVD decompoe a iteração entre usuarios e itens em partes menos e significativas,
ajudando a entender e prever as preferencias ds usuarios de maneira eficiente
'''

class SVD(nn.Module):
    def __init__(self, num_users, num_itens, embedding_dim=20):
        super(SVD, self).__init__()
        self.U = nn.Parameter(torch.randn(num_users, embedding_dim))
        self.Sigma = nn.Parameter(torch.randn(embedding_dim))
        self.VT = nn.Parameter(torch.randn(embedding_dim, num_itens))
        self.user_bias = nn.Parameter(torch.randn(num_users))
        self.item_bias = nn.Parameter(torch.randn(num_itens))

    def forward(self, user, item):
        user_embedded = self.U[user]*self.Sigma
        item_embedded = self.VT[:,item].t()
        user_bias = self.user_bias[user]
        item_bias = self.item_bias[item]
        dot_product = (user_embedded * item_embedded).sum(1)
        prediction = dot_product + user_bias + item_bias
        return prediction

'''
U é a matriz  de usuarios de tamanho (num_users, embedding_dim), cada linha representa um usuario e cada coluna representa uma caracteristica oculta dos usuarios, ajudando a entender como cada usuario se relaciona com essas caracteristicas ocultas
Sigma é um vetor de singularidade de tamanho embedding_dim, é um vetor que contem os valores que representam a importancia de cada caracteristica oculta, ajusta a contribuicao de cada caracteristica oculta no processo de recomendacao
VT Matriz tramposta  de itens de tamanho emdding_dim, num_itens, cada coluna representa um item e cada linha representa uma caracteristica oculta dos itens, ajudando a entender como cada item se relaciona com essas caracteristicas ocultas

Forward
user_embedding é uma combinacao das caracteristicas de um usuario ajustadas pela importancia dessas caracteristicas, pegando as caracteristicas da matriz U e multiplicando pelos valores de sigma
item_embedding é a combinacao das caracteristicas de um item pegamos a caracateriscas do item da matriz VT e transpoe para ficar no formado da user
'''