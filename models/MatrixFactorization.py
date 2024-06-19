import torch
import torch.nn as nn

'''
É um modelo de rede neural simples projetado para prever a avaliação que um usuario daria a um item especifico
com base em tecnicas de fatoração de matriz para decompor a relação entre usuarios e itens em componentes mais
simples e fazer previsões sobre como os usuarios avaliarão novos itens
'''
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

'''
Embeddings: São representacoes compactas de usuarios e itens. A classe cria uma representação unica para cada usuario e cada item, convertendo-os em conjuntos de números que a rede neural pode entender
Bias(Vies): São ajustes adicionais que a rede faz para cada usuario e cada item para melhorar a precisao das previsoes, como se fosse um ajuste fino que ajuda a capturar as preferências individuais dos usuario e as caracteristicas especificas dos itens
Predição: O modelo converte os ids dos usuarios e itens em representações numericas (embedding) usando os embedding de usuarios e itens, esses vetoes sao multiplicados entre si e a soma dos resultados é calculada, o modelo adiciona vieses especificos de cada usuario e item ao resultado, o resultado final é a previsão da avaliação do usuario para o item

'''