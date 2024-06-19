import torch
import torch.nn as nn

'''
O modelo NMF utiliza fatores latentes para representar padroes nos dados de avaliacoes de usuarios para itens
e faz previsoes de avaliações com base na similaridade entre esses fatores latentes. Ele é util quando deseja
aprender padrões complexos nos dados de avaliações mantendo as previsões e fatores sempre positivos
'''
class NMF(nn.Module):
    def __init__(self, num_users, num_itens, latent_factors):
        super(NMF, self).__init__()
        self.user_factors = nn.Parameter(torch.rand(num_users, latent_factors))
        self.item_factors = nn.Parameter(torch.rand(num_itens, latent_factors))

    def forward(self, user_ids, item_ids):
        user_embeddings = self.user_factors[user_ids]
        item_embeddings = self.item_factors[item_ids]
        return torch.sum(user_embeddings * item_embeddings, dim=1)
    
'''
Fatores latentes: São caracteristicas ocultas que o modelo tenta aprender nos dados. Esses fatores representam padrôes e preferencias dos usuarios e caracteriscas dos itens de forma compacta
Processo de aprendizado: O modelo possui dois conjuntos de fatores latentes: um para usuarios e outro para o itens. Durante o treinamento, o modelo ajusta esses fatores latentes para minimiza a diferença entre as avaliacoes reais dos usuarios e as previsoes feitas pelo modelo
Previsâo: Passando os ids do usuario e do item o modelo acessa os fatores latentes correspondentes a partir dos parametros que foram aprendidos durante o treimanento, o modelo calcula a similaridade entre os fatores latentes do usuario e do item, multiplicando-os e somando-os, essa similaridade representa a previsao da avaliacao do usuario para o item
'''