import torch
import torch.nn as nn
'''
Esse modelo usa uma rede neural com varias camadas para aprender as relações complexas entre usuarios e itens,
e fazer previsoes sobre como os usarios avaliarao novos itens. As camadas intermediarias ajudam a capturar e
refinar as caracteristicas relevantes dos usuarios e dos itens, resultando em uma previsao mais precisa
'''
class MLP(nn.Module):
    def __init__(self, num_users, num_itens, embedding_dim=20, hidden_dims=[64, 32]):
        super(MLP, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_itens, embedding_dim)

        input_size = embedding_dim * 2
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_size, hidden_dim))
            layers.append(nn.ReLU())
            input_size = hidden_dim

        layers.append(nn.Linear(input_size, 1))

        self.fc_layers = nn.Sequential(*layers)

        '''
        self.fc_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        '''

    def forward(self, user, item):
        user_embedded = self.user_embedding(user)
        item_embedded = self.item_embedding(item)

        x = torch.cat([user_embedded, item_embedded], dim=1)
        output = self.fc_layers(x).squeeze()
        return output
    
'''
Embeddings: São representação compactas de usuarios e itens. A classe cria uma representação unica para cada usuario e cada item, convertendo-os em conjuntos de numeros que a rede neural pode entender
Camadas de rede neural: São varias camdas de neuronios que processam as representações dos usuarios e itens, transformando-as passo a passo até chegar a uma previsão final. Essas camadas são chamadas de Fully Connected porque cada neuronio em uma camada esta conectado a todos os neuronios na proxima camada
    Os embeddings sao concatenados em um unico valor. O vetor cocatenado é passado pela primeira camada linear, cada neuronio dessa camada recebe como entrada todos os valores do vetor concatenado. cada neuronio realiza uma combinação linear das entradas e apos a combinação utiliza uma funcao de ativação ReLU. Se houver mais camadas no modelo o processo se repete, o resultado da primeira camda é passado para a proxima camada linear, onde é feito os mesmos passos. Na ultima camada tem apenas um neuronio, pois representa a prvisão unica do usuario para o item
Previsão: Os ids dos usuarios e iten sao convertidos em vetores, dps combinados em um unico vetor. O vetor combinado é passado por arias camadas de neuronios, onde cada camada aplica transformações e ativações para refinar a informação. A ultima camada gera a previsao da avaliacao do usuario para o item 
'''