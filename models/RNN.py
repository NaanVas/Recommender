import torch
import torch.nn as nn

'''
A RNN é util para lidar com sequencia de dados, capturando padroes que dependem da ordem dos elementos na sequencia
Ela mantem uma memoria interna que premite considerar o contexto anterior ao fazer previsoes
O modelo RNN usa enbeddings para representar ususario e itens, e uma RNN para aprender padreso sequenciais nos dados
de avaliações. Isso permite que o modelo faça previsoes considerando o historico de avaliações ao longo do tempo
'''

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
        output, _ = self.rnn(x) # _ é o estado oculto final da RNN, que representa a presentação latente ou a "memoria" da RNN
        output = output[:, -1, :] # ':' operador de fatiamento significa "todos os elementos", '-1' indica que estamos pegando o ultimo elemento ao longo de uma dimensao. Neste caso o indice -1 se refere a ultima etapa de tempo da sequencia
        output = self.fc(output).squeeze()
        return output
    
'''
Embeddings: São representações compactas de usuarios e itens. Cada usuario é convertido em um vetor numerico que a rede neural pode entender
RNN: é uma arquitetura de rede neural que processa dados sequenciais, como séries temporais ou sequencias de palavras. Nesse caso, a RNN é usada para capturar padões sequenciais nos dados de avaliações
Processo de aprendizado: O modelo recebe sequencias de avaliações de usuario para itens ao longo do tempo, a cada passo da sequencia, o modelo combina as representações dos usuarios e itens usando embeddings. A RNN processa essa combinação sequencialmente, aprendendo padrões sequenciais nos dados. A saída final da RNN é utiliza para fazer a previsao da avaliacao do usuario para o item
Previsão: Fornece os ids do usuario e do item, os embeddings correpondentes ao usuario e ao item sao combinados e passados para RNN, a RNN processa essa combinacao ao longo de toda a sequencia(em geral, é uma unica sequencia para cada par de usuario item). A saida final da RNN é passada por uma camada linear (fc) para gerar a previsao da avaliacao
'''

'''
Exemplo de  [:,-1,:]:
    entrada:
    [[[1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20]],

    [[21, 22, 23, 24, 25],
    [26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35],
    [36, 37, 38, 39, 40]],

    [[41, 42, 43, 44, 45],
    [46, 47, 48, 49, 50],
    [51, 52, 53, 54, 55],
    [56, 57, 58, 59, 60]]]

    saida:
    [[16, 17, 18, 19, 20],
    [36, 37, 38, 39, 40],
    [56, 57, 58, 59, 60]]
'''