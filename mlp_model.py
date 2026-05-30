# -*- coding: utf-8 -*-
"""
==========================================================================
MÓDULO 3: MODELAGEM DA REDE NEURAL MLP (Pytorch)
Responsabilidade: Cleiton Mizael
==========================================================================
Este módulo contém a definição da arquitetura da rede neural Perceptron
Multicamadas (MLP) utilizando o framework PyTorch.

A arquitetura foi projetada para ser flexível e permitir a parametrização de:
- Dimensão de entrada (número de atributos).
- Topologias customizadas (lista contendo o número de neurônios de cada camada oculta).
- Funções de ativação ReLU ou GELU nas camadas ocultas.
- Camadas de Dropout para prevenção de overfitting (regularização).
- Uma camada de saída linear simples de 1 neurônio, com ativação Sigmoid para
  produzir a probabilidade da transação ser fraude (classificação binária).
"""

import torch
import torch.nn as nn

class FraudMLP(nn.Module):
    """
    Classe que representa a rede neural MLP para classificação binária de fraudes.
    Herda de nn.Module do PyTorch.
    """
    def __init__(self, input_dim, hidden_topology=[10], activation_fn="ReLU", dropout_rate=0.2):
        """
        Construtor da rede neural MLP.
        
        Parâmetros:
        -----------
        input_dim : int
            Número de atributos na entrada (após o pré-processamento/one-hot encoding).
        hidden_topology : list of int
            Estrutura das camadas ocultas. Exemplo: [10] significa uma única camada oculta
            com 10 neurônios; [32, 16] significa duas camadas, com 32 e 16 neurônios.
        activation_fn : str
            Nome da função de ativação nas camadas ocultas. Opções: 'ReLU' ou 'GELU'.
        dropout_rate : float
            Fração de neurônios que serão desativados aleatoriamente durante o treino
            para evitar overfitting. Intervalo: [0, 1).
        """
        super(FraudMLP, self).__init__()
        
        # Guardar parâmetros para consulta e exibição posterior
        self.input_dim = input_dim
        self.hidden_topology = hidden_topology
        self.activation_fn_name = activation_fn
        self.dropout_rate = dropout_rate
        
        # Definir a função de ativação das camadas ocultas
        if activation_fn.upper() == "RELU":
            self.activation = nn.ReLU()
        elif activation_fn.upper() == "GELU":
            self.activation = nn.GELU()
        else:
            raise ValueError("Função de ativação não suportada. Escolha entre 'ReLU' ou 'GELU'.")
            
        # Lista onde iremos empilhar as camadas
        layers = []
        
        # 1. Construir as camadas ocultas dinamicamente
        in_features = input_dim
        for neurons in hidden_topology:
            # Camada linear (conecta todas as entradas com as saídas)
            layers.append(nn.Linear(in_features, neurons))
            # Função de ativação não linear
            layers.append(self.activation)
            # Camada de Dropout para regularização
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            # A próxima camada terá como entrada o número de neurônios da camada atual
            in_features = neurons
            
        # 2. Camada de Saída
        # Classificação binária: mapeia a última camada oculta para 1 único neurônio de saída
        # NOTA: NÃO aplicamos Sigmoid aqui - usaremos BCEWithLogitsLoss no treinamento
        # que combina sigmoid + BCE de forma numérica mais estável e permite pos_weight.
        layers.append(nn.Linear(in_features, 1))
        
        # nn.Sequential agrupa a nossa lista de camadas em um único bloco de execução sequencial
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Executa a passagem para a frente (feedforward) na rede neural.
        Calcula as saídas a partir dos tensores de entrada.
        """
        # Garante que os dados de entrada estejam no formato correto (float32)
        x = x.to(torch.float32)
        return self.network(x)

    def extra_repr(self):
        """
        Exibição humanizada de informações adicionais da topologia do modelo.
        """
        return (f"Topology: input({self.input_dim}) -> "
                f"{' -> '.join(map(str, self.hidden_topology))} -> output(1)\n"
                f"Activation: {self.activation_fn_name}, Dropout Rate: {self.dropout_rate}")

if __name__ == "__main__":
    # Teste rápido de criação do modelo para verificar se está funcionando
    print("-" * 70)
    print("MÓDULO 3 (Pedro Davi) - Testando modelagem da rede neural...")
    print("-" * 70)
    
    # Criar uma rede para entrada de 10 atributos e topologia [32, 16]
    modelo_teste = FraudMLP(input_dim=10, hidden_topology=[32, 16], activation_fn="GELU", dropout_rate=0.2)
    print(modelo_teste)
    
    # Simular uma entrada com 5 amostras e 10 colunas
    entrada_dummy = torch.randn(5, 10)
    saida = modelo_teste(entrada_dummy)
    print("\nSaída de teste (deve conter 5 valores entre 0 e 1):")
    print(saida)
    print("-" * 50)
