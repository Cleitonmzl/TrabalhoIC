# Detecção de Fraudes Financeiras com Perceptron Multicamadas (MLP)

## Identificação do Projeto

*   **Disciplina:** Inteligência Computacional
*   **Docente:** Prof. Dr. Antonino Alves Feitosa Neto
*   **Discentes:** Pedro Davi e Cleiton
*   **Contexto:** Trabalho prático de desenvolvimento e análise experimental de redes neurais artificiais

---

## Descrição do Projeto

Este projeto consiste no desenvolvimento e na avaliação experimental de uma rede neural artificial do tipo Perceptron Multicamadas (MLP), implementada com o framework PyTorch, voltada para a classificação binária de transações financeiras fraudulentas utilizando a base de dados PaySim.

O problema de detecção de fraudes é caracterizado por um forte desbalanceamento de classes. Para contornar essa limitação e garantir a reprodutibilidade dos testes, o pipeline do projeto trabalha sobre uma amostragem estratificada dos dados. O projeto abrange desde o pré-processamento estruturado até a análise comparativa de diferentes configurações de rede e otimizadores.

### Principais Etapas Desenvolvidas

1.  **Pré-processamento e Pipeline de Dados:** Limpeza, imputação de dados faltantes (numéricos e categóricos) e codificação de variáveis qualitativas via One-Hot Encoding. Os atributos contínuos são normalizados por Z-score (padronização), isolando-se o conjunto de teste de forma estrita para evitar vazamento de dados (*data leakage*).
2.  **Modelagem Arquitetural:** Implementação da classe da MLP em PyTorch, suportando topologias flexíveis (múltiplas camadas ocultas) e aplicação de regularização por Dropout para atenuar o sobreajuste (*overfitting*).
3.  **Metodologia Experimental (Estudo de Fatores):** Avaliação do impacto da combinação de fatores estruturais no desempenho do classificador (F1-Score):
    *   **Topologias das camadas ocultas:** Comparação entre uma estrutura simples de uma única camada e uma estrutura mais profunda com duas camadas ocultas.
    *   **Funções de ativação:** Análise comparativa entre as funções ReLU e GELU.
4.  **Otimização e Algoritmos de Aprendizado:** Estudo empírico comparando o comportamento e o custo computacional de diferentes abordagens de descida de gradiente:
    *   Batch Gradient Descent (Batch GD)
    *   Mini-batch Gradient Descent (com lotes de tamanhos variados)
    *   Stochastic Gradient Descent (SGD)
5.  **Análise de Desempenho e Generalização:** Treinamento do modelo final otimizado e validação robusta por meio de métricas como Acurácia, Precisão, Recall e F1-Score no conjunto de teste independente.

---
