# -*- coding: utf-8 -*-
"""
==========================================================================
MÓDULO 2: PIPELINE DE PRÉ-PROCESSAMENTO DOS DADOS
Responsabilidade: Cleiton Mizael
==========================================================================
Este módulo é responsável por estruturar o pipeline de pré-processamento,
garantindo que não ocorra vazamento de dados (data leakage) entre treino,
validação e teste.

O pipeline realiza as seguintes tarefas de forma sequencial:
1. Remoção de atributos identificadores não informativos (nameOrig, nameDest, isFlaggedFraud).
2. Separação inicial dos dados: 80% para treino/validação e 20% para teste final.
3. Imputação de valores faltantes (média para numéricos, moda para categóricos).
4. Codificação de atributos categóricos em numéricos (One-Hot Encoding para a coluna 'type').
5. Normalização (Z-score com StandardScaler) para todos os atributos preditores.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def obter_preditores_e_alvo(df):
    """
    Separa os dados em variaveis preditoras (X) e a variável alvo (y).
    Remove atributos que não agregam valor preditivo para classificação geral.
    """
    # nameOrig e nameDest são IDs únicos de contas e geram alta cardinalidade (causariam overfitting)
    # isFlaggedFraud é uma regra de negócio simples preexistente, não um atributo preditivo da transação
    colunas_para_remover = ["nameOrig", "nameDest", "isFlaggedFraud", "isFraud"]
    
    # Garantir que removemos apenas as colunas que realmente existem no DataFrame
    colunas_existentes_remover = [col for col in colunas_para_remover if col in df.columns]
    
    X = df.drop(columns=colunas_existentes_remover)
    y = df["isFraud"].values if "isFraud" in df.columns else None
    
    return X, y

def construir_pipeline(colunas_numericas, colunas_categoricas):
    """
    Constrói a estrutura do pipeline utilizando o ColumnTransformer do scikit-learn.
    Isso assegura que as transformações se ajustem apenas aos dados de treino (fit)
    e sejam apenas aplicadas (transform) aos de validação/teste.
    """
    # 1. Pipeline para atributos numéricos
    # Imputador substitui possíveis NaNs pela média da coluna
    #StandardScaler normaliza os dados (Média = 0, Desvio Padrão = 1)
    pipeline_numerico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    
    # 2. Pipeline para atributos categóricos (como a coluna 'type')
    # Imputador substitui NaNs pelo valor mais frequente (moda)
    # OneHotEncoder transforma a variável categórica em colunas binárias (0 ou 1)
    pipeline_categorico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # 3. Combinador de transformações por tipo de coluna
    preprocessador = ColumnTransformer(
        transformers=[
            ("num", pipeline_numerico, colunas_numericas),
            ("cat", pipeline_categorico, colunas_categoricas)
        ]
    )
    
    return preprocessador

def executar_pre_processamento(df_amostra, semente_matricula):
    """
    Função principal que executa a divisão de dados e o pipeline de pré-processamento.
    Retorna os dados limpos, as colunas após o one-hot encoding e os conjuntos divididos.
    """
    print("-" * 70)
    print("MÓDULO 2 (Cleiton Mizael) - Executando pré-processamento...")
    print("-" * 70)
    
    # 1. Obter X e y
    X, y = obter_preditores_e_alvo(df_amostra)
    
    # Identificar automaticamente quais colunas são numéricas e quais são categóricas
    colunas_numericas = X.select_dtypes(include=[np.number]).columns.tolist()
    colunas_categoricas = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"[INFO] Atributos numéricos detectados: {colunas_numericas}")
    print(f"[INFO] Atributos categóricos detectados: {colunas_categoricas}")
    
    # 2. Divisão dos dados: 80% treino/validação e 20% teste final
    # A divisão é estratificada (stratify=y) para manter a proporção de 1/6 de fraudes em ambos
    # Normalizar a semente (matrícula) para o intervalo aceito pelo scikit-learn
    # (int entre 0 e 2**32-1). Isso permite usar números grandes de matrícula.
    seed_32 = int(semente_matricula) % (2**32)
    X_treino_val, X_teste_final, y_treino_val, y_teste_final = train_test_split(
        X, y,
        test_size=0.20,
        random_state=seed_32,
        stratify=y
    )
    
    print(f"[INFO] Divisão concluída (20% para teste final):")
    print(f"       - Treino/Validação: {X_treino_val.shape[0]} amostras")
    print(f"       - Teste Final: {X_teste_final.shape[0]} amostras")
    
    # 3. Construir o pipeline e ajustá-lo APENAS nos dados de treino/validação
    preprocessor = construir_pipeline(colunas_numericas, colunas_categoricas)
    
    #O ajuste (fit) aprende a média/desvio-padrão e os valores categóricos do Treino
    preprocessor.fit(X_treino_val)
    
    # 4. Transformar os conjuntos de dados (Treino/Val e Teste Final)
    X_treino_val_processado = preprocessor.transform(X_treino_val)
    X_teste_final_processado = preprocessor.transform(X_teste_final)
    
    # 5. Obter os nomes das colunas após o One-Hot Encoding (útil para análise posterior)
    nomes_colunas_cat = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(colunas_categoricas)
    colunas_finais = colunas_numericas + list(nomes_colunas_cat)
    
    print(f"[SUCESSO] Dados pré-processados! Quantidade de atributos finais: {len(colunas_finais)}")
    print(f"[INFO] Atributos resultantes: {colunas_finais}")
    print("-" * 50)
    
    return {
        "X_treino_val": X_treino_val_processado,
        "y_treino_val": y_treino_val,
        "X_teste_final": X_teste_final_processado,
        "y_teste_final": y_teste_final,
        "colunas_finais": colunas_finais,
        "pipeline_ajustado": preprocessor,
        # Índices originais das amostras antes do processamento (úteis para mapear ao df bruto)
        "idx_treino_val": X_treino_val.index,
        "idx_teste_final": X_teste_final.index
    }

if __name__ == "__main__":
    # Teste de execução rapida
    from data_loader import baixar_e_preparar_dados, MATRICULA
    amostra = baixar_e_preparar_dados()
    if amostra is not None:
        executar_pre_processamento(amostra, MATRICULA)
