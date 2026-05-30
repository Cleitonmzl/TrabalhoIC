# -*- coding: utf-8 -*-
"""
==========================================================================
SCRIPT EXECUTOR DE EXPERIMENTOS E GERAÇÃO DE GRÁFICOS
Responsabilidade: Cleiton Mizael (com apoio de Pedro Davi)
==========================================================================
Este script orquestra a execução completa de todos os experimentos exigidos.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Importar nossos módulos locais
from data_loader import baixar_e_preparar_dados, MATRICULA
from preprocessor import executar_pre_processamento, obter_preditores_e_alvo
from mlp_model import FraudMLP
from trainer import executar_validacao_cruzada, treinar_modelo, avaliar_modelo, definir_sementes

# Configuração de Estilo para os Gráficos
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 16
})

# Sementes aleatórias controladas para as 10 execuções repetidas
SEMENTES = list(range(42, 52))

def rodar_experimento_fatores(df_amostra):
    print("\n" + "=" * 80)
    print("INICIANDO EXPERIMENTO 1: ANÁLISE DOS 2 FATORES (TOPOLOGIA x ATIVAÇÃO)")
    print("=" * 80)
    
    # CORREÇÃO: Aplicando o módulo na matrícula para não estourar o limite de 32-bits
    seed_32 = int(MATRICULA) % (2**32)
    
    X_bruto, y_bruto = obter_preditores_e_alvo(df_amostra)
    X_treino_val_cru, _, y_treino_val_array, _ = train_test_split(
        X_bruto, y_bruto,
        test_size=0.20,
        random_state=seed_32,
        stratify=y_bruto
    )
    
    topologias = {"1_camada_[10]": [10], "2_camadas_[32,16]": [32, 16]}
    ativacoes = ["ReLU", "GELU"]
    resultados_grid = []
    
    for nome_topo, topo in topologias.items():
        for ativa in ativacoes:
            config_label = f"{nome_topo}_{ativa}"
            print(f"\n>>> Testando Configuração: Topologia={topo}, Ativação={ativa}")
            
            f1_runs = []
            acc_runs = []
            prec_runs = []
            rec_runs = []
            tempos_runs = []
            historicos_runs = []
            
            for i, semente in enumerate(SEMENTES):
                print(f"    - Execução {i+1}/10 (Semente: {semente})... ", end="")
                
                # Batch de 64 fixo. Taxa de 0.001 para estabilidade.
                res_cv = executar_validacao_cruzada(
                    X_treino_val_cru=X_treino_val_cru,
                    y_treino_val=y_treino_val_array,
                    pipeline_preprocessador=None,
                    topologia=topo,
                    ativacao=ativa,
                    batch_size=64,
                    lr=0.001,
                    max_epocas=100,
                    patience=10,
                    semente=semente
                )
                
                f1_runs.append(res_cv["f1_score"])
                acc_runs.append(res_cv["accuracy"])
                prec_runs.append(res_cv["precision"])
                rec_runs.append(res_cv["recall"])
                tempos_runs.append(res_cv["tempo_processamento"])
                
                if i == 0:
                    historicos_runs = res_cv["historicos"]
                
                print("F1-score Médio: {:.4f}".format(res_cv["f1_score"]))
                
            resultados_grid.append({
                "configuracao": config_label,
                "topologia": str(topo),
                "ativacao": ativa,
                "f1_runs": f1_runs,
                "acc_runs": acc_runs,
                "prec_runs": prec_runs,
                "rec_runs": rec_runs,
                "tempos_runs": tempos_runs,
                "f1_medio": np.mean(f1_runs),
                "f1_desvio": np.std(f1_runs),
                "acc_medio": np.mean(acc_runs),
                "prec_medio": np.mean(prec_runs),
                "rec_medio": np.mean(rec_runs),
                "tempo_medio": np.mean(tempos_runs),
                "historicos_primeira_rodada": historicos_runs
            })
            
    return pd.DataFrame(resultados_grid)

def rodar_experimento_gradiente(df_amostra, melhor_topo, melhor_ativa):
    print("\n" + "=" * 80)
    print("INICIANDO EXPERIMENTO 2: COMPARAÇÃO DAS ESTRATÉGIAS DE GRADIENTE")
    print(f"Utilizando a melhor configuração encontrada: Topologia={melhor_topo}, Ativação={melhor_ativa}")
    print("=" * 80)
    
    # CORREÇÃO: Aplicando o módulo na matrícula para não estourar o limite de 32-bits
    seed_32 = int(MATRICULA) % (2**32)
    
    X_bruto, y_bruto = obter_preditores_e_alvo(df_amostra)
    X_treino_val_cru, _, y_treino_val_array, _ = train_test_split(
        X_bruto, y_bruto,
        test_size=0.20,
        random_state=seed_32,
        stratify=y_bruto
    )
    
    estrategias = {
        "Batch GD": "BATCH",
        "Mini-batch GD (32)": 32,
        "Mini-batch GD (128)": 128,
        "Stochastic GD": 1
    }
    
    resultados_gradiente = []
    
    for nome_est, bs in estrategias.items():
        print(f"\n>>> Testando Estratégia: {nome_est}")
        
        f1_runs = []
        acc_runs = []
        prec_runs = []
        rec_runs = []
        tempos_runs = []
        
        max_epocas_rodada = 15 if bs == 1 else 100
        patience_rodada = 5 if bs == 1 else 10
        
        for i, semente in enumerate(SEMENTES):
            if (bs == 1 or bs == "BATCH") and i >= 3:
                break
                
            print(f"    - Execução {i+1} (Semente: {semente})... ", end="")
            
            # Taxa menor para o Stochastic (bs=1) para evitar explosão
            lr_dinamico = 0.001 if bs == 1 else 0.01
            
            res_cv = executar_validacao_cruzada(
                X_treino_val_cru=X_treino_val_cru,
                y_treino_val=y_treino_val_array,
                pipeline_preprocessador=None,
                topologia=melhor_topo,
                ativacao=melhor_ativa,
                batch_size=bs,
                lr=lr_dinamico,
                max_epocas=max_epocas_rodada,
                patience=patience_rodada,
                semente=semente
            )
            
            f1_runs.append(res_cv["f1_score"])
            acc_runs.append(res_cv["accuracy"])
            prec_runs.append(res_cv["precision"])
            rec_runs.append(res_cv["recall"])
            tempos_runs.append(res_cv["tempo_processamento"])
            print("F1-score Médio: {:.4f}".format(res_cv["f1_score"]))
            
        resultados_gradiente.append({
            "estrategia": nome_est,
            "f1_medio": np.mean(f1_runs),
            "acc_medio": np.mean(acc_runs),
            "prec_medio": np.mean(prec_runs),
            "rec_medio": np.mean(rec_runs),
            "tempo_medio": np.mean(tempos_runs)
        })
        
    return pd.DataFrame(resultados_gradiente)

def gerar_graficos_e_tabelas(df_resultados_fatores, df_resultados_grad, prep_dados, melhor_modelo_treinado):
    print("\n" + "=" * 80)
    print("GERANDO GRÁFICOS E TABELAS DO RELATÓRIO")
    print("=" * 80)
    
    plt.figure(figsize=(10, 6))
    dados_boxplot = []
    labels_boxplot = []
    for _, row in df_resultados_fatores.iterrows():
        dados_boxplot.append(row["f1_runs"])
        labels_boxplot.append(row["configuracao"])
    
    try:
        plt.boxplot(dados_boxplot, tick_labels=labels_boxplot, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red', linewidth=2))
    except TypeError:
        plt.boxplot(dados_boxplot, labels=labels_boxplot, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red', linewidth=2))

    plt.title("Distribuição do F1-score por Configuração (10 Execuções)")
    plt.ylabel("F1-score")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("boxplot_fatores.png", dpi=150)
    plt.close()
    print("[GRAFICO] 'boxplot_fatores.png' gerado com sucesso.")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.barplot(data=df_resultados_fatores, x="configuracao", y="f1_medio", ax=axes[0], hue="configuracao", palette="viridis", legend=False)
    axes[0].set_title("Comparação de F1-score Médio")
    axes[0].set_ylabel("F1-score")
    axes[0].tick_params(axis='x', rotation=15)
    
    sns.barplot(data=df_resultados_fatores, x="configuracao", y="tempo_medio", ax=axes[1], hue="configuracao", palette="magma", legend=False)
    axes[1].set_title("Tempo de Processamento Médio por CV (Segundos)")
    axes[1].set_ylabel("Tempo (s)")
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig("colunas_comparacao_fatores.png", dpi=150)
    plt.close()
    print("[GRAFICO] 'colunas_comparacao_fatores.png' gerado com sucesso.")
    
    best_idx = df_resultados_fatores["f1_medio"].idxmax()
    historicos = df_resultados_fatores.loc[best_idx, "historicos_primeira_rodada"]
    hist = historicos[0]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist["treino_acc"], label="Treino", color="blue", linewidth=2)
    plt.plot(hist["val_acc"], label="Validação", color="orange", linewidth=2)
    plt.title("Evolução da Acurácia por Época")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(hist["treino_f1"], label="Treino", color="blue", linewidth=2)
    plt.plot(hist["val_f1"], label="Validação", color="orange", linewidth=2)
    plt.title("Evolução do F1-score por Época")
    plt.xlabel("Época")
    plt.ylabel("F1-score")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("curvas_evolucao_treino.png", dpi=150)
    plt.close()
    print("[GRAFICO] 'curvas_evolucao_treino.png' gerado com sucesso.")
    
    X_teste_proc = prep_dados["X_teste_final"]
    y_teste = prep_dados["y_teste_final"]
    
    avaliacoes_teste = avaliar_modelo(melhor_modelo_treinado, X_teste_proc, y_teste)
    matriz_conf = confusion_matrix(y_teste, avaliacoes_teste["predictions"])
    
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz_conf, display_labels=["Normal", "Fraude"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Matriz de Confusão no Teste Final")
    plt.tight_layout()
    plt.savefig("matriz_confusao_teste.png", dpi=150)
    plt.close()
    print("[GRAFICO] 'matriz_confusao_teste.png' gerado com sucesso.")
    
    print("\n" + "=" * 80)
    print("TABELA COMPARATIVA DE CONFIGURAÇÕES (Fatores Experimentais):")
    print("=" * 80)
    tabela_res_fatores = df_resultados_fatores[[
        "configuracao", "acc_medio", "prec_medio", "rec_medio", "f1_medio", "tempo_medio"
    ]].rename(columns={
        "configuracao": "Configuração",
        "acc_medio": "Acurácia",
        "prec_medio": "Precisão",
        "rec_medio": "Recall",
        "f1_medio": "F1-Score",
        "tempo_medio": "Tempo (s)"
    })
    print(tabela_res_fatores.to_string(index=False))
    tabela_res_fatores.to_csv("tabela_metricas_fatores.csv", index=False)
    
    print("\n" + "=" * 80)
    print("TABELA COMPARATIVA DE ESTRATÉGIAS DE GRADIENTE DESCENDENTE:")
    print("=" * 80)
    tabela_res_grad = df_resultados_grad.rename(columns={
        "estrategia": "Estratégia GD",
        "acc_medio": "Acurácia",
        "prec_medio": "Precisão",
        "rec_medio": "Recall",
        "f1_medio": "F1-Score",
        "tempo_medio": "Tempo (s)"
    })
    print(tabela_res_grad.to_string(index=False))
    tabela_res_grad.to_csv("tabela_metricas_gradiente.csv", index=False)
    
    print("\n" + "=" * 80)
    print("MÉTRICAS DO MELHOR MODELO AVALIADO NO TESTE FINAL:")
    print("=" * 80)
    print("Acurácia no Teste:  {:.4f}".format(avaliacoes_teste["accuracy"]))
    print("Precisão no Teste:  {:.4f}".format(avaliacoes_teste["precision"]))
    print("Recall no Teste:    {:.4f}".format(avaliacoes_teste["recall"]))
    print("F1-score no Teste:  {:.4f}".format(avaliacoes_teste["f1_score"]))
    print("=" * 80)
    
    df_teste_res = pd.DataFrame([{
        "Métrica": ["Acurácia", "Precisão", "Recall", "F1-score"],
        "Valor no Teste Final": [
            avaliacoes_teste["accuracy"], 
            avaliacoes_teste["precision"], 
            avaliacoes_teste["recall"], 
            avaliacoes_teste["f1_score"]
        ]
    }])
    df_teste_res.to_csv("tabela_melhor_modelo_teste.csv", index=False)

def treinar_modelo_final_completo(df_amostra, melhor_topo, melhor_ativa):
    print(f"\n>>> Treinando Modelo Final Completo (Topologia={melhor_topo}, Ativação={melhor_ativa})...")
    
    prep_dados = executar_pre_processamento(df_amostra, MATRICULA)
    
    X_treino_val = prep_dados["X_treino_val"]
    y_treino_val = prep_dados["y_treino_val"]
    
    definir_sementes(MATRICULA)
    seed_32 = int(MATRICULA) % (2**32)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_treino_val, y_treino_val, test_size=0.15, random_state=seed_32, stratify=y_treino_val
    )
    
    tensor_x_tr = torch.tensor(X_tr, dtype=torch.float32)
    tensor_y_tr = torch.tensor(y_tr, dtype=torch.float32)
    tensor_x_val = torch.tensor(X_val, dtype=torch.float32)
    tensor_y_val = torch.tensor(y_val, dtype=torch.float32)
    
    dataset_tr = TensorDataset(tensor_x_tr, tensor_y_tr)
    dataset_val = TensorDataset(tensor_x_val, tensor_y_val)
    
    loader_tr = DataLoader(dataset_tr, batch_size=64, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=256, shuffle=False)
    
    input_dim = X_treino_val.shape[1]
    modelo = FraudMLP(input_dim=input_dim, hidden_topology=melhor_topo, 
                      activation_fn=melhor_ativa, dropout_rate=0.2)
    
    modelo_final, _, _ = treinar_modelo(
        modelo, loader_tr, loader_val, lr=0.001, max_epocas=150, patience=15
    )
    
    return modelo_final, prep_dados

def main():
    df_amostra = baixar_e_preparar_dados()
    if df_amostra is None:
        print("[ERRO] Não foi possível prosseguir pois os dados não foram carregados.")
        return
        
    # EXECUÇÃO COMPLETA: Todos os passos ativos
    df_res_fatores = rodar_experimento_fatores(df_amostra)
    
    melhor_config_idx = df_res_fatores["f1_medio"].idxmax()
    melhor_row = df_res_fatores.loc[melhor_config_idx]
    
    import ast
    melhor_topo = ast.literal_eval(melhor_row["topologia"]) 
    melhor_ativa = melhor_row["ativacao"]
    
    print(f"\n[VENCEDOR] A melhor configuração experimental foi: {melhor_row['configuracao']}")
    print(f"           F1-score CV Médio: {melhor_row['f1_medio']:.4f}")
    
    df_res_gradiente = rodar_experimento_gradiente(df_amostra, melhor_topo, melhor_ativa)
    
    melhor_modelo, prep_dados = treinar_modelo_final_completo(df_amostra, melhor_topo, melhor_ativa)
    
    gerar_graficos_e_tabelas(df_res_fatores, df_res_gradiente, prep_dados, melhor_modelo)
    
    print("\n" + "=" * 80)
    print("FIM DO SCRIPT DE EXPERIMENTOS. TODOS OS PRODUTOS FORAM GERADOS E SALVOS.")
    print("=" * 80)

if __name__ == "__main__":
    main()
