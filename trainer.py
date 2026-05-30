# -*- coding: utf-8 -*-
"""
==========================================================================
MÓDULO 4: TREINAMENTO, VALIDAÇÃO CRUZADA E AVALIAÇÃO
Responsabilidade: Pedro Davi
==========================================================================
Este módulo é responsável por coordenar o treinamento da rede neural MLP,
a aplicação de Validação Cruzada Estratificada com 5 Folds e o Early Stopping.

Aqui implementamos:
- Métodos para controle de semente de aleatoriedade (reprodutibilidade).
- Um loop de treinamento robusto em PyTorch com o otimizador SGD.
- Suporte a diferentes estratégias de descida de gradiente:
  * Batch GD (lote = todo o conjunto de treino).
  * Mini-batch GD (lote de tamanho customizado, ex: 32, 64, 128).
  * Stochastic GD (lote = 1).
- Validação Cruzada Estratificada que ajusta o pipeline de pré-processamento
  dentro de cada fold para evitar o vazamento de dados.
- Early Stopping monitorando a perda na validação com um parâmetro de paciência.
"""

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from mlp_model import FraudMLP
from preprocessor import obter_preditores_e_alvo, construir_pipeline

def definir_sementes(semente):
    """
    Define as sementes aleatórias em todas as bibliotecas para
    garantir que todos os experimentos sejam reprodutíveis.
    """
    # Normalizar a semente para o intervalo aceito pelas bibliotecas
    # (int entre 0 e 2**32 - 1). Isso permite usar números grandes de matrícula.
    seed_32 = int(semente) % (2**32)
    random.seed(seed_32)
    np.random.seed(seed_32)
    torch.manual_seed(seed_32)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_32)
    # Garante comportamento determinístico em operações do PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def treinar_modelo(modelo, train_loader, val_loader, lr=0.001, max_epocas=200, patience=15, dispositivo="cpu"):
    """
    Realiza o loop de treinamento da MLP para uma única configuração de dados.
    Utiliza SGD como otimizador e a perda de entropia cruzada binária (BCE).
    """
    modelo = modelo.to(dispositivo)
    criterio = nn.BCELoss()
    otimizador = optim.SGD(modelo.parameters(), lr=lr, momentum=0.9)
    
    melhor_perda_val = float("inf")
    epocas_sem_melhoria = 0
    melhor_estado_modelo = None
    
    historico = {
        "treino_loss": [], "treino_acc": [], "treino_f1": [],
        "val_loss": [], "val_acc": [], "val_f1": []
    }
    
    tempo_inicio = time.time()
    
    for epoca in range(max_epocas):
        # 1. Fase de Treino
        modelo.train()
        perda_acumulada_treino = 0.0
        predicoes_treino = []
        alvos_treino = []
        
        for lote_x, lote_y in train_loader:
            lote_x, lote_y = lote_x.to(dispositivo), lote_y.to(dispositivo).float().unsqueeze(1)
            
            otimizador.zero_grad()
            saidas = modelo(lote_x)
            
            # --- BLINDAGEM MATEMÁTICA ---
            # 1. Se a saída não estiver entre 0 e 1, a Sigmoid foi removida do modelo. Aplicamos aqui.
            if saidas.min() < 0.0 or saidas.max() > 1.0:
                saidas = torch.sigmoid(saidas)
            # 2. Cortamos os extremos por segurança contra erros de arredondamento float32 (evita log(0))
            saidas = torch.clamp(saidas, min=1e-7, max=1.0 - 1e-7)
            # -----------------------------
            
            perda = criterio(saidas, lote_y)
            perda.backward()
            
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            otimizador.step()
            
            perda_acumulada_treino += perda.item() * lote_x.size(0)
            
            preds = (saidas >= 0.5).int().cpu().numpy().flatten()
            predicoes_treino.extend(preds.tolist())
            alvos_treino.extend(lote_y.cpu().numpy().flatten().tolist())
            
        perda_media_treino = perda_acumulada_treino / len(train_loader.dataset)
        acc_treino = accuracy_score(alvos_treino, predicoes_treino)
        f1_treino = f1_score(alvos_treino, predicoes_treino, zero_division=0)
        
        # 2. Fase de Validação
        modelo.eval()
        perda_acumulada_val = 0.0
        predicoes_val = []
        alvos_val = []
        
        with torch.no_grad():
            for lote_x, lote_y in val_loader:
                lote_x, lote_y = lote_x.to(dispositivo), lote_y.to(dispositivo).float().unsqueeze(1)
                saidas = modelo(lote_x)
                
                # --- BLINDAGEM MATEMÁTICA (Validação) ---
                if saidas.min() < 0.0 or saidas.max() > 1.0:
                    saidas = torch.sigmoid(saidas)
                saidas = torch.clamp(saidas, min=1e-7, max=1.0 - 1e-7)
                # ----------------------------------------
                
                perda = criterio(saidas, lote_y)
                
                perda_acumulada_val += perda.item() * lote_x.size(0)
                preds = (saidas >= 0.5).int().cpu().numpy().flatten()
                predicoes_val.extend(preds.tolist())
                alvos_val.extend(lote_y.cpu().numpy().flatten().tolist())
                
        perda_media_val = perda_acumulada_val / len(val_loader.dataset)
        acc_val = accuracy_score(alvos_val, predicoes_val)
        f1_val = f1_score(alvos_val, predicoes_val, zero_division=0)
        
        historico["treino_loss"].append(perda_media_treino)
        historico["treino_acc"].append(acc_treino)
        historico["treino_f1"].append(f1_treino)
        historico["val_loss"].append(perda_media_val)
        historico["val_acc"].append(acc_val)
        historico["val_f1"].append(f1_val)
        
        # 3. Lógica do Early Stopping
        if perda_media_val < melhor_perda_val:
            melhor_perda_val = perda_media_val
            epocas_sem_melhoria = 0
            melhor_estado_modelo = modelo.state_dict().copy()
        else:
            epocas_sem_melhoria += 1
            if epocas_sem_melhoria >= patience:
                break
                
    tempo_fim = time.time()
    tempo_total = tempo_fim - tempo_inicio
    
    if melhor_estado_modelo is not None:
        modelo.load_state_dict(melhor_estado_modelo)
        
    return modelo, historico, tempo_total

def avaliar_modelo(modelo, X_dados, y_dados, dispositivo="cpu"):
    """
    Avalia a rede neural sobre um conjunto de dados e calcula as métricas exigidas:
    Acurácia, Precisão, Recall e F1-score.
    """
    modelo.eval()
    modelo = modelo.to(dispositivo)
    tensor_x = torch.tensor(X_dados, dtype=torch.float32).to(dispositivo)
    
    with torch.no_grad():
        saidas = modelo(tensor_x)
        
        # --- BLINDAGEM MATEMÁTICA ---
        # Só aplica Sigmoid se a saída tiver vazado dos limites de probabilidade
        if saidas.min() < 0.0 or saidas.max() > 1.0:
            saidas = torch.sigmoid(saidas)
        # Corta os extremos para evitar problemas de arredondamento
        saidas = torch.clamp(saidas, min=1e-7, max=1.0 - 1e-7)
        # -----------------------------
        
        predicoes = (saidas >= 0.5).int().cpu().numpy().flatten()
        
    # Garantir que y_dados esteja em formato 1D
    y_true = np.array(y_dados).flatten()
    acc = accuracy_score(y_true, predicoes)
    prec = precision_score(y_true, predicoes, zero_division=0)
    rec = recall_score(y_true, predicoes, zero_division=0)
    f1 = f1_score(y_true, predicoes, zero_division=0)
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "predictions": predicoes
    }

def executar_validacao_cruzada(X_treino_val_cru, y_treino_val, pipeline_preprocessador, 
                               topologia, ativacao, batch_size, lr=0.01, 
                               max_epocas=200, patience=15, semente=42):
    """
    Executa a validação cruzada estratificada em 5 Folds.
    Garante que o ajuste de parâmetros do pré-processamento ocorra estritamente nos folds de
    treino do fold corrente, aplicando nos folds de validação de forma isolada (previne data leakage).
    """
    definir_sementes(semente)
    
    # 5 folds estratificados
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=semente)
    
    metricas_folds = []
    historicos_folds = []
    tempos_folds = []
    
    # Obter identificadores de colunas para remontar o pipeline interno de cada fold
    colunas_numericas = X_treino_val_cru.select_dtypes(include=[np.number]).columns.tolist()
    colunas_categoricas = X_treino_val_cru.select_dtypes(exclude=[np.number]).columns.tolist()
    
    fold_idx = 1
    for indice_treino, indice_val in skf.split(X_treino_val_cru, y_treino_val):
        # 1. Separar dados brutos do fold
        X_treino_bruto = X_treino_val_cru.iloc[indice_treino]
        y_treino = y_treino_val[indice_treino]
        X_val_bruto = X_treino_val_cru.iloc[indice_val]
        y_val = y_treino_val[indice_val]
        
        # 2. Recriar e ajustar pipeline no treino do fold
        pipeline_fold = construir_pipeline(colunas_numericas, colunas_categoricas)
        X_treino_proc = pipeline_fold.fit_transform(X_treino_bruto)
        X_val_proc = pipeline_fold.transform(X_val_bruto)
        
        input_dim = X_treino_proc.shape[1]
        
        # 3. Converter para tensores do PyTorch
        tensor_x_treino = torch.tensor(X_treino_proc, dtype=torch.float32)
        tensor_y_treino = torch.tensor(y_treino, dtype=torch.float32)
        tensor_x_val = torch.tensor(X_val_proc, dtype=torch.float32)
        tensor_y_val = torch.tensor(y_val, dtype=torch.float32)
        
        dataset_treino = TensorDataset(tensor_x_treino, tensor_y_treino)
        dataset_val = TensorDataset(tensor_x_val, tensor_y_val)
        
        # Configurar batch_size dinamicamente se for 'BATCH' (tamanho total do conjunto de treino do fold)
        lote_final = len(dataset_treino) if str(batch_size).upper() == "BATCH" else int(batch_size)
        
        loader_treino = DataLoader(dataset_treino, batch_size=lote_final, shuffle=True)
        # Validação não precisa de shuffle e pode usar lote maior
        loader_val = DataLoader(dataset_val, batch_size=256, shuffle=False)
        
        # 4. Inicializar modelo
        modelo = FraudMLP(input_dim=input_dim, hidden_topology=topologia, 
                          activation_fn=ativacao, dropout_rate=0.2)
        
        # 5. Treinar modelo com Early Stopping
        modelo_treinado, historico, tempo = treinar_modelo(
            modelo, loader_treino, loader_val, lr=lr, 
            max_epocas=max_epocas, patience=patience
        )
        
        # 6. Avaliar o modelo no fold de validação
        resultados_val = avaliar_modelo(modelo_treinado, X_val_proc, y_val)
        
        metricas_folds.append(resultados_val)
        historicos_folds.append(historico)
        tempos_folds.append(tempo)
        
        fold_idx += 1
        
    # Calcular médias das métricas entre os 5 folds
    media_acc = np.mean([m["accuracy"] for m in metricas_folds])
    media_prec = np.mean([m["precision"] for m in metricas_folds])
    media_rec = np.mean([m["recall"] for m in metricas_folds])
    media_f1 = np.mean([m["f1_score"] for m in metricas_folds])
    media_tempo = np.mean(tempos_folds)
    
    return {
        "accuracy": media_acc,
        "precision": media_prec,
        "recall": media_rec,
        "f1_score": media_f1,
        "tempo_processamento": media_tempo,
        "historicos": historicos_folds,
        "metricas_folds": metricas_folds
    }
