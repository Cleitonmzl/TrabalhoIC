# -*- coding: utf-8 -*-
"""
==========================================================================
MÓDULO 1: CARREGAMENTO E AMOSTRAGEM DE DADOS (paysim_sample.csv)
Responsabilidade: Pedro Davi
==========================================================================
Este módulo é responsável por carregar a base de dados original do PaySim,
realizar a amostragem estratificada utilizando a matrícula como semente
para fins de reprodutibilidade e salvar o arquivo 'paysim_sample.csv' contendo:
- 500 instâncias fraudulentas (isFraud == 1)
- 2500 instâncias normais (isFraud == 0)
- Total de 3000 instâncias.

Caso o arquivo original não esteja na pasta de trabalho, o script tenta baixá-lo
automaticamente usando a biblioteca kagglehub e o copia para a pasta atual.
"""

import os
import shutil
import pandas as pd

# matricula real
MATRICULA = 20250026503

def baixar_e_preparar_dados():
    """
    Tenta localizar o arquivo 'paysim.csv' localmente.
    Se não encontrar, baixa usando kagglehub e copia para a pasta do projeto.
    Retorna o dataframe correspondente à amostra de 3000 linhas gerada.
    """
    arquivo_original = "paysim.csv"
    arquivo_amostra = "paysim_sample.csv"
    
    print("-" * 70)
    print("MÓDULO 1 (Cleiton Mizael) - Carregamento dos dados iniciado...")
    print("-" * 70)

    # 1. Verificar se a amostra já existe para evitar reprocessamento desnecessário
    if os.path.exists(arquivo_amostra):
        print(f"[INFO] O arquivo de amostra '{arquivo_amostra}' já existe.")
        print("[INFO] Carregando a amostra existente para visualização...")
        amostra_df = pd.read_csv(arquivo_amostra)
        print(f"[SUCESSO] Amostra carregada com sucesso! Dimensões: {amostra_df.shape}")
        exibir_estatisticas_basicas(amostra_df)
        return amostra_df

    # 2. Se o paysim.csv original não estiver no diretório, tenta baixar via kagglehub
    if not os.path.exists(arquivo_original):
        print(f"[AVISO] O arquivo '{arquivo_original}' não foi encontrado no diretório local.")
        print("[INFO] Tentando baixar a base de dados via 'kagglehub' (ealaxi/paysim1)...")
        try:
            import kagglehub
            path_baixado = kagglehub.dataset_download("ealaxi/paysim1")
            print(f"[INFO] Download concluído. Arquivos baixados em: {path_baixado}")
            
            # Procurar pelo arquivo CSV dentro do caminho retornado pelo kagglehub
            csv_encontrado = None
            for root, dirs, files in os.walk(path_baixado):
                for file in files:
                    if file.endswith(".csv"):
                        csv_encontrado = os.path.join(root, file)
                        break
            
            if csv_encontrado:
                print(f"[INFO] Copiando o arquivo '{csv_encontrado}' para a pasta de trabalho como '{arquivo_original}'...")
                # Copiar para a pasta local com o nome 'paysim.csv'
                shutil.copy(csv_encontrado, arquivo_original)
                print("[SUCESSO] Base original copiada com sucesso!")
            else:
                raise FileNotFoundError("Nenhum arquivo CSV encontrado nos arquivos baixados do Kaggle.")
        except Exception as e:
            print(f"[ERRO CRÍTICO] Falha ao baixar ou extrair os dados automaticamente: {e}")
            print("[DICA] Certifique-se de instalar 'kagglehub' (pip install kagglehub) ou coloque")
            print(f"       manualmente o arquivo '{arquivo_original}' na mesma pasta deste script.")
            return None

    # 3. Gerar a amostra conforme as regras da atividade
    print(f"[INFO] Carregando a base original '{arquivo_original}'...")
    try:
        # Carregar em pedaços ou direto dependendo da memória. paysim.csv tem cerca de 6.3 milhões de linhas.
        # Vamos ler o arquivo completo com tratamento de tipos otimizado para economizar RAM.
        df = pd.read_csv(arquivo_original)
        print(f"[INFO] Base de dados carregada! Dimensões totais: {df.shape}")
        
        # Filtrar as fraudes e normais
        print("[INFO] Filtrando e coletando amostras...")
        fraudes = df[df["isFraud"] == 1]
        normais = df[df["isFraud"] == 0]
        
        # Validar se temos dados suficientes
        if len(fraudes) < 500:
            raise ValueError(f"A base contém apenas {len(fraudes)} fraudes, mas precisamos de 500.")
        if len(normais) < 2500:
            raise ValueError(f"A base contém apenas {len(normais)} transações normais, mas precisamos de 2500.")
        
        # Amostragem usando a matrícula como semente aleatória (reprodutível)
        amostra_fraudes = fraudes.sample(n=500, random_state=MATRICULA)
        amostra_normais = normais.sample(n=2500, random_state=MATRICULA)
        
        # Concatenar e embaralhar tudo aleatoriamente usando a matrícula
        amostra = pd.concat([amostra_fraudes, amostra_normais])
        amostra = amostra.sample(frac=1, random_state=MATRICULA).reset_index(drop=True)
        
        # Salvar a amostra gerada em paysim_sample.csv
        amostra.to_csv(arquivo_amostra, index=False)
        print(f"[SUCESSO] Amostra gerada e salva com sucesso em '{arquivo_amostra}'!")
        exibir_estatisticas_basicas(amostra)
        return amostra
        
    except Exception as e:
        print(f"[ERRO CRÍTICO] Erro ao ler a base de dados original ou realizar a amostragem: {e}")
        return None

def exibir_estatisticas_basicas(df):
    """
    Exibe informações básicas do dataset gerado para controle e estudo da dupla.
    """
    print("-" * 50)
    print("ESTATÍSTICAS DA AMOSTRA GERADA:")
    print(f"Total de linhas: {len(df)}")
    print(f"Número de Fraudes (isFraud == 1): {len(df[df['isFraud'] == 1])}")
    print(f"Número de Normais (isFraud == 0): {len(df[df['isFraud'] == 0])}")
    print("Proporção de Fraudes na amostra: {:.2f}%".format((len(df[df['isFraud'] == 1]) / len(df)) * 100))
    print("Atributos presentes:", list(df.columns))
    print("-" * 50)

if __name__ == "__main__":
    # Teste de execução isolada do módulo de carregamento
    baixar_e_preparar_dados()
