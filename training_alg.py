import os
import csv
import numpy as np
from scipy import misc
import sklearn.utils
from sklearn.model_selection import cross_val_score
from copy import deepcopy

# Função para converter as imagens contidas em "path_to_folder" em um conjunto
# de dados, para ser usado no treinamento
# "classifier_obj" mantém sua lista de objetos de extração de características e
# processamento de imagens intrinsecos
def create_data_set(path_to_folder, classifier_obj):
    
    # Inicializa as listas vazias
    sample_list = []
    label_list  = []
    
    print("Creating data set...")
    print("Path to images: " + path_to_folder)
    
    # Itera pela listagem de nomes do diretório
    # Supõe que existem apenas as imagens etiquetadas
    for name in os.listdir(path_to_folder):

        # Carrega a imagem
        img = misc.imread(os.path.join(path_to_folder,name))
        img = img[:,:,0]
        # Extrai as características e lê a categoria conhecida
        # no primeiro caractere do nome do arquivo
        sample = classifier_obj.describe_img(img)
        
        # ALTERAR!!!!!!!!!!!!!!!
        if (name[0] == 'A'):
            label = 0
        if (name[0] == 'B'):
            label = 1
        if (name[0] == 'C'):
            label = 2
        # Converte explicitamente cada amostra e etiqueta, inclui nas listas
        sample_list.append(np.array(sample, dtype=float))
        label_list.append(np.array(label, dtype=int))


    # Converte explicitamente as listas com todas as amostras e etiquetas
    X = np.array(sample_list, dtype=float)
    X = np.concatenate(sample_list, axis=0)
    y = np.array(label_list, dtype=int)
    
    print("Read %d images" % X.ndim)
    
    return (X, y)

# funcao para leitura de conjunto de dados
# recebe o caminho para um arquivo
# retorna uma tupla de objetos do tipo numpy.mat, (X, y):
#    X: matriz de dados, com num_amostras linhas e num_caracteristicas colunas
#    y: matriz de categorias, com num_amostras linhas e uma coluna
def load_data_set(filepath):
    
    sample_list = []   # linhas do arquivo armazenadas como listas de np.array
    label_list = [] 
    
    # abre o arquivo, interpreta linha por linha
    with open(filepath, newline='') as csv_file:
    
        print("Leu o arquivo",filepath)
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line in csv_reader:
           
            # divide a linha em data e label
            sample = line[:len(line)-1]
            label = line[-1]
            
            # adiciona na lista correspondente
            sample_list.append(np.array(sample, dtype=float))
            label_list.append(np.array(label, dtype=int))

    # converte as listas para np.ndarray
    X = np.array(sample_list, dtype=float)
    y = np.array(label_list, dtype=int)
    return (X, y)               


# Funcao para treinar e testar um classificador pelo método de validação kfold
# Recebe um conjunto de dados X, um conjunto de resposta y, um objeto classificador
def test_accuracy(X,y,n_splits,classifier_obj):
    
    print("Evaluating classifier precision:")
    print("Performing k-fold cross validation...")
    print("k value = " + str(n_splits))
    
    scores = cross_val_score(classifier_obj.model,X,y,cv=n_splits)
    scores = np.mean(scores)*100.0
    print("Accuracy score: %.2f" % scores)
    return scores
