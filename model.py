import numpy as np
from scipy import stats

class MLPModel(object):
    
    def __init__(self, hidden_layer_sizes=(5,2)):
        self.model = neural_network.MLPClassifier(hidden_layer_sizes,activation,solver,learning_rate,learning_rate_init,max_iter,tol,momentum)
    
    def train(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        self.model.fit(self.X, self.y)
        
    def predict(self,sample):
        return self.model.predict(sample)
    

class KNNModel(object):
    
    def __init__(self, k, weights='distance'):
        self.model = neighbors.KNeighborsClassifier(n_neighbors, weights, algorithm)

    def train(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        self.model.fit(self.X, self.y)

        
    def predict(self,sample):
        return self.model.predict(sample)    
        
