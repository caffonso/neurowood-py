##  @package model
#
#   Defines interface to statistical model framework classes and methods

##  Multilayer Perceptron Classifier   
class MLPModel(object):

    ##  Class constructor
    #
    #   For further model and parameters understanding, please check documentation:
    #   http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    #
    #   Args:
    #                  hidden_layer_sizes: Tuple of ints. The n-th element is
    #                                      the size of the n-th hidden layer;                                       
    #                          activation: Activation function;
    #                              solver: Solver for weights optimization;
    #                       learning_rate: Weights optimization learning rate;
    #                  learning_rate_init: Initial value for learning rate;
    #                            max_iter: Maximum allowed iterations;
    #                                 tol: Tolerance value to set convergence;
    #                            momentum: Gradient descendent method parameter;
    #
    def __init__(self,  hidden_layer_sizes,
                        activation,solver,
                        learning_rate,
                        learning_rate_init,
                        max_iter,
                        tol,
                        momentum):

        self.model = neural_network.MLPClassifier  (hidden_layer_sizes,
                                                    activation,solver,
                                                    learning_rate,
                                                    learning_rate_init,
                                                    max_iter,
                                                    tol,
                                                    momentum)
    
    ##  Train the model using provided data set
    #   
    #   Args:
    #              X_train: Sample descriptors set.
    #              y_train: Label set                            
    def train(self, X_train, y_train):
        self.model.fit(self.X, self.y)
    
    ##  Asks the model for a sample or a set of samples label prediction
    def predict(self,sample):
        return self.model.predict(sample)
    
##  k-Nearest Neighbors Classifier 
class KNNModel(object):
    
    ##  Class constructor
    #
    #   For further model and parameters understanding, please check documentation:
    #   http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html  
    #
    #   Args:
    #              n_neighbors: Number of neighbors to use in search;
    #                  weights: Weight function to use in search;
    #                algorithm: Algorithm to compute nearest neighbors;
    #                                
    def __init__(self, n_neighbors, weights, algorithm):
        self.model = neighbors.KNeighborsClassifier(n_neighbors, weights, algorithm)

    ##  Train the model using provided data set
    #   
    #   Args:
    #              X_train: Sample descriptors set.
    #              y_train: Label set
    def train(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        self.model.fit(self.X, self.y)

    ##  Asks the model for a sample or a set of samples label prediction        
    def predict(self,sample):
        return self.model.predict(sample)    