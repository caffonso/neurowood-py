import classifier
import model
import training_alg
import ftr_extraction
import img_processing
import pickle
import sklearn.neighbors as neighbors
import sklearn.neural_network as neural_network

from ast import literal_eval as make_tuple

class ClassifierBuilder(object):
    
    #
    def __init__(self):
        pass
    

    #
    def load_classifier(self,filepath):
        file = open(filepath,'rb')
        unpickled = pickle.load(file)
        file.close()
        return unpickled

    #
    def save_classifier(self,filepath, classifier_obj):
        file = open(filepath,'wb')
        pickle.dump(classifier_obj, file)
        file.close()
    #
    def parse_arguments(self,args):

        # Cria uma lista vazia para retornar
        param_dict_list = []

        # Process model arguments
        model_arguments = {}
        model_arguments['model'] = args.model_type
    
        if (args.model_type == 'knn'): 
            model_arguments['n_neighbors'] = args.n_neighbors
            model_arguments['weights'] = args.weights
            model_arguments['algorithm'] = args.algorithm
        
        elif (args.model_type == 'mlp'):
            model_arguments['hidden_layer_sizes'] = make_tuple(args.hidden_layer_sizes)
            model_arguments['activation'] = args.activation
            model_arguments['solver'] = args.solver
            model_arguments['learning_rate'] = args.learning_rate
            model_arguments['learning_rate_init'] = args.learning_rate_init
            model_arguments['max_iter'] = args.max_iter
            model_arguments['tol'] = args.tol
            model_arguments['momentum'] = args.momentum
                
        param_dict_list.append(model_arguments)
        
        if (args.glcm == 'True'):
            glcm_arguments = {}
            glcm_arguments['extractor'] = 'glcm'
            param_dict_list.append(glcm_arguments)
        
        if (args.lbp == 'True'):
            lbp_arguments = {}
            lbp_arguments['extractor'] = 'lbp'
            lbp_arguments['numPoints'] = args.numPoints
            lbp_arguments['radius'] = args.radius
            lbp_arguments['method'] = args.method
            param_dict_list.append(lbp_arguments)
            
        # Retorna a lista de dicionarios
        return param_dict_list

    # Recebe um dicionario contendo os parâmetros para criação do modelo.
    # Os modelos suportados e seus parâmetros configuráveis são:
    #
    #   - param['model']='knn'
    #       - param['k']=int
    #       - param['weights']='distance'
    #
    #   - param['model']='mlp'    
    #       - param['hidden_layer_size']=(dim_0,dim_1)
    #       
    # Retorna um modelo pronto para ser treinado.
    def build_model_obj(self,param):
        # k-Nearest Neighbors
        if (param[0]['model'] == 'knn'):
            param[0].pop('model')
            return neighbors.KNeighborsClassifier(**param[0])
        
        # Multi Layer Perceptron
        elif (param[0]['model'] == 'mlp'):
            param[0].pop('model')
            return neural_network.MLPClassifier(**param[0])
        
        # Necessário tratar erros
        else:
            print("Invalid model type")

    # Recebe uma lista de dicionarios, cada um contendo a informação
    # para a construção de um extrator de características. Os possíveis
    # extratores e seus parametros são:
    #
    #   - param['extractor']='glcm'
    #       - param['axis']=[num_points,radius]
    #
    #   - param['extractor']='lbp'    
    #       - param['numPoints']=int
    #       - param['radius']=
    #       - param['method']=
    #       
    # Retorna uma lista de objetos extratores de características.
    def build_feature_extractor_obj(self,param_list):

        ftr_extractor_obj_list = []
        # Itera pela lista de dicionarios
        for param in param_list:
            # Gray Level Coocurence Matrix (GLCM)
            if (param['extractor'] == 'glcm'):
                param.pop('extractor')
                ftr_extractor_obj_list.append(ftr_extraction.GLCM(**param))
            
            # Local Binary Patterns (LBP)
            elif (param['extractor'] == "lbp"):
                param.pop('extractor')
                ftr_extractor_obj_list.append(ftr_extraction.LocalBinaryPatterns(**param))
            
            # Precisa de tratamento de erros
            else:
                print("Invalid extractor type")
        return ftr_extractor_obj_list


    def build_img_processing_obj(self, param_list):
        img_proc_obj_list = []

        for param in param_list:
            if param['type'] == 'pass':
                img_proc_obj_list.append(img_processing.DummyFilter())
        
        return img_proc_obj_list
            

    # Recebe uma lista de dicionários com parâmetros para instanciar cada objeto que irá compor o
    # classificador.
    # Retorna um objeto da classe classifier.Classifier(), pronto para uso
    def build_classifier(self,param):
        
        # Separa os parâmetros de criação para cada tipo de objeto que compõe o classificador
        model_param = [i for i in param if 'model' in i]
        extractor_param_list = [i for i in param if 'extractor' in i]
        img_processing_param_list = [i for i in param if 'img_processing' in i]

        # Constroi os objetos que compõem o classificador
        model_obj = self.build_model_obj(model_param)
        extractor_obj_list = self.build_feature_extractor_obj(extractor_param_list)
        img_proc_obj_list = self.build_img_processing_obj(img_processing_param_list)

        # Monta o classificador
        return classifier.Classifier(model_obj, extractor_obj_list, img_proc_obj_list)
