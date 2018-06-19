import training_alg
import numpy as np

# Classe utilitária principal
# É através desta interface que o usuário realiza a maioria das tarefas
class Classifier(object):

    # Construtor
    # Recebe:
    # model_obj: Um modelo de aprendizado que deve possuir os métodos fit(X,y) e predict(sample)
    # extractor_obj_list: Uma lista de objetos de extração de característica
    def __init__(self, model_obj, extractor_obj_list, img_proc_obj_list):
        self.model = model_obj
        self.extractor_list = extractor_obj_list
        self.img_proc_list = img_proc_obj_list

#
    def classify_img(self,img):

        # Aplica todos os processos à imagem
        sample = img

        for process in self.img_proc_list:
            sample = process.apply(sample)


        # Calcula o vetor descritor
        descriptor = []
        for extractor in self.extractor_list:
            descriptor.append(extractor.calculate(sample))

        # Retorna a predição do modelo
        desc = np.concatenate(descriptor, axis=0)
        desc=desc.reshape(1,-1)
        return self.model.predict(desc)

#
    def predict(self,sample):
        return self.model.predict(sample)

#
    def describe_img(self,img):

        # Aplica todos os processos à imagem
        sample = img
        for process in self.img_proc_list:
            sample = process.apply(sample)

        # Calcula o vetor descritor
        descriptor = []
        for extractor in self.extractor_list:
            descriptor.append(extractor.calculate(sample))
            #descriptor = np.concatenate(descriptor,extractor.calculate(sample))

        # Retorna a predição do modelo
        desc = np.concatenate(descriptor, axis=0)
        desc = desc.reshape(1,-1)

        return desc

#
    def train(self, X_train, y_train):
        print("Training classifier...")
        print("\tNumber of samples: %d" % y_train.size)        
        print("\tThis may take a while...")
        self.X = X_train
        self.y = y_train
        self.model.fit(self.X, self.y)
        print("\tTraining complete!")
        
