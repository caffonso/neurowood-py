import training_alg
import numpy as np

##  @package classifier
#
#   Defines the general Classifier interface

##  The Classifier class is the main object in the classification process.
#   
#   This class purpose is to set classifier behavior, so that different
#   classifier configurations choosen by the user can be used the same way.
#
#   Notice that Classifier specializations are handled by the 
#   classifier_builder module.
#
class Classifier(object):

    ##  Class constructor
    #
    #   Args:
    #              model_obj: A model object. See model;
    #     extractor_obj_list: A list of feature extraction objects.
    #                         See ftr_extraction;
    #      img_proc_obj_list: A list of image processing operation objects.
    #                         See img_processing;
    #
    def __init__(self, model_obj, extractor_obj_list, img_proc_obj_list):
        self.model = model_obj
        self.extractor_list = extractor_obj_list
        self.img_proc_list = img_proc_obj_list

    ##  Takes a image and returns it's predicted label.
    #
    #   Args:
    #                    img: Image loaded as numpy array; 
    #   Returns:
    #                  label: Predicted label;
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

    ##  Takes a image descriptor and returns it's predicted label.
    #
    #   Args:
    #                 sample: Image features;
    #   Returns:
    #                  label: Predicted label;
    #
    def predict(self,sample):
        return self.model.predict(sample)

    ##  Asks classifier to describe a image, but not to predict it's label
    #
    #   Args:
    #                    img: Image loaded as numpy array;
    #   Returns:
    #             descriptor: Image features;
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

    ##  Train the classifier model using provided data.
    #
    #   Args:
    #            X_train: Numpy bidimensional array containing a vector of 
    #                     features for each image.
    #            y_train: Numpy column array with a int value for the label of
    #                     corresponding element in X.
    def train(self, X_train, y_train):

        # Feedback
        print("Training classifier...")
        print("\tNumber of samples: %d" % y_train.size)        
        print("\tThis may take a while...")
        
        self.model.fit(X_train, y_train)

        print("\tTraining complete!")        
