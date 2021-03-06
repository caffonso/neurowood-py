import numpy as np
import sklearn.neighbors as neighbors
import queue
import os, time
from scipy import misc

import classifier_builder as bldr
import classifier
import model
import training_alg
import ftr_extraction
import img_processing
import system_sim
import parser


def main():

    # Collects user options
    args = parser.gui_parser()    
    print("selected action:")
    print(args.wich)
    
   
    if (args.wich == 'new_classifier'):
        
        # Constroi o classificador
        builder = bldr.ClassifierBuilder()
        parameters = builder.parse_arguments(args)
        classifier = builder.build_classifier(parameters)
        
        # Salva no caminho especificado
        filepath = os.path.join(args.output_folder,args.output_filename)
        builder.save_classifier(filepath, classifier)
        print("Saved file %s" % filepath)
        
    elif (args.wich == 'train_classifier'):
        
        # Carrega um classificador previamente salvo
        builder = bldr.ClassifierBuilder()
        classifier = builder.load_classifier(args.classifier_file)
        
        # Constroi o conjunto de dados
        X,y = training_alg.create_data_set(args.images_path, classifier)

        # Se o usuário escolheu cross validation, executa o teste
        if (args.cross_validate == 'yes'):
            training_alg.test_accuracy(X,y,args.num_folds,classifier)
        
        # Treina o modelo
        classifier.train(X,y)
            
        builder.save_classifier(args.classifier_file+'.trained',classifier)
        

    elif (args.wich == 'classification_service'):
        
        # Carrega o classificador
        builder = bldr.ClassifierBuilder()
        classifier = builder.load_classifier(args.classifier_file)
        
        # Acessa a origem das imagens     
        filename_list = os.listdir(args.images_path)
        full_path_filename_list = [os.path.join(args.images_path,filename) for filename in filename_list]
        
        if args.video_capture_device == 'True':
            video_capture_device = True
            vcd_id = args.vcd_id
        else:
            video_capture_device = False        


        # Comunicação entre as threads
        image_q = queue.Queue()
        label_q = queue.Queue()
        
        # Cria thread pool
        thread_pool = []
        thread_pool.append(system_sim.image_producer(full_path_filename_list, image_q, video_capture_device))
        thread_pool.append(system_sim.image_consumer(classifier,image_q, label_q))
        
        # Inicia threads
        for t in thread_pool:
            t.start()
            
        # Aguarda o fim da execução
        for t in thread_pool:
            t.join()
                
        
    
main()
