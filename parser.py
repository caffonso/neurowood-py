import argparse
from gooey import Gooey, GooeyParser


import numpy as np
import sklearn.neighbors as neighbors
import queue, threading
import os, time
from scipy import misc

import classifier_builder
import classifier
import model
import training_alg
import ftr_extraction
import img_processing
import system_sim


##  @package parser
#
#   Visual argument parser to assist in classifier usage.

##  A GooeyParser definition.
#   
#   See more on: https://github.com/chriskiehl/Gooey
@Gooey
def gui_parser():
    parser = GooeyParser(description='Create or train a classifier')
    
    subparser = parser.add_subparsers()

    # tab1
    # Building Classifier suparser
    parser_model = subparser.add_parser(name='New_Classifier')
    parser_model.set_defaults(wich='new_classifier')
    # Model selection
    parser_model.add_argument(type=str, default='knn', choices=['knn','mlp'], help='The model used in classification', dest='model_type')
    # kNN only arguments section
    knn_args = parser_model.add_argument_group('kNN parameters')
    knn_args.add_argument(type=int, default=5, help='number of neighbors in search', dest='n_neighbors')
    knn_args.add_argument(type=str,choices=['uniform','distance'],default='uniform',help='weight function',dest='weights')
    knn_args.add_argument(type=str,choices=['auto','brute','ball_tree','kd_tree'],default='auto',help='Algorithm used in neighbors search',dest='algorithm')
    knn_args.add_argument(type=int,default=2,help='Minkowski distance power parameter. 2 is equivalent to euclidian distance',dest='p')
    knn_args.add_argument(type=int, default=1, help='Number of parallel jobs for neighbors search. -1 will set to the number of CPU cores.', dest='n_jobs')
    # MLP only arguments section
    mlp_args = parser_model.add_argument_group('MLP parameters')
    # hidden_layer_sizes must be parsed from type str to tuple
    mlp_args.add_argument(dest='hidden_layer_sizes', type=str, help='The nth element is the size of the nth hidden layer', default='(100,)')
    mlp_args.add_argument(dest='activation', type=str, help='Hidden layers activation function', default='logistic', choices=['logistic','relu','linear','tanh'])
    mlp_args.add_argument(dest='solver', type=str, help='Method used to optimize weights in learning', default='sgd', choices=['sgd','lbfgs','adam'])
    mlp_args.add_argument(dest='learning_rate', type=str, help='Learning rate behavior for weight update', choices=['constant', 'invscaling', 'adaptive'], default='constant')    
    mlp_args.add_argument(dest='learning_rate_init', type=float, help='Learning rate inital value', default=0.001)
    mlp_args.add_argument(dest='max_iter', type=int, help='Max iteration. When using sgd and adam, determines the number of epochs', default=200)
    mlp_args.add_argument(dest='tol', type=float, help='Tolerance value to determine convergence', default=1e-4)
    # gooey_option adds a validation functionality, but it's only supported using GooeyParser parser.
    mlp_args.add_argument(dest='momentum',type=float,help='Gradient descendent momentum term', default=0.9,
                             gooey_options={
                                'validator': {
                                    'test': '0.0 <= float(user_input) <= 1.0',
                                    'message': 'Must be between 0.0 and 1.0'
                                }
                            })
    # Feature extraction section
    extractor_args = parser_model.add_argument_group('Feature extraction parameters')
    # LBP section
    extractor_args.add_argument(dest='lbp', help='Use Local Binary Pattern', choices=['True','False'],default='True')
    extractor_args.add_argument(dest='numPoints',type=int, help='',default=24)
    extractor_args.add_argument(dest='radius',type=int, help='', default=8)
    extractor_args.add_argument(dest='method',type=str,choices=['uniform'],default='uniform')
    # Haralick section
    extractor_args.add_argument(dest='glcm',help='Use gray level co-occurrence matrix', choices=['True','False'],default='True')
    
    # Output file section
    file_args = parser_model.add_argument_group('Save file options')
    file_args.add_argument('output_filename', type=str,help='Output filename')
    file_args.add_argument('output_folder',type=str,widget='DirChooser',help='Target folder to save classifier')
    
    
    
    
    # tab2
    # Training Classifier subparser
    parser_training = subparser.add_parser(name='Train_Classifier')
    parser_training.set_defaults(wich='train_classifier')
    parser_training.add_argument('classifier_file',type=str,widget='FileChooser',help='Select classifier file to train')
    parser_training.add_argument('images_path',type=str,widget='DirChooser', help='Select training images source')
    parser_training.add_argument('cross_validate',type=str,choices=['yes','no'],help='k-Fold cross validation hit ratio test',default='no')
    parser_training.add_argument('num_folds',type=int,help='Number of splits on data set to validate the model',default=10)
    


    # tab3
    # Classification service subparser
    parser_classification = subparser.add_parser(name='Classification_Service')
    parser_classification.set_defaults(wich='classification_service')
    parser_classification.add_argument('classifier_file',type=str,widget='FileChooser',help='Select previously trained classifier file')
    parser_classification.add_argument('images_path',type=str,widget='DirChooser',help='Image source location')
    
    args = parser.parse_args()
    return args
    
def main():
    gui_parser()
    
if __name__ == '__main__':
    main()
