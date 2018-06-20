import os
import csv
import numpy as np
from copy import deepcopy

from scipy import misc
import sklearn.utils
from sklearn.model_selection import cross_val_score

##  @package training_alg
#   
#   This module defines functions to support classifier training.
#   Includes data set handling and classifier accuracy evaluation.

##  Auxiliar function. Converts label as characters in filename to int values.
#
#   Notice that this allows a maximum of 26 different classes.
#
def label_map(char):

    mapping = { 'A':0,'a':0,
                'B':1,'b':1,
                'C':2,'c':2,
                'D':3,'d':3,
                'E':4,'e':4,
                'F':5,'f':5,
                'G':6,'g':6,
                'H':7,'h':7,
                'I':8,'i':8,
                'J':9,'j':9,
                'K':10,'k':10,
                'L':11,'l':11,
                'M':12,'m':12,
                'N':13,'n':13,
                'O':14,'o':14,
                'P':15,'p':15,
                'Q':16,'q':16,
                'R':17,'r':17,
                'S':18,'s':18,
                'T':19,'t':19,
                'U':20,'u':20,
                'V':21,'v':21,
                'W':22,'w':22,                
                'X':23,'x':23,
                'Y':24,'y':24,
                'Z':25,'z':25 }

    return mapping[char]
  
##  Read images from specified path. Extracts images descriptors and labels.
#
#   This function takes the path to an image set and a classifier object.
#   It extracts features according to the feature extractor configured
#   in classifier. 
#   
#   The labels are assigned from the first character in the images filename.
#   A image named A1.bmp will be interpreted as a representative of class "0",
#   B156.bmp is labed as being of class "1", C will be labeled as class "2"
#   and so forth.
#
#   Args:
#       path_to_folder: String containing the path of images source folder.
#       classifier_obj: Existing classifier object. See classifier.py. 
#
#   Returns:
#                    X: Numpy bidimensional array containing a vector of 
#                       features for each image.
#                    y: Numpy column array with a int value for the label of
#                       corresponding element in X.
#
def create_data_set(path_to_folder, classifier_obj):
                        
    # Creating lists
    sample_list = []
    label_list  = []
    
    # Feedback for user
    print("Creating data set...")
    print("\tPath to images: " + path_to_folder)
    print("\tReading images...")    

    # Iterates over folder listage. 
    for name in os.listdir(path_to_folder):

        # Load image
        img = misc.imread(os.path.join(path_to_folder,name))
        
        # Color channel reduction        
        img = img[:,:,0]

        # Asks the classifier feature extractors for the image descriptors. 
        sample = classifier_obj.describe_img(img)
        
        # The labeling requires the first character in filename to be a capital
        # letter indicating that sample known class.
        label = name[0]
        label = label_map(label)

        # Converting data types
        sample_list.append(np.array(sample, dtype=float))
        label_list.append(np.array(label, dtype=int))

    # Constructing arrays
    X = np.array(sample_list, dtype=float)
    y = np.array(label_list, dtype=int)
    
    # Ensures correct dimension
    X = np.concatenate(sample_list, axis=0)
  
    print("\t\tRead %d images" % y.size)    
    return (X, y)



##  k-Fold cross-validation to evaluate a classifier performance given the
#   sample and label sets.
#
#   Args:
#                        X: Sample descriptors set.
#                        y: Label set.
#                 n_splits: Number of splits to the data sets.
#           classifier_obj: A classifier object.
#
#   Returns:
#                   scores: Avarage classifier accuracy in percentage.
# 
def test_accuracy(X,y,n_splits,classifier_obj):
    
    print("Evaluating classifier precision:")
    print("\tPerforming k-fold cross validation:")
    print("\tThis may take a while...")
    print("\tk value = " + str(n_splits))
    
    scores = cross_val_score(classifier_obj.model,X,y,cv=n_splits)
    scores = np.mean(scores)*100.0
    print("\tAccuracy score: %.2f%%" % scores)
    return scores
