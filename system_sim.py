import numpy as np
from scipy import misc
import os, time
import threading, queue
import classifier

##  @package system_sim
#
#   Simulates the interaction between image capture and image classification
#   threads.

##  Reads image from selected source and put them in queue for classification.
class image_producer(threading.Thread):

    ##  Class constructor
    #   Args:
    #            img_source: Path to images;  
    #              output_q: Queue to put images as numpy array;
    def __init__(self,img_source,output_q):
      
                                 
        # Base class initialization
        super(image_producer,self).__init__()

        # Sets thread communication
        self.filename_list = img_source
        self.output_q = output_q
        self.stop_request = threading.Event()

    ##  Start thread
    def run(self):

        # Repeat task until stop_request is set.
        while (not self.stop_request.isSet()):
            
            try:
                # TODO:
                # Initialize device                
                # Capture image from device
                
                # Iterates over filename list load each image
                for filename in self.filename_list:
                
                    time.sleep(0.5)
                    
                    # Load image        
                    img = misc.imread(filename)
                    # Color channel reductioin                    
                    img = img[:,:,0]
                    print(filename)
                    # Puts image in output
                    self.output_q.put(img) 
            
            except queue.Empty:
                continue

    ## Join method to end thread activities
    def join(self, timeout=None):
        self.stop_request.set()
        super(image_producer,self).join(timeout)
    
##  Reads images from input image queue and puts resulting classification
#   in the output label queue
class image_consumer(threading.Thread):    
    
    ##  Class constructor
    #
    #   Args:
    #       classifier_obj: Classifier object;
    #              input_q: Image queue;
    #             output_q: Label output;
    def __init__(self,classifier_obj,input_q,output_q):


        # Base class initialization                                          
        super(image_consumer,self).__init__()

        
        self.classifier_obj = classifier_obj

        # Thread communication
        self.input_q = input_q
        self.output_q = output_q
        self.stop_request = threading.Event()
    
    ## Start thread  
    def run(self):

        while (not self.stop_request.isSet()):
            try:   
                image = self.input_q.get(True,0.05)
                
                label = self.classifier_obj.classify_img(image)
                print(label)
                
                self.output_q.put(label)
                                
            except queue.Empty:
                continue    
    
    ## Join method to end thread activities
    def join(self, timeout=None):
    
        self.stop_request.set()
        super(image_consumer,self).join(timeout)
