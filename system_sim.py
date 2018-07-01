import numpy as np
from scipy import misc
import os, time
import threading, queue
import classifier
import cv2

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
    def __init__(self,img_source,output_q, video_capture_device=False, vcd_id=0):
      
                                 
        # Base class initialization
        super(image_producer,self).__init__()

        # Sets thread communication        
        self.filename_list = img_source
        self.output_q = output_q
        self.stop_request = threading.Event()

        # Vide capture device
        self.video_capture_device = video_capture_device
        self.vcd_id = vcd_id
        if self.video_capture_device:
            cv2.namedWindow("VCD", cv2.WINDOW_AUTOSIZE)        
            self.vcd = cv2.VideoCapture()
            retval = self.vcd.open(self.vcd_id)
            print(retval)
                
    

    ##  Start thread
    def run(self):

        # Repeat task until stop_request is set.
        while (not self.stop_request.isSet()):
            
            try:
                if self.video_capture_device:
                    # Checks if video capture device is open,
                    # trying to get first frame
                    if self.vcd.isOpened():
                        print("isOpened")
                        print(self.vcd.get(3))
                        print(self.vcd.get(4))
                        print(self.vcd.get(5))
                        print("here") 
                        rval, frame = self.vcd.read()
                        print("there")
                        print(frame)
                        if frame:
                            
                            cv2.imshow("VCD", frame)
                            key = cv2.waitKey(0)  
                    else:
                            print("not frame")
                            rval = False
                    
                    # If rval was assigned True from call to read(), we
                    # know that camera is up, so we start capturing
                    # images from there    
                    while rval:

                        # Capture image from device                    
                        rval, frame = self.vcd.read()
                        self.output_q.put(frame)            
                        cv2.imshow("VCD", frame)

                        # Exit on esc          
                        key = cv2.waitKey(0)              
                        if key == 27:
                            break

                else:                          
                    # Iterates over filename list load each image
                    for filename in self.filename_list:
                        
                                        
                        time.sleep(0.2)
                        
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
        if self.video_capture_device:
            self.vcd.release()
            cv2.destroyWindow("VCD")
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
