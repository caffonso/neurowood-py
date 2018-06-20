##  @package img_processing
#
#   Image processing operations

##  Does nothing
class DummyFilter(object):

    ## Class constructor    
    def __init__(self):
        pass

    ## Apply class operation
    def apply(self,img):
        return img
