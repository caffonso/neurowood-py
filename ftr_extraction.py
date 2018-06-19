import numpy as np
from skimage import feature

class GLCM:
        def __init__(self, axis=[0, np.pi/4]):
            #
            self.axis = axis

        def features(self, image, measure):
            #Calculate the grey-level co-occurrence matrix.
            #result = greycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
            glcm = feature.greycomatrix(image, [1], self.axis, normed=True, symmetric=True)
            glcmFeat = feature.greycoprops(glcm, measure)

            return glcmFeat

        def allFeatures(self, image):
            #Calculate the grey-level co-occurrence matrix.
            #result = greycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
            #glcmFeat = []
            #print(type(image))
            #print(image.shape)
            glcm = feature.greycomatrix(image, [1], self.axis, normed=True, symmetric=True)
            #print(type(feature.greycoprops(glcm, 'contrast')))
            glcmFeat = feature.greycoprops(glcm, 'contrast').reshape(-1,)
            #print(type(glcmFeat))
            #print(glcmFeat)
            glcmFeat = np.concatenate((glcmFeat, feature.greycoprops(glcm, 'dissimilarity').reshape(-1,)))
            glcmFeat = np.concatenate((glcmFeat, feature.greycoprops(glcm, 'homogeneity').reshape(-1,)))
            glcmFeat = np.concatenate((glcmFeat, feature.greycoprops(glcm, 'ASM').reshape(-1,)))
            glcmFeat = np.concatenate((glcmFeat, feature.greycoprops(glcm, 'energy').reshape(-1,)))
            glcmFeat = np.concatenate((glcmFeat, feature.greycoprops(glcm, 'correlation').reshape(-1,)))

            return glcmFeat

        def calculate(self,image):
            return self.allFeatures(image)

class LocalBinaryPatterns:
    def __init__(self, numPoints=24, radius=8, method="uniform"):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
        self.method = method

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method=self.method)

        #n_bins = int(lbp.max() + 1)
        #hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
        (hist, _) = np.histogram(lbp.ravel(),  bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))
        #(hist, _) = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist

    def calculate(self,image):
        return self.describe(image)
