import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from skimage import exposure
from sklearn import svm
from joblib import dump, load

from functions import colorFilter
from functions import pyramid
from functions import sliding_window
from functions import showDetections
from functions import mergeBoundingBoxes
from functions import showMergedBoxes

clf = load('cans.joblib')
# read the image you want to detect the object in:
image = cv2.imread("test/gazebo1.png")
originalSize = image.shape

canImg = colorFilter(image)

detections = []
confidences = []
# Parameters fo pyramid and sliding window
winW = 104
winH = 181
downscale=1.5
windowSize=(winW,winH)
scale = 1
stepSize = int(70) # Higher number makes algorithm faster, but might not detect everything
# Parameters for HOG detection
orientations = 8
pixels_per_cell = (26, 26)
cells_per_block = (1,1)

# Change to true for debugging
show = False

for resized in pyramid(canImg, scale=downscale):
    newSize = resized.shape
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize=stepSize, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        # There is no need to slide the window through most of the
        # bottom part of the image, since the cans are always high up,
        # on the tables
        if y > 0.55*resized.shape[0]:
            continue
        if cv2.countNonZero(window[:,:,0]) != 0:
            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE # WINDOW
            window = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
            if show:
                clone = resized.copy()
                cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
                cv2.imshow("Window", clone)
                cv2.waitKey(1)
                time.sleep(0.0001)
            descr_hog, Ihog = hog(window, orientations=orientations, pixels_per_cell=pixels_per_cell,
                        cells_per_block=cells_per_block, visualize=True, multichannel=True, block_norm='L2-Hys')
            pred = clf.predict([descr_hog])
            if pred == 1:
                confidence = clf.decision_function([descr_hog]) 
                if show:
                    print(confidence)
                if confidence >= 0.3:
                    xOriginal = int(x*originalSize[1]/newSize[1])
                    yOriginal = int(y*originalSize[0]/newSize[0])
                    detections.append((xOriginal, yOriginal))
                    confidences.append(np.asscalar(confidence))
                    if show:
                        print("Scaled Coordinates: ({}, {})".format(x, y))
                        print("Original Coordinates: ({}, {})".format(xOriginal, yOriginal))
                        print("Scale ->  {} | Confidence Score {} \n".format(scale,confidence))
    scale = scale*downscale

cv2.destroyAllWindows()

#showDetections(detections, image, winW, winH)

acceptedRects = mergeBoundingBoxes(detections, 50, winW, winH)

showMergedBoxes(acceptedRects, image)
