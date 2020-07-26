# %%
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from skimage import exposure
import time
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
# %%
def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        width = int(image.shape[1]/scale)
        height = int(image.shape[0]/scale)
        dim = (width, height)
        image = cv2.pyrDown(image)
        #image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def plotImg(I, gray=False):
    plt.figure()
    if gray:
        plt.imshow(I, cmap='gray'); plt.axis('off')
    else:
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        plt.imshow(I); plt.axis('off')

# %%
# load the image and define the window width and height
""" image = cv2.imread("test/gazebo1.png")
(winW, winH) = (128, 128*2)

for resized in pyramid(image, scale=1.5):
	# loop over the sliding window for each layer of the pyramid
	for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
		# WINDOW
		# since we do not have a classifier, we'll just draw the window
		clone = resized.copy()
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		cv2.imshow("Window", clone)
		cv2.waitKey(1)
		time.sleep(0.005)

cv2.destroyAllWindows() """

# %%
# Generate positive image listing
# 370, 472
""" im_path = r"cokeTrain/" # This is the path of our positive input dataset
im_listing = os.listdir(im_path)
imgNum = 1
for file in im_listing:
    path = im_path + file
    I = cv2.imread(path)
    #I = I[2:470, 190:430]
    I = I[360:576, 1044:1180]
    savePath = "pos3/img" + str(imgNum) + "_3.png" 
    cv2.imwrite(savePath, I)
    imgNum = imgNum + 1 """

# Generate negative image listing
""" I = cv2.imread("test/rviz4.png")
winW = 196
winH = 458
imgNum = 1
for (x, y, window) in sliding_window(I, stepSize=128, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
		# WINDOW
		# since we do not have a classifier, we'll just draw the window
        clone = I.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        savePath = "neg/img" + str(imgNum) + "_8.png"
        cv2.imwrite(savePath, clone[y:y+winH, x:x+winW])
        imgNum = imgNum  + 1
        cv2.waitKey(1)
        time.sleep(0.005)

cv2.destroyAllWindows() """
# %%
# define parameters of HOG feature extraction
orientations = 8
pixels_per_cell = (32, 32)
cells_per_block = (1,1)

# This is the path of our positive input dataset
pos_im_path = r"pos/" 
# define the same for negatives
neg_im_path= r"neg/"

# read the image files:
pos_im_listing = os.listdir(pos_im_path) # it will read all the files in the positive image path (so all the required images)
neg_im_listing = os.listdir(neg_im_path)
num_pos_samples = len(pos_im_listing) # simply states the total no. of images
num_neg_samples = len(neg_im_listing)
# prints the number value of the no.of samples in positive dataset
print(num_pos_samples)
print(num_neg_samples)

data= []
labels = []

windowDim = (104, 181)

print("Positive HOG IMAGES")
print("-------------------------")
for file in pos_im_listing:
    path = pos_im_path + file
    I = cv2.imread(path)
    # convert the image to RGB
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    I = cv2.resize(I, windowDim, interpolation = cv2.INTER_AREA)
    # calculate HOG for positive features
    descr_hog, Ihog = hog(I, orientations=orientations, pixels_per_cell=pixels_per_cell,
                      cells_per_block=cells_per_block, visualize=True, multichannel=True, block_norm='L2-Hys')
    data.append(descr_hog)
    #plotImg(Ihog, gray=True)
    labels.append(1)

#plt.show()
print("Negative HOG IMAGES")
print("-------------------------")

for file in neg_im_listing:
    path = neg_im_path + file
    I = cv2.imread(path)
    # convert the image to RGB
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    I = cv2.resize(I, windowDim, interpolation = cv2.INTER_AREA)
    # calculate HOG for positive features
    descr_hog, Ihog = hog(I, orientations=orientations, pixels_per_cell=pixels_per_cell,
                      cells_per_block=cells_per_block, visualize=True, multichannel=True, block_norm='L2-Hys')
    data.append(descr_hog)
    #plotImg(Ihog, gray=True)
    labels.append(0)

#plt.show()

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainData, testData, trainLabels, testLabels) = train_test_split(
	np.array(data), labels, test_size=0.20, random_state=42)

# Train
clf = svm.SVC()
clf.fit(trainData, trainLabels)


y_true = testLabels
y_pred = clf.predict(testData)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print("Falsos positivos: ", fp)
print("Flasos negativos: ", fn)
print("Verdaderos positivos: ", tp)
print("Verdaderos negativos: ", tn)
A = (tp + tn) / len(y_true)
print("Exactitud en el conjunto de entrenamiento: ", A)
# %%
# Save trained set
dump(clf, 'cans.joblib')   
# %%
