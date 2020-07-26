import cv2
import numpy as np
from sklearn import svm
from joblib import dump, load

def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        width = int(image.shape[1]/scale)
        height = int(image.shape[0]/scale)
        dim = (width, height)
        #image = cv2.pyrDown(image)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
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

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                        np.where(overlap > overlapThresh)[0])))
        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick].astype("int")

def mergeBoundingBoxes(detections, xThr, winW, winH):
    rects = [[x, y, winW, winH] for (x, y) in detections] # do nms on the detected bounding boxes
    # Bool array indicating which initial bounding rect has
    # already been used
    rectsUsed = [False for i in range(len(rects))]

    # Array of accepted rects
    acceptedRects = []

    # Iterate all initial bounding rects
    for supIdx, supVal in enumerate(rects):
        if (rectsUsed[supIdx] == False):

            # Initialize current rect
            currxMin = supVal[0]
            currxMax = supVal[0] + supVal[2]
            curryMin = supVal[1]
            curryMax = supVal[1] + supVal[3]

            # This bounding rect is used
            rectsUsed[supIdx] = True

            # Iterate all initial bounding rects
            # starting from the next
            for subIdx, subVal in enumerate(rects[(supIdx+1):], start = (supIdx+1)):

                # Initialize merge candidate
                candxMin = subVal[0]
                candxMax = subVal[0] + subVal[2]
                candyMin = subVal[1]
                candyMax = subVal[1] + subVal[3]

                # Check if x distance between current rect
                # and merge candidate is small enough
                if (abs(currxMin - candxMin)  <= xThr):

                    # Reset coordinates of current rect
                    currxMax = max(currxMax, candxMax)
                    currxMin = min(currxMin, candxMin)
                    curryMin = min(curryMin, candyMin)
                    curryMax = max(curryMax, candyMax)

                    # Merge candidate (bounding rect) is used
                    rectsUsed[subIdx] = True

            # No more merge candidates possible, accept current rect
            acceptedRects.append([currxMin, curryMin, currxMax - currxMin, curryMax - curryMin])
    return acceptedRects

def colorFilter(image):
    # Morph
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # Convertir BGR a HSV
    Ihsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([5,255,255])
    mask0 = cv2.inRange(Ihsv, lower_red, upper_red)

    # Red upper mask (170-180)
    lower_red = np.array([175,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(Ihsv, lower_red, upper_red)

    # Blue mask
    lower_blue = np.array([113,140,0])
    upper_blue = np.array([140,255,180])
    mask2 = cv2.inRange(Ihsv, lower_blue, upper_blue)

    # Joint masks
    canMask = mask0+mask1+mask2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))
    canMask = cv2.morphologyEx(canMask, cv2.MORPH_CLOSE, kernel)
    canMask = cv2.dilate(canMask, kernel, iterations=2)
    canImg = cv2.bitwise_and(image,image,mask = canMask)
    return canImg

def showDetections(detections, image, winW, winH):
    copyImg = np.copy(image)
    rects = [[x, y, winW, winH] for (x, y) in detections] # do nms on the detected bounding boxes

    for (xA, yA, w, h) in rects:
        cv2.rectangle(copyImg, (xA, yA), (xA + w, yA + h), (0,255,0), 2)
    
    cv2.imshow("All detections", copyImg)

def showMergedBoxes(acceptedRects, image):
    mergeImg = np.copy(image)
    for rect in acceptedRects:
        cv2.rectangle(mergeImg, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (121, 11, 189), 2)

    cv2.imshow("Merged boxes", mergeImg)