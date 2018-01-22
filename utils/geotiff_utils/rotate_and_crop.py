import cv2
import numpy as np
import sys

LETTUCE_ROTATION = 39.7
# LETTUCE_ROTATION = 40.6

def rotate(img, rotation):
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), rotation, 1)
    return cv2.warpAffine(img, M, (cols, rows))
    
def find_crop(img):
    ret, thresh = cv2.threshold(img, 244, 255, 0)
    dilation = cv2.dilate(thresh, np.ones((5, 5)), iterations=1)
    _, contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[-1])
    return x, y, w, h

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1], -1)
    rotated = rotate(img, LETTUCE_ROTATION)
    x, y, w, h = find_crop(rotated[:,:,3])
    cropped = rotated[y:y+h, x:x+w]
    cropped[np.where(cropped[:,:,3] < 255)] = 0
    cv2.imwrite(sys.argv[2], cropped[:,:,:3])
