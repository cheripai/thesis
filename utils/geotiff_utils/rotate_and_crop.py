import cv2
import numpy as np
import sys
from operator import mul

LETTUCE_ROTATION = 39.7


def maximalRectangleInHistogram(histogram):
    """
    Calculate the maximum rectangle area in terms of a given histogram.
    :param histogram: List[int]
    """
    # Stack for storing the index.
    posStack = []
    i = 0
    maxArea = 0
    maxHeight = 0
    maxWidth = 0
    colMax = 0
    while i < len(histogram):
        if len(posStack) == 0 or histogram[i] > histogram[posStack[-1]]:
            # Advance the index when either the stack is empty or the
            # current height is greater than the top one of the stack.
            posStack.append(i)
            i += 1
        else:
            curr = posStack.pop()
            width = i if len(posStack) == 0 else i - posStack[-1] - 1
            area = width * histogram[curr]
            if area > maxArea:
                maxArea = area
                maxHeight = histogram[curr]
                maxWidth = width
                colMax = curr
    # Clean the stack.
    while posStack:
        curr = posStack.pop()
        width = i if len(posStack) == 0 else len(histogram) - posStack[-1] - 1
        area = width * histogram[curr]
        if area > maxArea:
            maxArea = area
            maxHeight = histogram[curr]
            maxWidth = width
            colMax = curr
    return maxArea, maxHeight, maxWidth, colMax


def maximalRectangle(matrix):
    """
        :type matrix: List[List[int]]
    """
    maxArea = 0
    maxHeight = 0
    maxWidth = 0
    maxRow = len(matrix)
    maxCol = len(matrix[0])
    rowMax, colMax = 0, 0
    # For every row in the given 2D matrix, it is a "Largest Rectangle in
    # Histogram" problem, which is the subproblem.
    lookupTable = [0 for _ in range(maxCol)]
    for row in range(maxRow):
        for col in range(maxCol):
            # If it is "1"
            if matrix[row][col] > 0:
                # Accumulate the column if if's 1's.
                lookupTable[col] += matrix[row][col]
            else:
                # Clean the column if it's 0's.
                lookupTable[col] = 0
        # Calculate the maximum area.
        area, height, width, c = maximalRectangleInHistogram(lookupTable)
        if area > maxArea:
            maxArea = area
            maxHeight = height
            maxWidth = width
            rowMax, colMax = row, c
    return maxHeight, maxWidth, rowMax, colMax


if __name__ == "__main__":
    img = cv2.imread(sys.argv[1], -1)
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), LETTUCE_ROTATION, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))
    # binarized = np.where(rotated[:, :, 3] < 255, 0, 1).tolist()
    # height, width, row, col = maximalRectangle(binarized)
    # rotated = rotated[row-height:row, col:col+width]
    ret, thresh = cv2.threshold(rotated[:,:,3], 244, 255, 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[-1])
    rotated = rotated[y:y+h, x:x+w]
    rotated[np.where(rotated[:,:,3] < 255)] = 0
    cv2.imwrite("rotated.png", rotated[:,:,:3])
