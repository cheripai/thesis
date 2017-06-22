import cv2
import math
import numpy as np
import os
import sys


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def get_thresh(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.bitwise_not(thresh)


def get_contour(thresh):
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


def crop(img, x, y, w, h):
    return img[y:y + h, x:x + w]


def segment(img, n):
    rows, cols = img.shape[:2]
    segment_size = min(rows, cols) // n

    if cols > rows:
        rows = segment_size * n
        cols = segment_size * math.ceil(cols / segment_size)
    else:
        rows = segment_size * math.ceil(cols / segment_size)
        cols = segment_size * n

    img = cv2.resize(img, (cols, rows))

    segments = []
    for i in range(rows // segment_size):
        for j in range(cols // segment_size):
            segments.append(img[i * segment_size:(i + 1) * segment_size, j * segment_size:(j + 1) * segment_size, :])

    return segments


if __name__ == "__main__":
    img_path = sys.argv[1]
    out_path = sys.argv[2]

    img = cv2.imread(img_path)
    rows, cols = img.shape[:2]
    thresh = get_thresh(img)
    cnt = get_contour(thresh)
    rect = cv2.minAreaRect(cnt)
    rotation = rect[-1]
    dst = rotate_bound(thresh, rotation)

    cnt = get_contour(dst)
    x, y, w, h = cv2.boundingRect(cnt)
    rotated = rotate_bound(img, rotation)
    cropped = crop(rotated, x, y, w, h)

    kernel = np.ones((5, 5), np.uint8)
    thresh = get_thresh(cropped)
    thresh = cv2.erode(thresh, kernel, iterations=5)
    thresh = cv2.dilate(thresh, kernel, iterations=20)
    cnt = get_contour(thresh)
    x, y, w, h = cv2.boundingRect(cnt)
    cropped = crop(cropped, x, y, w, h)

    mean_color = int(np.mean(cropped))
    cropped[cropped == 0] = mean_color

    segments = segment(cropped, 2)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for i, img in enumerate(segments):
        cv2.imwrite(os.path.join(out_path, str(i) + ".png"), img)
