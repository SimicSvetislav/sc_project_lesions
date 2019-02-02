import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

import preprocessing as pp

def segment(img):
    
    out = pp.preprocess(img)
    
    img_bin = cv2.adaptiveThreshold(out, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 10)

    ret, img_bin = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    img_ret, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    largestContour = max(contours, key = cv2.contourArea)
    
    return largestContour

def segmentString(image):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return segment(img)

