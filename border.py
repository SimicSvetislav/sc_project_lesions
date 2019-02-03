import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

import segmentation as sg
import importlib

def border_feature(image):
    
    out_contour = sg.segment(image)

    img_copy = image.copy()
    cv2.drawContours(img_copy, [out_contour], -1, (0, 255, 0), 2)
    
    hull = cv2.convexHull(out_contour, returnPoints=True)
    cv2.drawContours(img_copy, hull, -1, (255, 0, 0), 5)
    
    countour_area = cv2.contourArea(out_contour);
    hull_area = cv2.contourArea(hull)
    
    B1 = countour_area / hull_area
    
    return img_copy, B1

def border_feature_string(image):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return border_feature(img)
    
    