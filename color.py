import numpy as np
import cv2
import segmentation as sg
import matplotlib
import matplotlib.pyplot as plt
import importlib

def color_feature(img_c):
    importlib.reload(sg)
    
    out_contour_c = sg.segment(img_c)
    
    color_mask = np.zeros(img_c.shape, np.uint8)
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2GRAY)
    color_mask = cv2.drawContours(color_mask, [out_contour_c], -1, (255), thickness=-1)
    
    # Primeceno da svetlija boja pri ivicama utice na devijaciju boja
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    color_mask = cv2.erode(color_mask, kernel, iterations = 1)
    
    mask3 = np.dstack((color_mask, color_mask, color_mask))
    
    img_c = cv2.bitwise_and(img_c, mask3)
    
    contour_pixels = img_c[(img_c[:,:,0] != 0) & (img_c[:,:,1] != 0) & (img_c[:,:,2] != 0)]
    
    stddev = np.std(contour_pixels, axis=0)
    
    return (stddev[0] + stddev[1] + stddev[2])/3;
    
def color_pipe(original_image, contour):
    
    color_mask = np.zeros(original_image.shape, np.uint8)
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2GRAY)
    
    # Ucrtava konturu na binarnu sliku
    color_mask = cv2.drawContours(color_mask, [contour], -1, (255), thickness=-1)
    
    # Primeceno da svetlija boja pri ivicama utice na devijaciju boja
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    color_mask = cv2.erode(color_mask, kernel, iterations = 1)
    
    mask3 = np.dstack((color_mask, color_mask, color_mask))
    
    img_c = cv2.bitwise_and(original_image, mask3)
    
    contour_pixels = img_c[(img_c[:,:,0] != 0) & (img_c[:,:,1] != 0) & (img_c[:,:,2] != 0)]
    
    stddev = np.std(contour_pixels, axis=0)
    
    C = (stddev[0] + stddev[1] + stddev[2])/3
    
    return C;
