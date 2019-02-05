import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

import segmentation as sg
import importlib

from math import atan2, sqrt, sin, cos, pi

matplotlib.rcParams['figure.figsize'] = 8,6

def asymmetry(img_a):
    
    importlib.reload(sg)
    
    out_contour = sg.segment(img_a)
    
    #img_copy = img_a.copy()
    
    lesion_binary = np.zeros(img_a.shape, np.uint8)
    lesion_binary = cv2.cvtColor(lesion_binary, cv2.COLOR_BGR2GRAY)
    lesion_binary = cv2.drawContours(lesion_binary, [out_contour], -1, (255), thickness=-1)
    
    #img_axis = img_a.copy()
    #img_axis_bin = lesion_binary.copy()
    points = np.empty((len(out_contour),2), dtype=np.float64)
    for i in range(points.shape[0]):
        points[i,0] = out_contour[i,0,0]
        points[i,1] = out_contour[i,0,1]
    
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean)
    
    pca_center = (int(mean[0,0]), int(mean[0,1]))
    
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0])
    angle_degrees = angle*180/pi
    
    # Poklapanja pri rotiranju po glavnoj osi
    lb1 = lesion_binary.copy()
    
    matrix = cv2.getRotationMatrix2D(pca_center, angle_degrees - 90, 1.0)
    
    # Korekcija i pravljenje duplikata
    lb1 = cv2.warpAffine(lb1, matrix, lb1.shape[1::-1], flags = cv2.INTER_LINEAR)
    lb2 = lb1.copy()
    
    # Rotacija oko glavne ose
    matrix = cv2.getRotationMatrix2D(pca_center, 180, 1.0)
    
    lb1 = cv2.warpAffine(lb1, matrix, lb1.shape[1::-1], flags = cv2.INTER_LINEAR)
    
    difference = cv2.bitwise_and(lb1, lb2)
    
    deviation = (np.count_nonzero(difference)/np.count_nonzero(lb1))
    
    return deviation
    
def asymmertryString(img):
    img_a = cv2.imread(img)
    img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    
    return asymmetry(img_a)

def asymmetry_pipe(original_image, contour):
    
    lesion_binary = np.zeros(original_image.shape, np.uint8)
    lesion_binary = cv2.cvtColor(lesion_binary, cv2.COLOR_BGR2GRAY)
    
    # Ucrtavanje konture na binarnu sliku
    lesion_binary = cv2.drawContours(lesion_binary, [contour], -1, (255), thickness=-1)
    
    points = np.empty((len(contour),2), dtype=np.float64)
    for i in range(points.shape[0]):
        points[i,0] = contour[i,0,0]
        points[i,1] = contour[i,0,1]
    
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean)
    
    # Centar konture u osnovu na koga ce se vrsiti rotacije
    pca_center = (int(mean[0,0]), int(mean[0,1]))
    
    # Izracunavanje ugla glavne ose konture
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0])
    angle_degrees = angle*180/pi
    
    # Poklapanja pri rotiranju po glavnoj osi
    lb1 = lesion_binary.copy()
    
    # Korekcija orijentacije slike i pravljenje duplikata
    matrix = cv2.getRotationMatrix2D(pca_center, angle_degrees - 90, 1.0)
    lb1 = cv2.warpAffine(lb1, matrix, lb1.shape[1::-1], flags = cv2.INTER_LINEAR)
    lb2 = lb1.copy()
    
    # Rotacija oko glavne ose
    matrix = cv2.getRotationMatrix2D(pca_center, 180, 1.0)
    lb1 = cv2.warpAffine(lb1, matrix, lb1.shape[1::-1], flags = cv2.INTER_LINEAR)
    
    matching_conture = cv2.bitwise_and(lb1, lb2)
    
    A = (np.count_nonzero(matching_conture)/np.count_nonzero(lb1))
   
    return A

#ii = cv2.imread('images/ISIC_0033834.jpg')
#ii = cv2.cvtColor(ii, cv2.COLOR_BGR2RGB)
#A1 = asymmetry(ii)
#print(A1)