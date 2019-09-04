# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 18:20:26 2019

@author: Sveta
"""

# OpenCV
import cv2

# MumPy
import numpy as np

# Other imports
import csv
#import random
#from pathlib import Path
import pathlib as pl
import math as math

import sklearn
import sklearn.model_selection
import sklearn.svm
import sklearn.neighbors
import sklearn.neural_network
import skimage as ski
import sklearn.neighbors.typedefs
import sklearn.metrics
#import sklearn.neighbors.ball_tree
#import sklearn.neighbors.dist_metrics

#Sklearn
#from sklearn.svm import SVC
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.neural_network import MLPClassifier

#from sklearn.metrics import confusion_matrix

#from skimage.measure import compare_ssim
#from math import atan2, pi

# Keras - Tensorflow pravi problem
#from keras.models import Sequential
#from keras.layers.core import Dense,Activation
#from keras.optimizers import SGD

def load_dataset(file):

    return_set = []
    
    with open(file, 'r') as metadata:
        lines = csv.reader(metadata)
        dataset = list(lines)

        # Ne ucitava se ceo set podataka zbog neizbalansiranosti
        #for x in range(1, len(dataset)):
        for x in range(0, 4200):
            if dataset[x][2] == 'nv':
                return_set.append([dataset[x][1], 0])
            elif dataset[x][2] == 'mel':
                return_set.append([dataset[x][1], 1])
            
    #print("Total :", repr(len(return_set)))
                
    return return_set

def preprocess(img):
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    img_dilated = cv2.dilate(img_gray, kernel, iterations=5)
    img_eroded = cv2.erode(img_dilated, kernel, iterations=5)
    
    (score, img_diff) = ski.measure.compare_ssim(img_eroded, img_gray, full=True)
    
    img_diff = (img_diff * 255).astype("uint8")
    #print(img_diff.dtype)
    
    mask_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    img_dilated_mask = cv2.dilate(img_diff, mask_kernel, iterations=3)
    img_eroded_mask = cv2.erode(img_dilated_mask, mask_kernel, iterations=5)
    
    ret, img_bin_mask = cv2.threshold(img_eroded_mask, 0, 255, cv2.THRESH_OTSU)
    img_bin_mask_inv = cv2.bitwise_not(img_bin_mask)
    
    #img_copy_gray = np.copy(img_gray)
    img_copy_gray = img_gray
    img_copy_gray[img_bin_mask==0] = 0
    
    img_ip_gray = cv2.inpaint(img_copy_gray,img_bin_mask_inv,3,cv2.INPAINT_TELEA)
    
    ksize = 15

    img_final_g = cv2.medianBlur(img_ip_gray, ksize)
    
    return img_final_g   

def segment_pipe(img):
    
    ret, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    largestContour = max(contours, key = cv2.contourArea)
    
    return largestContour

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
    angle = math.atan2(eigenvectors[0,1], eigenvectors[0,0])
    angle_degrees = angle*180/math.pi
    
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

def border_pipe(in_contour):
      
    hull = cv2.convexHull(in_contour, returnPoints=True)
    
    countour_area = cv2.contourArea(in_contour);
    hull_area = cv2.contourArea(hull)
    
    B = countour_area / hull_area
    
    return B

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

def feature_extraction():
    
    goal_dataset = load_dataset('metadata.csv')
    
    features = []
    
    # Training
    i = 0
    not_found = 0
    for tr in goal_dataset:
        
        if i % 100 == 0: 
            print(i)
        i += 1
        
        img_path = 'images/' + tr[0] + '.jpg'
        path_obj = pl.Path(img_path)
        if path_obj.is_file():
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            pp_image = preprocess(img)
            contour = segment_pipe(pp_image)
            
            A = asymmetry_pipe(img, contour)
            B = border_pipe(contour)
            C = color_pipe(img, contour)
            
            
            features.append([tr[0], A, B, C, tr[1]])
         
        else:
            not_found += 1
    #features = normalize(normalize)            

    with open('features.csv', 'w', newline='') as csvFile:
        print()
        print("Writing features to file...")
        writer = csv.writer(csvFile)
        writer.writerows(features)

def means():
    pos = [] 
    neg = []
    
    with open('features.csv', 'r') as metadata:
        lines = csv.reader(metadata)
        dataset = list(lines)
        for x in range(0, len(dataset)):
            if dataset[x][4] == '0':
                neg.append(dataset[x])
            else:
                pos.append(dataset[x])
    
    print("\nPositives :", len(pos))
    print("Negatives :", len(neg))
    print()
    
    a_p = []
    a_n = []
    
    b_p = []
    b_n = []
    
    c_p = []
    c_n = []
    
    for p in pos:
        a_p.append(float(p[1]))
        b_p.append(float(p[2]))
        c_p.append(float(p[3]))
        
    for n in neg:
        a_n.append(float(n[1]))
        b_n.append(float(n[2]))
        c_n.append(float(n[3]))
    
    print("Positives symmetry mean :", np.mean(a_p))
    print("Negatives symmetry mean : ", np.mean(a_n))
    
    print("Positives border regularity mean :", np.mean(b_p))
    print("Negatives border regularity mean :", np.mean(b_n))
    
    print("Positives color deviation mean :", np.mean(c_p))
    print("Negatives color deviation mean :", np.mean(c_n))    
    print()
    
def normalize():
    
    data = []
    a_f = []
    b_f = []
    c_f = []    
    
    with open('features.csv', 'r') as metadata:
        lines = csv.reader(metadata)
        dataset = list(lines)
        dataset = np.array(dataset)
        img_names = dataset[:, 0]
        labels = dataset[:, 4]
        a_f = dataset[:, 1].astype(np.float)
        b_f = dataset[:, 2].astype(np.float)
        c_f = dataset[:, 3].astype(np.float)
        a_normed = a_f / a_f.max(axis=0)
        b_normed = b_f / b_f.max(axis=0)
        c_normed = c_f / c_f.max(axis=0)
        data = np.column_stack((img_names, a_normed, b_normed, c_normed, labels))
        
        with open('features.csv', 'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(data)

def prepare():
    
    with open('features.csv', 'r') as metadata:
        lines = csv.reader(metadata)
        dataset = list(lines)
        dataset = np.array(dataset)
        y = dataset[:, 4].astype(np.uint) # oznake
        # x = dataset[:, 1:4].astype(np.float) # atributi
        
        x = dataset[:, 0:4]

    return sklearn.model_selection.train_test_split(x, y, test_size=0.3)
    

def report(y_test, predicted_test, names):
    cm = sklearn.metrics.confusion_matrix(y_test,predicted_test)
    
    
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*((precision*recall)/(precision+recall))

    with open("predictions.txt", "w") as text_file:        
        print("Test size : ", names.size, file=text_file)
        for i in range(names.size):    
            print(names[i], end=" - ", file=text_file)
            print(predicted_test[i], file=text_file)


    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_test, predicted_test).ravel()
    specificity = tn / (tn+fp)

    print("\nReport")
    
    print("-------------")
    
    print("Accuracy : {:.3f}".format(sklearn.metrics.accuracy_score(y_test, predicted_test)))
    print("Recall : {:.3f}".format(recall))
    print("Precision {:.3f}".format(precision))
    print("F1 score {:.3f}".format(f1_score))
    print("Specificity: {:.3f}".format(specificity))

    print("-------------")
    
    with open("results.txt", "a") as text_file:
        print("Accuracy : {:.3f}".format(sklearn.metrics.accuracy_score(y_test, predicted_test)), file=text_file)
        print("Recall : {:.3f}".format(recall), file=text_file)
        print("Precision {:.3f}".format(precision), file=text_file)
        print("F1 score {:.3f}".format(f1_score), file=text_file)
        print("Specificity: {:.3f}".format(specificity), file=text_file)

def run_SVM(x_train, x_test, y_train, y_test):
    
    
    test_names = x_test[:, 0]
    
    x_test = x_test[:, 1:4].astype(np.float)
    x_train = x_train[:, 1:4].astype(np.float)
    
    clf_svm = sklearn.svm.SVC(kernel='linear', probability=True) 
    clf_svm.fit(x_train, y_train)
    y_train_pred = clf_svm.predict(x_train)
    y_test_pred = clf_svm.predict(x_test)
    
    print("Train accuracy: ", sklearn.metrics.accuracy_score(y_train, y_train_pred))
    print("Validation accuracy: ", sklearn.metrics.accuracy_score(y_test, y_test_pred))
    
    with open("results.txt", "a") as text_file:
        print(f"\nSVM", file=text_file)
        print(f"Train accuracy: ", sklearn.metrics.accuracy_score(y_train, y_train_pred), file=text_file)
        print(f"Validation accuracy: ", sklearn.metrics.accuracy_score(y_test, y_test_pred), file=text_file)

    report(y_test, y_test_pred, test_names)

    return y_train_pred, y_test_pred

def run_KNN(x_train, x_test, y_train, y_test):
    
    best_score = [0,0,0]
    ret_vals = [0,0]
    
    test_names = x_test[:, 0]
    
    x_test = x_test[:, 1:4].astype(np.float)
    x_train = x_train[:, 1:4].astype(np.float)
    
    for n in range(1, 20):
        clf_knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n)
        clf_knn = clf_knn.fit(x_train, y_train)
        y_train_pred = clf_knn.predict(x_train)
        y_test_pred = clf_knn.predict(x_test)
        
        temp = sklearn.metrics.accuracy_score(y_test, y_test_pred)
        if (temp > best_score[1]):
            best_score[0] = sklearn.metrics.accuracy_score(y_train, y_train_pred)
            best_score[1] = sklearn.metrics.accuracy_score(y_test, y_test_pred)
            best_score[2] = n
            
            ret_vals[0] = y_train_pred
            ret_vals[1] = y_test_pred
            
        print("Train accuracy: ", sklearn.metrics.accuracy_score(y_train, y_train_pred))
        print("Validation accuracy: ", sklearn.metrics.accuracy_score(y_test, y_test_pred))
        
    with open("results.txt", "a") as text_file:
        print(f"\nKNN", file=text_file)
        print(f"n =", best_score[2], file=text_file)
        print(f"Train accuracy: ", best_score[0], file=text_file)
        print(f"Validation accuracy: ", best_score[1], file=text_file)
        
    report(y_test, ret_vals[1], test_names)
        
    return ret_vals

# Problem sa Tensorflow-om
# =============================================================================
# 
# def winner(output): # output je vektor sa izlaza neuronske mreze
#     return max(enumerate(output), key=lambda x: x[1])[0]
# 
# 
# def run_ANN(x_train, x_test, y_train, y_test):
#     
#     ann = Sequential()
#     ann.add(Dense(5, input_dim=784, activation='relu'))
#     ann.add(Dense(2, activation='relu'))
#      
#     # definisanje parametra algoritma za obucavanje
#     sgd = SGD(lr=0.01, momentum=0.9) # Stohastic gradient descent
#     ann.compile(loss='mean_squared_error', optimizer=sgd)
# 
#     # obucavanje neuronske mreze
#     ann.fit(x_train, y_train, epochs=2000, batch_size=1, verbose = 0, shuffle=False) 
#     
#     result = ann.predict(x_test)
#     
#     print(result)
#     
# =============================================================================
    
def run_MLP(x_train, x_test, y_train, y_test):
    
    test_names = x_test[:, 0]
    
    x_test = x_test[:, 1:4].astype(np.float)
    x_train = x_train[:, 1:4].astype(np.float)
    
    clf_mlp = sklearn.neural_network.MLPClassifier(solver='sgd', alpha=1e-5, 
                        hidden_layer_sizes=(5,3),
                        activation='relu',
                        learning_rate='adaptive',
                        learning_rate_init=0.001,
                        momentum=0.9,
                        max_iter=5000
                    )
    clf_mlp.fit(x_train, y_train)
    y_train_pred = clf_mlp.predict(x_train)
    y_test_pred = clf_mlp.predict(x_test)
    print("Train accuracy: ", sklearn.metrics.accuracy_score(y_train, y_train_pred))
    print("Validation accuracy: ", sklearn.metrics.accuracy_score(y_test, y_test_pred))
    
    with open("results.txt", "a") as text_file:
        print(f"\nMLP", file=text_file)
        print(f"Train accuracy: ", sklearn.metrics.accuracy_score(y_train, y_train_pred), file=text_file)
        print(f"Validation accuracy: ", sklearn.metrics.accuracy_score(y_test, y_test_pred), file=text_file)
    
    report(y_test, y_test_pred, test_names)
    
    return y_train_pred, y_test_pred
    
def count_data():
    with open('metadata.csv', 'r') as metadata:
        lines = csv.reader(metadata)
        dataset = list(lines)
    
        nv, mel = 0,0
        for x in range(1, 4200):
            if dataset[x][2] == 'nv':
                nv += 1
            elif dataset[x][2] == 'mel':
                mel += 1
            
    print("Negatives :", nv)
    print("Positives :", mel)

def main():
    while(1):
        print("Menu: ")
        print(">> 1. Extract features")
        print(">> 2. SVM classification")
        print(">> 3. KNN classification")
        print(">> 4. MLP classification")
        print(">> 5. Normalize")
        print(">> 6. Show means")
        print(">> 0. Quit")

        option = int(input("Select your option: "))
        print();
        if (option == 1):
            print("Features extraction...")
           
            #count_data()
            feature_extraction()
            
            print("\nFeatures extraction finished\n")
        elif(option == 2):
            print("\nClassification in progress...\n")
            
            x_train, x_test, y_train, y_test = prepare() 
            
            print()
            
            run_SVM(x_train, x_test, y_train, y_test)
            
            print("\nClassification finished\n")
        elif(option == 3):
            print("\nClassification in progress...\n")
            
            x_train, x_test, y_train, y_test = prepare() 
            
            run_KNN(x_train, x_test, y_train, y_test)
            
            print("\nClassification finished!\n")
        elif option == 4:
            print("\nClassification in progress...\n")
            
            x_train, x_test, y_train, y_test = prepare() 
            
            run_MLP(x_train, x_test, y_train, y_test)
            
            print("\nClassification finished!\n")
        elif option == 5:
            print("\nNormalizing...")
            
            normalize()
            
            print("\nFinished normalizing\n")
        elif option == 6:
            
            means()
        
        elif option == 0:
            break
        else:
            print("Invalid option chosen")
            print("Please try again\n")



main()