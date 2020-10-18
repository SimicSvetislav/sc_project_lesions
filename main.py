# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 18:20:26 2019

@author: Sveta
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# OpenCV
import cv2

# MumPy
import numpy as np

# Other imports
import csv
#import random
from pathlib import Path
import time
import os
# import matplotlib.pyplot as plt
import shutil

#Sklearn
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# constants
import constant

import concurrent.futures
import multiprocessing
from math import ceil

import skimage.data
import skimage.transform

# Potrebni moduli koji su kreirani
import asymmetry as a_module
import border as b_module
import color as c_module
import preprocessing as p_module
import segmentation as s_module


accuracies = []
precisions = []
specifities = []
recalls = []
f1_scores = []

gs = {}
cs = {}
ks = {}

fs = [0]


def report(y_test, predicted_test, names):
    cm = confusion_matrix(y_test, predicted_test)
    
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*((precision*recall)/(precision+recall))
    accuracy = accuracy_score(y_test, predicted_test)
    specificity = tn / (tn+fp)

    with open("predictions.txt", "w") as text_file:        
        print("Test size : ", names.size, file=text_file)
        for i in range(names.size):    
            print(names[i], end=" - ", file=text_file)
            print(predicted_test[i], file=text_file)

    print("\nReport")
    
    print("-------------")
    
    print("Accuracy : {:.3f}".format(accuracy))
    print("Recall : {:.3f}".format(recall))
    print("Precision {:.3f}".format(precision))
    print("F1 score {:.3f}".format(f1_score))
    print("Specificity: {:.3f}".format(specificity))

    print("-------------")
    
    '''
    with open("results.txt", "a") as text_file:
        print("Accuracy : {:.3f}".format(accuracy_score(y_test, predicted_test)), file=text_file)
        print("Recall : {:.3f}".format(recall), file=text_file)
        print("Precision {:.3f}".format(precision), file=text_file)
        print("F1 score {:.3f}".format(f1_score), file=text_file)
        print("Specificity: {:.3f}".format(specificity), file=text_file)
    '''
    
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)
    specifities.append(specificity)


def load_dataset(file):
    
    return_set = []
    pos = 0
    neg = 0
    
    with open(file, 'r') as metadata:
        lines = csv.reader(metadata)
        dataset = list(lines)[1:]

        # Ne ucitava se ceo set podataka zbog neizbalansiranosti
        for x in range(1, len(dataset)):
        # for x in range(constant.INPUT_SIZE):
            if dataset[x][2] == 'nv' and pos < 1225:
                return_set.append([dataset[x][1], 0])
                pos += 1
            elif dataset[x][2] == 'mel':
                return_set.append([dataset[x][1], 1])
                neg += 1
            
    print("Positives :", pos)
    print("Negatives :", neg)
    #print("Total :", repr(len(return_set)))
                
    return return_set


def filter_classes(file='metadata.csv'):
    
    with open(file, 'r') as metadata:
        lines = csv.reader(metadata)
        dataset = list(lines)
        
        root_path = os.getcwd()
        # print(root_path)
        
        all_images_path = root_path + '\\images\\'
        # print(all_images_path)
        
        destination_path = root_path + '\\filtered\\'
        # print(destination_path)
        
        valid = 0
        for x in range(1, len(dataset)):
            data_class = dataset[x][2]
            if data_class == 'nv' or data_class == 'mel':
                img_name = dataset[x][1]
                # print(img_name)
                
                img_path = all_images_path + img_name + '.jpg'
                # print(img_path)

                path_obj = Path(img_path)
                if path_obj.is_file():
                    valid += 1
                    # shutil.copy2(img_path, destination_path + data_class + '_' + img_name + '.jpg')
                    shutil.copy2(img_path, destination_path)
    
        print("Relevant images :", valid)


def extract_chunk(data, proc_num, correction = False, skip_B = False):
    
    features = []
    
    i = 0
    
    print("Total data length :", len(data))
    
    for tr in data:
        
        if i % 100 == 0: 
            print(i, "- Process", proc_num)
        i += 1
    
        img_path = 'filtered/' + tr[0] + '.jpg'
        path_obj = Path(img_path)
        
        if path_obj.is_file():
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            n = 0
            
            A = B = C = 0
            
            if correction:
                while n < 10: 
                    pp_image, pp_image_color = p_module.preprocess(img)
                    contour = s_module.segment_pipe(pp_image)
                    
                    A = a_module.asymmetry_pipe(img, contour)
                    
                    if A > 0.3:
                        break
                    
                    height, width, channels = img.shape
                    img = img[int(0.05*height):int(0.95*height), int(0.05*width):int(0.95*width)]
                    
                    n += 1
                else:
                    # Neuspesna ekstrakcija obelezja
                    print("Izbaceno :", proc_num)
                    continue
            else:
                pp_image, pp_image_color = p_module.preprocess(img)
                contour = s_module.segment_pipe(pp_image)
            
                A = a_module.asymmetry_pipe(pp_image_color, contour)    
            
            if not skip_B:
                B = b_module.border_pipe(contour)
            
            C = c_module.color_pipe(pp_image_color, contour)
            
            features.append([tr[0], A, B, C, tr[1]])
            
    # print("Process", proc_num, "- relevant data size :", len(features))
        
    return features
    

def parallel_feature_extraction_man():
    
    start_time = time.time()
    
    goal_dataset = load_dataset('metadata.csv')
   
    dataset_size = len(goal_dataset)
    print("\nDataset size :", dataset_size)
    
    cores = multiprocessing.cpu_count()
    print("Number of cores:", cores, end="\n\n")
    
    features = []
    futures = [None, None, None, None]
    
    if __name__ == "__main__":
        with concurrent.futures.ProcessPoolExecutor() as executor:
            
            goal_dataset = load_dataset('metadata.csv')
            
            chunked_data = chunks(goal_dataset, ceil(dataset_size / cores))
            
            skip_B = False
            correction = True
            
            print("Skip B :", skip_B)
            print("Correction :", correction, end='\n\n')
            
            for i in range(cores):
                futures[i] = executor.submit(extract_chunk, next(chunked_data), i+1, correction, skip_B)
                
            for i, future in enumerate(futures, start=1):
                print("Result", i, "length", len(future.result()))
                features.extend(future.result())
    
    print("Final features length :", len(features))
    
    with open('features.csv', 'w', newline='') as csvFile:
        print()
        print("Writing features to file...")
        writer = csv.writer(csvFile)
        writer.writerows(features)
        print("Features written to file features.csv")
        
    return time.time() - start_time


def chunks(data, size):
    return (data[i:i+size] for i in range(0, len(data), size))


def feature_extraction():
    
    start_time = time.time()
    
    goal_dataset = load_dataset('metadata.csv')
    
    print("\nDataset size :", len(goal_dataset))
    
    features = []
    
    # Training
    i = 0
    not_found = 0
    for tr in goal_dataset:
        
        if i % 100 == 0: 
            print(i)
        i += 1
        
        img_path = 'images/' + tr[0] + '.jpg'
        path_obj = Path(img_path)
        if path_obj.is_file():
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            pp_image, pp_image_color = p_module.preprocess(img)
            contour = s_module.segment_pipe(pp_image)
            
            A = None
            B = None
            C = None
            
            A = a_module.asymmetry_pipe(pp_image_color, contour)
            B = b_module.border_pipe(contour)
            C = c_module.color_pipe(pp_image_color, contour)
            
            features.append([tr[0], A, B, C, tr[1]])
        else:
            not_found += 1
         
    #features = normalize(normalize)            

    with open('features.csv', 'w', newline='') as csvFile:
        print()
        print("Writing features to file...")
        writer = csv.writer(csvFile)
        writer.writerows(features)
        print("Features written to file features.csv")
        
    return time.time() - start_time


def means():
    pos = [] 
    neg = []
    
    with open('features.csv', 'r') as metadata:
        lines = csv.reader(metadata)
        dataset = list(lines)
        for x in range(len(dataset)):
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
        max_b = b_f.max(axis=0)
        if max_b != 0:
            b_normed = b_f / max_b
        else:
            b_normed = b_f
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
        
    return train_test_split(x, y, test_size=0.2)


def run_svm_cv(x_train, x_test, y_train, y_test, folds):
    
    test_names = x_test[:, 0]
    
    x_test = x_test[:, 1:4].astype(np.float)
    x_train = x_train[:, 1:4].astype(np.float)
    
    clf_svm = SVC(kernel='rbf') 
    
    Cs = [0.001, 0.01, 0.1, 1, 10, 100]
    gammas = [0.001, 0.01, 0.1, 1, 10, 100]
    param_grid = {'C': Cs, 'gamma' : gammas}
    classifier = GridSearchCV(clf_svm, param_grid, cv=folds)
    
    classifier.fit(x_train, y_train)
    print("Best params :", classifier.best_params_)
    print("Best score :", classifier.best_score_)

    y_train_pred = classifier.predict(x_train)
    y_test_pred = classifier.predict(x_test)
    
    print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
    print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))
    
    with open("results.txt", "a") as text_file:
        print(f"\nSVM", file=text_file)
        print(f"Train accuracy: ", accuracy_score(y_train, y_train_pred), file=text_file)
        print(f"Validation accuracy: ", accuracy_score(y_test, y_test_pred), file=text_file)
        
    report(y_test, y_test_pred, test_names)
    
    try:
        gs[str(classifier.best_params_['gamma'])] += 1
    except:
        gs[str(classifier.best_params_['gamma'])] = 1
        
    try:
        cs[str(classifier.best_params_['C'])] += 1
    except:
        cs[str(classifier.best_params_['C'])] = 1

    return y_train_pred, y_test_pred


def run_SVM(x_train, x_test, y_train, y_test):
   
    clf_svm = SVC(kernel='linear', probability=True) 
    clf_svm.fit(x_train, y_train)
    y_train_pred = clf_svm.predict(x_train)
    y_test_pred = clf_svm.predict(x_test)
    
    print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
    print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))

    
    with open("results.txt", "a") as text_file:
        print(f"\nSVM", file=text_file)
        print(f"Train accuracy: ", accuracy_score(y_train, y_train_pred), file=text_file)
        print(f"Validation accuracy: ", accuracy_score(y_test, y_test_pred), file=text_file)

    return y_train_pred, y_test_pred


def run_knn_cv(x_train, x_test, y_train, y_test, folds):
    
    clf_knn = KNeighborsClassifier()
    params = {'n_neighbors': np.arange(1,43)}
    knn_gscv = GridSearchCV(clf_knn, params, cv=folds)
    
    knn_gscv.fit(x_train, y_train)
    
    print("Best n :", knn_gscv.best_params_)
    print("Best score :", knn_gscv.best_score_)
    
    y_train_pred = knn_gscv.predict(x_train)
    y_test_pred = knn_gscv.predict(x_test)
    
    print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
    print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))
    
    with open("results.txt", "a") as text_file:
        print(f"\nKNN", file=text_file)
        print(f"n =", knn_gscv.best_score_, file=text_file)
        print(f"Train accuracy: ", accuracy_score(y_train, y_train_pred), file=text_file)
        print(f"Validation accuracy: ", accuracy_score(y_test, y_test_pred), file=text_file)
    
    report(y_test, y_test_pred, None)
    
    try:
        ks[str(knn_gscv.best_params_['n_neighbors'])] += 1
    except:
        ks[str(knn_gscv.best_params_['n_neighbors'])] = 1
    
    return y_train_pred, y_test_pred


def run_KNN(x_train, x_test, y_train, y_test):
    
    best_score = [0,0,0]
    ret_vals = [0,0]
    
    for n in range(1, 20):
        clf_knn = KNeighborsClassifier(n_neighbors=n)
        clf_knn.fit(x_train, y_train)
        y_train_pred = clf_knn.predict(x_train)
        y_test_pred = clf_knn.predict(x_test)
        
        temp = accuracy_score(y_test, y_test_pred)
        if (temp > best_score[1]):
            best_score[0] = accuracy_score(y_train, y_train_pred)
            best_score[1] = accuracy_score(y_test, y_test_pred)
            best_score[2] = n
            
            ret_vals[0] = y_train_pred
            ret_vals[1] = y_test_pred
            
        print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
        print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))
        
        
        
    with open("results.txt", "a") as text_file:
        print(f"\nKNN", file=text_file)
        print(f"n =", best_score[2], file=text_file)
        print(f"Train accuracy: ", best_score[0], file=text_file)
        print(f"Validation accuracy: ", best_score[1], file=text_file)
        
    return ret_vals
    
def run_MLP(x_train, x_test, y_train, y_test):
    
    clf_mlp = MLPClassifier(solver='sgd', alpha=1e-5, 
                        hidden_layer_sizes=(7,7),
                        activation='relu',
                        learning_rate='adaptive',
                        learning_rate_init=0.1,
                        momentum=0.9,
                        max_iter=1000
                    )
    clf_mlp.fit(x_train, y_train)
    y_train_pred = clf_mlp.predict(x_train)
    y_test_pred = clf_mlp.predict(x_test)
    print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
    acc = accuracy_score(y_test, y_test_pred)
    print("Validation accuracy: ", acc)
    
    if acc < 0.61:
        fs[0] += 1
    
    with open("results.txt", "a") as text_file:
        print(f"\nMLP", file=text_file)
        print(f"Train accuracy: ", accuracy_score(y_train, y_train_pred), file=text_file)
        print(f"Validation accuracy: ", accuracy_score(y_test, y_test_pred), file=text_file)
    
    report(y_test, y_test_pred, None)
    
    return y_train_pred, y_test_pred
    
def count_data():
    with open('metadata.csv', 'r') as metadata:
        lines = csv.reader(metadata)
        dataset = list(lines)
    
        nv, mel = 0,0
        for x in range(constant.INPUT_SIZE):
            if dataset[x][2] == 'nv':
                nv += 1
            elif dataset[x][2] == 'mel':
                mel += 1
            
    print("Negatives :", nv)
    print("Positives :", mel)
    

def get_labels(directory, file):
    
    return_dict = {}
    
    with open(file, 'r') as metadata:
        lines = csv.reader(metadata)
        dataset = list(lines)

        for x in range(len(dataset)):
        # for x in range(constant.INPUT_SIZE):
            img_path = constant.DATASET_DIR + dataset[x][1] + '.jpg'
            path_obj = Path(img_path)
            if not path_obj.is_file():
                continue
            if dataset[x][2] == 'nv':
                return_dict[img_path] = 0
            elif dataset[x][2] == 'mel':
                return_dict[img_path] = 1
            
    print("Dictionary length :", len(return_dict), end="\n\n")
    # print("Random value :", return_dict['ISIC_0024306'])
                
    return return_dict
    
def get_batch(images, labels, num, batch_size):
    
    indices = []
    
    start_ind = num*batch_size
    
    if start_ind + batch_size > len(images):
        indices = list(range(start_ind, len(images)))
    else:
        indices = list(range(start_ind, start_ind + batch_size))
    
    ret_images = images[indices]
    ret_labels = labels[indices]
    
    return ret_images, ret_labels
    

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
           
            # count_data()
            # elapsed_time = feature_extraction()
            elapsed_time = parallel_feature_extraction_man()
            
            print("Time elapsed: {:.2f} seconds".format(elapsed_time))
            
            print("\nFeatures extraction finished\n")
        elif(option == 2):
            print("\nClassification in progress...\n")
            
            for i in range(1):
            
                print("\nITERATION", i+1)
                print()
                
                x_train, x_test, y_train, y_test = prepare() 
                
                # run_SVM(x_train, x_test, y_train, y_test)
                run_svm_cv(x_train, x_test, y_train, y_test, constant.FOLDS)
            
            print("ACCURACY:")
            np_arr = np.array(accuracies)
            print("Mean :", np.mean(np_arr, axis=0))
            print("Min :", np.min(np_arr, axis=0))
            print("Max :", np.max(np_arr, axis=0))
            
            print("PRECISION:")
            np_arr = np.array(precisions)
            print("Mean :", np.mean(np_arr, axis=0))
            print("Min :", np.min(np_arr, axis=0))
            print("Max :", np.max(np_arr, axis=0))
            
            print("RECALL:")
            np_arr = np.array(recalls)
            print("Mean :", np.mean(np_arr, axis=0))
            print("Min :", np.min(np_arr, axis=0))
            print("Max :", np.max(np_arr, axis=0))
            
            print("F1 SCORE:")
            np_arr = np.array(f1_scores)
            print("Mean :", np.mean(np_arr, axis=0))
            print("Min :", np.min(np_arr, axis=0))
            print("Max :", np.max(np_arr, axis=0))
            
            print("SPECIFICITY")
            np_arr = np.array(specifities)
            print("Precision mean :", np.mean(np_arr, axis=0))
            print("Precision min :", np.min(np_arr, axis=0))
            print("Precision max :", np.max(np_arr, axis=0))
            
            print("GAMMAS")
            for key in gs:
                print("Key:", key, "-", gs[key])
                
            print("CS")
            for key in cs:
                print("Key:", key, "-", cs[key])
            
            print("\nClassification finished\n")
        elif(option == 3):
            print("\nClassification in progress...\n")
            
            for i in range(100):
            
                print("\nITERATION", i+1)
                print()
                
                x_train, x_test, y_train, y_test = prepare() 
                
                # run_KNN(x_train, x_test, y_train, y_test)
                run_knn_cv(x_train, x_test, y_train, y_test, constant.FOLDS)
            
            print("ACCURACY:")
            np_arr = np.array(accuracies)
            print("Mean :", np.mean(np_arr, axis=0))
            print("Min :", np.min(np_arr, axis=0))
            print("Max :", np.max(np_arr, axis=0))
            
            print("PRECISION:")
            np_arr = np.array(precisions)
            print("Mean :", np.mean(np_arr, axis=0))
            print("Min :", np.min(np_arr, axis=0))
            print("Max :", np.max(np_arr, axis=0))
            
            print("RECALL:")
            np_arr = np.array(recalls)
            print("Mean :", np.mean(np_arr, axis=0))
            print("Min :", np.min(np_arr, axis=0))
            print("Max :", np.max(np_arr, axis=0))
            
            print("F1 SCORE:")
            np_arr = np.array(f1_scores)
            print("Mean :", np.mean(np_arr, axis=0))
            print("Min :", np.min(np_arr, axis=0))
            print("Max :", np.max(np_arr, axis=0))
            
            print("SPECIFICITY")
            np_arr = np.array(specifities)
            print("Precision mean :", np.mean(np_arr, axis=0))
            print("Precision min :", np.min(np_arr, axis=0))
            print("Precision max :", np.max(np_arr, axis=0))
            
            print("KS")
            for key in ks:
                print("Key:", key, "-", ks[key])
            
            print("\nClassification finished!\n")
        elif option == 4:
            print("\nClassification in progress...\n")
            
            for i in range(100):
            
                print("\nITERATION", i+1)
                print()
                
                x_train, x_test, y_train, y_test = prepare() 
                
                run_MLP(x_train, x_test, y_train, y_test)
            
            print("ACCURACY:")
            np_arr = np.array(accuracies)
            print("Mean :", np.mean(np_arr, axis=0))
            print("Min :", np.min(np_arr, axis=0))
            print("Max :", np.max(np_arr, axis=0))
            
            print("PRECISION:")
            np_arr = np.array(precisions)
            print("Mean :", np.mean(np_arr, axis=0))
            print("Min :", np.min(np_arr, axis=0))
            print("Max :", np.max(np_arr, axis=0))
            
            print("RECALL:")
            np_arr = np.array(recalls)
            print("Mean :", np.mean(np_arr, axis=0))
            print("Min :", np.min(np_arr, axis=0))
            print("Max :", np.max(np_arr, axis=0))
            
            print("F1 SCORE:")
            np_arr = np.array(f1_scores)
            print("Mean :", np.mean(np_arr, axis=0))
            print("Min :", np.min(np_arr, axis=0))
            print("Max :", np.max(np_arr, axis=0))
            
            print("SPECIFICITY")
            np_arr = np.array(specifities)
            print("Precision mean :", np.mean(np_arr, axis=0))
            print("Precision min :", np.min(np_arr, axis=0))
            print("Precision max :", np.max(np_arr, axis=0))
            
            print("Fails :", fs[0])
            
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

if __name__ == "__main__":
    main()