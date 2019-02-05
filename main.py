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
from pathlib import Path

#Sklearn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Keras - Tensorflow pravi problem
#from keras.models import Sequential
#from keras.layers.core import Dense,Activation
#from keras.optimizers import SGD

# Potrebni moduli koji su kreirani
import asymmetry as a_module
import border as b_module
import color as c_module
import preprocessing as p_module
import segmentation as s_module

def load_dataset(file):
    
    return_set = []
    
    with open(file, 'r') as metadata:
        lines = csv.reader(metadata)
        dataset = list(lines)

        # Ne ucitava se ceo set podataka zbog neizbalansiranosti
        #for x in range(1, len(dataset)):
        for x in range(1, 4200):
            if dataset[x][2] == 'nv':
                return_set.append([dataset[x][1], 0])
            elif dataset[x][2] == 'mel':
                return_set.append([dataset[x][1], 1])
            
    #print("Total :", repr(len(return_set)))
                
    return return_set

def feature_extraction():
    
    goal_dataset = load_dataset('metadata.csv')
    
    features = []
    
    # Training
    i = 0
    for tr in goal_dataset:
        
        if i % 100 == 0: 
            print(i)
        i += 1
        
        img_path = 'images/' + tr[0] + '.jpg'
        path_obj = Path(img_path)
        if path_obj.is_file():
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            pp_image = p_module.preprocess(img)
            contour = s_module.segment_pipe(pp_image)
            
            A = a_module.asymmetry_pipe(img, contour)
            B = b_module.border_pipe(contour)
            C = c_module.color_pipe(img, contour)
            
            
            features.append([tr[0], A, B, C, tr[1]])
         
            
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
        for x in range(1, len(dataset)):
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
        x = dataset[:, 1:4].astype(np.float) # atributi
        
        
    return train_test_split(x, y, test_size=0.3)
    

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

def run_KNN(x_train, x_test, y_train, y_test):
    
    best_score = [0,0,0]
    ret_vals = [0,0]
    
    for n in range(1, 20):
        clf_knn = KNeighborsClassifier(n_neighbors=n)
        clf_knn = clf_knn.fit(x_train, y_train)
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
    clf_mlp = MLPClassifier(solver='sgd', alpha=1e-5, 
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
    print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
    print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))
    
    with open("results.txt", "a") as text_file:
        print(f"\nMLP", file=text_file)
        print(f"Train accuracy: ", accuracy_score(y_train, y_train_pred), file=text_file)
        print(f"Validation accuracy: ", accuracy_score(y_test, y_test_pred), file=text_file)
    
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