import numpy as np
import matplotlib.pyplot as plt

from urllib.request import urlretrieve
from os.path import isfile, isdir, getsize
from os import mkdir, makedirs, remove, listdir
from tqdm import tqdm

import zipfile
import pickle

from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator

import glob
import shutil

import pickle

import constant

import csv
from pathlib import Path


from sklearn.linear_model import LogisticRegression
from keras import applications

data_path = 'data'
train_folder = data_path+'/train'
test_folder = data_path+'/test'
img_height = img_width = 100
channels = 3

def distribute():

    return_dict = {}
    
    with open(constant.METADATA_FILE, 'r') as metadata:
        lines = csv.reader(metadata)
        dataset = list(lines)
        print("Lines :", len(dataset))
        # for x in range(len(dataset)):
        for x in range(constant.INPUT_SIZE):
            img_path = constant.DATASET_DIR + dataset[x][1] + '.jpg'
            path_obj = Path(img_path)
            if not path_obj.is_file():
                continue
            if dataset[x][2] == 'nv':
                return_dict[img_path] = 0
            elif dataset[x][2] == 'mel':
                return_dict[img_path] = 1
    
    print("Valid :", len(return_dict))
    
    len_data = len(return_dict)
    
    train_num = int(len_data*0.8)
    test_num = len_data - train_num
    
    files = list(return_dict.keys())
    labels = list(return_dict.values())
    
    print("Files :", len(files))
    
    permutation = np.random.permutation(len_data)
    train_set = [files[i] for i in permutation[:][:train_num]]
    test_set = [files[i] for i in permutation[-test_num:]]
    train_labels = [labels[i] for i in permutation[:][:train_num]]
    test_labels = [labels[i] for i in permutation[-test_num:]]
    
    if isdir(train_folder):
        shutil.rmtree(train_folder)    
    if isdir(test_folder):
        shutil.rmtree(test_folder)    
    makedirs(train_folder+'/nv/')
    makedirs(train_folder+'/mel/')
    makedirs(test_folder+'/nv/')
    makedirs(test_folder+'/mel/')
    
    for f,i in zip(train_set, train_labels):
        if i==0:
            shutil.copy2(f, train_folder + '/nv/nv_' + f[f.index('ISIC'):])
        else:
            shutil.copy2(f, train_folder + '/mel/mel_' + f[f.index('ISIC'):])
            
    for f,i in zip(test_set, test_labels):
        if i==0:
            shutil.copy2(f, test_folder + '/nv/nv_' + f[f.index('ISIC'):])
        else:
            shutil.copy2(f, test_folder + '/mel/mel_' + f[f.index('ISIC'):])
    
    
def samples():
    
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = datagen.flow_from_directory(
        train_folder,
        color_mode = "rgb",
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode=None)
    
    i = 0
    img_list = []
    for batch in train_generator:
        img_list.append(batch)
        i += 1
        if i > 5:
            break
    
    for img in img_list:
        plt.imshow(np.squeeze(img))
        plt.show()
    
def logistic_regression():
    
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,
        zoom_range=0.2,
        horizontal_flip=True)
    
    batch_size = 1000
    train_generator = datagen.flow_from_directory(
        train_folder,
        color_mode = "rgb",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')
    
    x_train, y_train = next(train_generator)
    x_test, y_test = next(train_generator)
    
    logistic = LogisticRegression()
    logistic.fit(x_train.reshape(batch_size,-1), y_train)
    
    y_pred = logistic.predict(x_test.reshape(len(x_test), -1))
    print(y_pred[:10])
    
    print(logistic.predict_proba(x_test[:3].reshape(3,-1)))
    print(np.count_nonzero(y_pred == y_test)/len(y_test))
    
def tl():
    datagen = ImageDataGenerator(rescale=1.0/255)
    model = applications.VGG16(include_top=False, input_shape=(img_width, img_height, channels), weights='imagenet')
    model.summary()
    
    batch_size = 128
    generator = datagen.flow_from_directory(
        train_folder,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    
    valid_generator = datagen.flow_from_directory(
        test_folder,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    
    with open('bottleneck_features_train.npy','rb') as f:
        bottleneck_features_train = pickle.load(f)
        print(bottleneck_features_train.shape)
    
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_features_train.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    labels = np.array([0 if f.startswith('nv') else 1 for f in generator.filenames])[:len(bottleneck_features_train)]
    model.fit(bottleneck_features_train, labels, epochs=15, batch_size=batch_size)
    

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def get_bottleneck_features():
    if not isfile('bottleneck_features_train.npy'):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Bottleneck features') as pbar:
            urlretrieve(
                    'https://www.dropbox.com/s/a38gpvdcryw0kfc/bottleneck.zip?dl=1',
                    'bottleneck.zip',
                    pbar.hook)
    
        with zipfile.ZipFile('bottleneck.zip') as f:
            f.extractall('./')
            
        files = listdir('bottleneck 2/')
        
        for f in files:
            shutil.move('bottleneck 2/'+f,'./')
        shutil.rmtree('bottleneck 2/')
        remove('bottleneck.zip')
    
if __name__ == "__main__":
    
    # distribute()
    
    # samples()
    
    # logistic_regression()
    
    # get_bottleneck_features()
    
    tl()
    