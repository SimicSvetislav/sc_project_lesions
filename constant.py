import os
import tensorflow as tf

INPUT_SIZE = 4200
FOLDS = 5
DATASET_DIR = os.getcwd() + '\\filtered\\'
METADATA_FILE = 'metadata.csv'
IMG_SIZE = 160
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
SPLIT_WEIGHTS = (8, 1, 1)
BATCH_SIZE = 16
SHUFFLE_BUFFER_SIZE = 1000
AUTOTUNE = tf.data.experimental.AUTOTUNE
EPOCHS = 50