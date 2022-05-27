import pandas as pd
import numpy as np
import numpy.random as rng
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from matplotlib import pyplot
matplotlib.use('Agg')
import PIL
import cv2
import mtcnn
import gc
from mtcnn.mtcnn import MTCNN
from PIL import Image
import os
import shutil
import random
import os
import gc
import re
import csv
import cv2
import sys
import PIL
import math
import time
import uuid
import mtcnn
import scipy
import scipy.sparse
import random
import pathlib
import pickle
import pandas as pd
import numpy as np
from mtcnn.mtcnn import MTCNN
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

retval = os.getcwd()
print("[>>SLD MODEL<<] Current working directory %s" % retval, \
      "__NAME__ = ", __name__)

from LooksLikeWho.SLD_tools import *
from LooksLikeWho.SLD_quad_model import *

import matplotlib.pyplot as plt
import seaborn as sns

import keras
from tensorflow import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, \
    Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D, Activation, Input, \
    Concatenate, multiply
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback as CB
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, \
    CSVLogger,EarlyStopping
from tensorflow.keras.utils import to_categorical  #EDITSLD
from tensorflow.keras import models
from tensorflow.keras.layers import Lambda, Layer

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.utils import class_weight

import warnings

############################### USER INPUTS ##################################

# PARAMETERS AND SETTING FOR THE DATASET CREATION AND TRAINING

## Technical parameters
local=True            # True if running locally
memory_issues=False   # True if GPU memory issues ("failed to allocate" / OOM)
disable_gpu=False     # True if above setting is not sufficient

## Generate train/test set
generate_train_test_sets=False  # True to gen train/test set, run only once

## Select data set (select only one amongst choices)
dataset_tag = "_litecropped" # 2 x 10 x 4
# dataset_tag = "_medium_cropped" # 2 x 100 x 4
# dataset_tag = "_mediumcroppedx10"# 2 x 100 x 10
# dataset_tag = "_ALL-HQ-UNZOOMED" 2 x 9131 x 4
# dataset_tag = "_ALL-HQ-UNZOOMED-10X" # 10931 x 10 (train) + 1931 x 4 (test)

## Set training parameters
MODEL_BASE_TAG = 'FaceNet'      # Select model among:
# 'MobileNetV2', 'ResNet50', 'VGG16', 'InceptionV3', 'Xception', 'FaceNet'
CUSTOM_FILE_NAME= "_quad_final" # Custom note
BATCH_SIZE = 8                  # Use 8 for final run
EPOCHS = 2                      # Use 6 for final run
STEPS_PER_EPOCH = 20          # Use 2000 for final run
k_ONESHOTMETRICS = 10           # Use 10 final run
START_LR = 0.001                # Adam default 0.001
MARGIN = 0.25                   # Use 0.25 for final run
MARGIN2 = 0.03                  # Use 0.03 for final run
EMBEDDINGSIZE = 128             # Use 10 final run
IMAGE_WIDTH, IMAGE_HEIGHT = (160, 160) if MODEL_BASE_TAG=="FaceNet" else (224, 224)
# MODELS CATALOG
models_catalog=[
    'MobileNetV2',
    'ResNet50',
    'VGG16',
    'InceptionV3',
    'Xception',
    'FaceNet']

models_objects=[
    MobileNetV2,
    ResNet50,
    VGG16,
    InceptionV3,
    Xception]

models_weight_paths=[
    r".\models\weights\mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5",
    r".\models\weights\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
    r".\models\weights\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
    r".\models\weights\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
    r".\models\weights\xception_weights_tf_dim_ordering_tf_kernels_notop.h5",
    r".\models\weights\facenet_keras_weights.h5"
]

class QuadrupletLossLayer(Layer):
    '''
    A custom tf.keras layer that computes the quadruplet loss from distances
    ap_dist, an_dist, and nn_dist.
    The computed loss is independant from the batch size.

    Arguments:
        alpha, beta: margin factors used in the loss formula.
        inputs: (ap_dist, an_dist, nn_dist) with:
            ap_dist: distance between the anchor image (A) and the positive
                image (P) (of the same class),
            an_dist: distance between the anchor image (A) and the first image
                of a different class (N1),
            nn_dist: distance between the two images from different classes N1
                and N2 (that do not belong to the anchor class).

    External Arguments:
        LooksLikeWho.SLD_models.BATCH_SIZE: batch size used for training

    Output:
        The quadruplet loss per sample (averaged over one batch).
    '''

    def __init__(self, alpha, beta, **kwargs):
        self.alpha = alpha
        self.beta = beta

        super(QuadrupletLossLayer, self).__init__(**kwargs)

    def quadruplet_loss(self, inputs):
        ap_dist,an_dist,nn_dist = inputs

        #square
        ap_dist2 = K.square(ap_dist)
        an_dist2 = K.square(an_dist)
        nn_dist2 = K.square(nn_dist)

        return (K.sum(K.maximum(ap_dist2 - an_dist2 + self.alpha, 0), axis=0) + K.sum(K.maximum(ap_dist2 - nn_dist2 + self.beta, 0), axis=0)) / BATCH_SIZE

    def call(self, inputs):
        loss = self.quadruplet_loss(inputs)
        self.add_loss(loss)
        return loss

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'alpha': self.alpha,
            'beta': self.beta,
        })
        return config


def get_what_from_full_set_with_face_crop(what, data_path_source, data_path_dest, max_samples_per_class=1,
                                          max_classes=None, target_size=(224, 224), mini_res=False):
    '''
    Browse full train AND test folders, copies samples from each class and move them to the dest train folder.

    Arguments:
        what: "train" or "test".
        data_path_source: the folder where the 'train' and 'test' folders are (FULL SET)
        data_path_destination: the folder where the WHAT folders are (DESTINATION)
        ratio: fraction of sample extracted from full set and moved to the train folder

    Outputs:
        A train folder in the data_path_dest.
    '''
    train_folder_source = os.path.join(data_path_source, 'train')
    test_folder_source = os.path.join(data_path_source, 'test')
    folder_dest = os.path.join(data_path_dest, what)
    count_classes = 0
    other = ["train", "test"]
    other.remove(what)
    other = other[0]
    other_folder_dest = os.path.join(data_path_dest, other)
    total_count = 0
    incomplete_classes = 0
    gc.collect()

    # prepare the face detector
    detector = MTCNN()

    for source in [train_folder_source, test_folder_source]:
        if count_classes == max_classes:
            break
        for subf in os.listdir(source):
            if count_classes == max_classes:
                break
            count_classes += 1
            imdir = os.listdir(os.path.join(source, subf))
            random.shuffle(imdir)
            nb_samples = max_samples_per_class
            newimdir = imdir
            os.makedirs(os.path.join(folder_dest, subf), exist_ok=True)
            # Count files already in destination
            count = len(os.listdir(os.path.join(folder_dest, subf)))
            print("Need to transfer {} samples.".format(max(0, nb_samples - count)))
            print(os.path.join(folder_dest, subf))
            if count >= nb_samples:
                total_count += count
                continue
            attempts = 0
            native_res_factor = 1
            cropped_res_factor = 1
            confidence_factor = 1
            while True:
                for im in newimdir:
                    if not os.path.exists(os.path.join(folder_dest, subf, im)) and not os.path.exists(
                            os.path.join(other_folder_dest, subf, im)):
                        # impath=os.path.join(folder_dest, subf, im)
                        # shutil.copy(os.path.join(source, subf, im), impath)
                        impath = os.path.join(source, subf, im)
                        impath_target = os.path.join(folder_dest, subf, im)
                        try:
                            # img = cv2.imread(impath)
                            img = pyplot.imread(impath)
                            full_height, full_width, channels = img.shape
                            print('==================', full_height, full_width)
                            if mini_res and min(full_height, full_width) < mini_res * 2.2 * native_res_factor:
                                # print("NATIVE RESOLUTION TOO LOW: skipping image.")
                                continue
                            detections = detector.detect_faces(img)
                            print("Detections:", detections)
                            if detections == []:
                                print("NO FACE DETECTED: skipping image {}/{}.".format(subf, im))
                                continue

                            if detections[0]['confidence'] < 0.99 * confidence_factor:
                                print("CONFIDENCE TOO LOW: skipping image.")
                                continue
                            x1, y1, width, height = detections[0]['box']
                            current_res = min(width, height)
                            print("Image resolution:", current_res)
                            if mini_res and current_res < mini_res * cropped_res_factor:
                                print("CROPPED RESOLUTION TOO LOW: skipping image.")
                                continue
                            # Make image square
                            w2 = width // 2
                            xc = x1 + w2  # X centroid
                            h2 = height // 2
                            yc = y1 + h2  # Y centroid

                            d = max(height, width) // 2
                            print(yc - d, yc + d, xc - d, xc + d)
                            Y0, Y1, X0, X1 = yc - d, yc + d, xc - d, xc + d

                            # Check that nothing is outside the frame
                            check = all([
                                Y0 >= 0,
                                X0 >= 0,
                                Y1 <= full_height,
                                X1 <= full_width,
                            ])

                            if not check:
                                print("FACE PARTIALLY SHOWN ON ORIGINAL IMAGE (TOO ZOOMED IN): skipping image.")
                                continue

                            face = img[Y0:Y1, X0:X1, ::-1]  # <--- Invert 1st and last channel

                            # resize pixels to the model size
                            face = PIL.Image.fromarray(face, mode='RGB')
                            face = face.resize(target_size, Image.BICUBIC)
                            # Remove original image
                            # os.remove(impath)

                            # save croppped image
                            face.save(impath_target, "JPEG", icc_profile=face.info.get('icc_profile'))
                            del face
                            count += 1
                            total_count += 1

                            if count == nb_samples: break
                        except:
                            # delete image and try with another image
                            # os.remove(impath)
                            continue

                if count == nb_samples:
                    print("******************************************")
                    print("Total Count: {} || Completion: {:.2%}.".format(total_count, total_count / (
                            max_classes * max_samples_per_class)))
                    print("******************************************")
                    break

                attempts += 1
                if attempts == 1:
                    native_res_factor = 1.05 / 2.2
                    confidence_factor = 0.97 / 0.99
                    print("******************************************")
                    print("******************************************")
                    print("**WARNING : CHECKING MORE IMAGES *********")
                    print("******************************************")
                    print("******************************************")
                if attempts == 2:
                    native_res_factor = 0.9 / 2.2
                    cropped_res_factor = 0.75
                    confidence_factor = 0.95 / 0.99
                    print("*********************************************")
                    print("********************************************")
                    print("**WARNING : LOWER RESOLUTION MODE ENABLED **")
                    print("********************************************")
                    print("********************************************")
                if attempts == 3:
                    native_res_factor = 0.60 / 2.2
                    cropped_res_factor = 0.50
                    confidence_factor = 0.90 / 0.99
                    print("***********************************************")
                    print("***********************************************")
                    print("**WARNING : ULTRALOW RESOLUTION MODE ENABLED **")
                    print("***********************************************")
                    print("***********************************************")
                if attempts == 4:
                    native_res_factor = 0
                    cropped_res_factor = 0.30
                    confidence_factor = 0.80 / 0.99
                    print("***********************************************")
                    print("***********************************************")
                    print("**WARNING : LAST CHANCE ROUND !!!!!!!!!!!!!! **")
                    print("***********************************************")
                    print("***********************************************")
                if attempts == 5:
                    print("Not enough good quality images.")
                    incomplete_classes += 1
                    break

    print("Incomplete clases:", str(incomplete_classes))
    print('Done.')

def get_weight_path(model_name):
    '''
    Return the path where model weights are stored.

    Arguments:
        model_name: name of the model used

    Output:
        full path containing the model weights
        '''

    i=models_catalog.index(model_name)
    path=models_weight_paths[i]
    if not os.path.exists(path):
        path="."+path

    return path

def populate_classes(data_path, model_name, dataset_tag):
    '''
    Obtain the list of classes and saves it as .npy for later use on AWS.

    Arguments:
        data_path: the folder where the 'train' and 'test' folders are.
        model_name: string corresponding to the model.
        dataset_tag: name of the current dataset.

    Outputs:
        Two .npy files written to the disk.
    '''

    global classes
    global classes_mapping

    classes_path=os.path.join('../models', 'bottlenecks',  model_name + dataset_tag + '_CLASSES.npy')
    classes_path_mapping=os.path.join('../models', 'bottlenecks', model_name + dataset_tag + '_CLASSES_MAPPING.npy')

    if os.path.exists(classes_path) and os.path.exists(classes_path_mapping):
        classes = np.load(classes_path, allow_pickle=True)
        classes_mapping = np.load(classes_path_mapping, allow_pickle=True)
    else:
        # Identify labels based on the available folders in the train and test set
        train_test_labels=get_labels(data_path)
        classes = train_test_labels[0]
        classes_mapping = dict(enumerate(classes))

        # Save classes for AWS use
        np.save(classes_path, classes)
        np.save(classes_path_mapping, classes_mapping)

def get_labels(data_path, mapfile="identity_meta.csv"):
    '''
    Obtains the match between the folder in the train/ test folders and their labels.

    Arguments:
        data_path: the folder where the 'train' and 'test' folders are
        mapfile: the name of the file with the mapping, located in data_path

    Outputs:
        res = [npar11, nparr2]: a list of two np.arrays with the labels for the train and test sets.
    '''
    res=[]
    mapfile_path=os.path.join(data_path, mapfile)

    for folder in ['train', 'test']:
        path=os.path.join(data_path, folder)
        ldir=np.array(os.listdir(path))
        df_mapfile=pd.read_csv(mapfile_path, sep=', ', engine='python', encoding='utf8') # <--- This is because some names have a "," in them
        res.append(np.array(df_mapfile.loc[df_mapfile['Class_ID'].map(lambda x: x in ldir),'Name'].tolist()))

    return res

def get_base_model(model_name, from_notebook=True):
    '''
    Create the base model object used for transfer learning.

    Arguments:
        model_name: string corresponding to the model
        weight_path: location of the base model weights

    Outputs:
        base model with preset weights
    '''
    if from_notebook:
        relative = ".."
    else:
        relative = "."

    weight_path = get_weight_path(model_name)
    # print('* Loading model weights from: '+path+'...')
    if model_name!="FaceNet":
        model = models_objects[models_catalog.index(model_name)]
    else:
        model_path = os.path.join(relative, "models", "weights", "facenet_keras.h5")
        print(model_path)
        model = load_model(model_path)

    # print('Base model input shape', IMAGE_WIDTH, IMAGE_HEIGHT)
    if model_name!="FaceNet":
        base_model = model(
            include_top=False,
            weights=weight_path,
            input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)
        )
    else:
        print('BBBBBBBBBBB')
        base_model = model

    return base_model

def build_network(input_shape, embeddingsize):
    '''
    Defines the neural network to refine image embeddings.

    Arguments:
            input_shape: shape of input images.
            embeddingsize: vector size used to encode our picture.

    Outpout:
            A model to output refined embeddings from a pre-encoded image.
    '''

    # Convolutional Neural Network
    network = Sequential(name="encoder")
    network.add(Flatten(input_shape=input_shape))
    network.add(Dropout(0.30))
    network.add(Dense(embeddingsize, activation=None,
                      kernel_initializer='he_uniform'))

    return network


def build_metric_network(single_embedding_shape):
    '''
    Defines the neural network to learn the metric (similarity function)

    Arguments:
            single_embedding_shape : shape of input embeddings or feature map.
                                    Must be an array.

    Output:
            A model that takes a pair of two images emneddings, concatenated
            into a single array: concatenate(img1, img2) -> single probability.
    '''

    #compute shape for input
    input_shape = single_embedding_shape
    #the two input embeddings will be concatenated
    input_shape[0] = input_shape[0]*2

    # Neural Network
    network = Sequential(name="learned_metric")
    network.add(Dense(30, activation='relu',
                      input_shape=input_shape,
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer='he_uniform'))
    network.add(Dense(20, activation='relu',
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer='he_uniform'))
    network.add(Dense(10, activation='relu',
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer='he_uniform'))

    #Last layer : binary softmax
    network.add(Dense(2, activation='softmax'))

    #Select only one output value from the softmax
    network.add(Lambda(lambda x: x[:,0]))

    return network


def build_quad_model(input_shape, network, metricnetwork, margin, margin2):
    '''
    Define the Keras Model for training

    Arguments:
        input_shape: shape of input images.
        network: Neural network to train outputing embeddings.
        metricnetwork: Neural network to train the learned metric.
        margin: minimal distance between Anchor-Positive and Anchor-Negative
            for the lossfunction (alpha1).
        margin2: minimal distance between Anchor-Positive and
            Negative-Negative2 for the lossfunction (alpha2).

    Ouput:
        The complete quadruplet losss model.

    '''
    # Define the tensors for the four input images
    anchor_input = Input(input_shape, name="anchor_input")
    positive_input = Input(input_shape, name="positive_input")
    negative_input = Input(input_shape, name="negative_input")
    negative2_input = Input(input_shape, name="negative2_input")

    # Generate the encodings (feature vectors) for the four images
    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)
    encoded_n2 = network(negative2_input)

    #compute the concatenated pairs
    encoded_ap = Concatenate(axis=-1,name="Anchor-Positive")([encoded_a,encoded_p])
    encoded_an = Concatenate(axis=-1,name="Anchor-Negative")([encoded_a,encoded_n])
    encoded_nn = Concatenate(axis=-1,name="Negative-Negative2")([encoded_n,encoded_n2])

    #compute the distances AP, AN, NN
    ap_dist = metricnetwork(encoded_ap)
    an_dist = metricnetwork(encoded_an)
    nn_dist = metricnetwork(encoded_nn)

    #QuadrupletLoss Layer
    loss_layer = QuadrupletLossLayer(alpha=margin,beta=margin2,name='4xLoss')([ap_dist,an_dist,nn_dist])

    # Connect the inputs with the outputs
    network_train = Model(inputs=[anchor_input,positive_input,negative_input,negative2_input],outputs=loss_layer)

    # return the model
    return network_train


def train_quad_siamese(data_dir, test=False, from_notebook=False, verbose=True, \
                       realistic_method="average"):
    '''
    Train model and save weights.

    Arguments:
        data_dir: location of the train and test folders
        test: if test, only use 1 epoch and save results in a special directory
        from_notebook: use to update relative paths
        verbose: print progress and step completions
        realistic_method: "max" or "average". Decide how the predicted class
            will be computed, by either selecting the class corresponding to
            the image with the smallest distance ("min"), or by selecting the
            class whose top 3 matches have the smallest average ("average").

    External Arguments:
        IMAGE_WIDTH: image width (used for input in base model)
        IMAGE_HEIGHT: image height (used for input in base model)
        BATCH_SIZE: batch size used for training
        MODEL_BASE_TAG: name of the base model (used to retrieve weights and
            generate bottleneck features)
        EPOCHS: number of epochs

    Outputs:
        checkpoints: folder containing best model weights
        history.csv: history of model training
        history.png: plots of accuracy, loss, and learning rates
        summary.txt: model structure
    '''

    ## start timer for runtime
    time_start = time.time()
    print('\n\n******************* {} *******************\n'.format(MODEL_BASE_TAG))
    ## create location for train and test directory
    if not os.path.exists(data_dir):
        raise Exception("specified data directory does not exist.")
    if not os.path.exists(os.path.join(data_dir,'train')):
        raise Exception("training directory does not exist.")
    if not os.path.exists(os.path.join(data_dir,'test')):
        raise Exception("specified test directory does not exist.")

    # Get the specific name of the dataset
    dataset_tag = data_dir.split('\\')[-1]

    ## adjust relative path
    if from_notebook:
        base_weight_path = '.' + get_weight_path(MODEL_BASE_TAG)
        relative = ".."
    else:
        base_weight_path = get_weight_path(MODEL_BASE_TAG)
        relative = ".."

    populate_classes(data_dir, MODEL_BASE_TAG, dataset_tag)

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    ## initialize datagenerator
    datagen = ImageDataGenerator(rescale=1/255.)

    ## run generator
    if verbose: print("* Creating generators...")

    # Note: one subdirectory per class in the train/test folder.
    print('   > ', end='')
    generator_train = datagen.flow_from_directory(
        train_dir,
        color_mode="rgb",
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    print('   > ', end='')
    generator_test = datagen.flow_from_directory(
        test_dir,
        color_mode='rgb',
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # extract info from generator
    ## extract info from generator

    train_filenames=generator_train.filenames
    test_filenames=generator_test.filenames

    # Save train filename for identification use later on
    np.save(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_train_filenames.npy'), train_filenames)
    nb_train_samples = len(train_filenames)
    nb_test_samples = len(test_filenames)
    # print(generator_test.filenames) # ['n000001\\0001_01.jpg', 'n000001\\0002_01.jpg', 'n000001\\0003_01.jpg',
    num_classes = len(generator_train.class_indices)
    num_step_train = int(math.ceil(nb_train_samples / BATCH_SIZE))
    num_step_test = int(math.ceil(nb_test_samples / BATCH_SIZE))
    # num_classes = generator_train.classes_count

    ## create path for models if needed
    if not os.path.exists(os.path.join(relative,"models")):
        os.mkdir(os.path.join(relative,"models"))

    ## check if bottleneck weights exist
    if not os.path.exists(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_bottleneck_features_train.npy')):
        if verbose: print("* Creating bottleneck features...")
        base_model = get_base_model(MODEL_BASE_TAG)

        ## create bottle neck by passing the training data into the base model
        print('   > ', end='')
        bottleneck_features_train = base_model.predict(
            generator_train,
            steps=num_step_train,
            verbose=1
        )
        ## save bottleneck weights
        np.save(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_bottleneck_features_train.npy'), bottleneck_features_train)
        del bottleneck_features_train

        ## create bottle neck by passing the training data into the base model
        print('   > ', end='')
        bottleneck_features_test = base_model.predict(
            generator_test,
            steps=num_step_test,
            verbose=1
        )
        ## save bottleneck weights
        np.save(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_bottleneck_features_test.npy'), bottleneck_features_test)
        del bottleneck_features_test
        #_gc.collect()

    ## load the bottleneck features saved earlier
    if verbose: print("* Loading bottleneck features...")
    # print(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_bottleneck_features_train.npy'))
    #_gc.collect()
    train_data = np.load(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_bottleneck_features_train.npy'))
    #_gc.collect()
    test_data = np.load(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_bottleneck_features_test.npy'))


    ## get the class labels for the training data, in the original order
    train_labels = generator_train.classes
    test_labels = generator_test.classes

    print("Train_data shape: {}, train_labels shape: {}.".format(train_data.shape, train_labels.shape))

    ## Shuffle train and test set - NOT USED
    # np.random.seed(1986)
    orders = np.arange(train_data.shape[0])
    # np.random.shuffle(orders)
    # train_data = train_data[orders]
    # train_labels = train_labels[orders]
    ORDERS=orders


    ## Save ORDERS for predictions later
    np.save(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_orders.npy'), ORDERS)

    if verbose: print("   > Training data shape", train_data.shape)
    if verbose: print("   > Testing data shape", test_data.shape)
    if verbose: print("   > Training label data shape", train_labels.shape)
    if verbose: print("   > Testing label data shape", test_labels.shape)

    ## convert the training labels to categorical vectors
    if verbose: print("* Encoding labels...")
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    test_labels = to_categorical(test_labels, num_classes=num_classes)

    # Save classes
    if verbose: print("* Saving classes as sparse matrices...")

    if not os.path.exists(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_train_labels_sparse.npz')) \
            or not os.path.exists(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_test_labels_sparse.npz')) \
            or not os.path.exists(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_train_labels_AWS.npz')):

        scipy.sparse.save_npz(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_train_labels_sparse.npz'), \
                              scipy.sparse.csc_matrix(train_labels))

        scipy.sparse.save_npz(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_test_labels_sparse.npz'), \
                              scipy.sparse.csc_matrix(test_labels))

        # Create a smaller train_label matrix for fast loading in AWS.
        train_labels_AWS = np.argmax(train_labels, axis=1)

        np.save(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_train_labels_AWS.npy'), \
                train_labels_AWS)

    ## build other CNNs
    if verbose: print("* Creating model...")

    network = build_network(input_shape=train_data.shape[1:], embeddingsize=EMBEDDINGSIZE)
    metricnetwork = build_metric_network(single_embedding_shape=[EMBEDDINGSIZE])
    model = build_quad_model(train_data.shape[1:], network, metricnetwork, margin=MARGIN, margin2=MARGIN2)

    ## save info about models
    # model_name = re.sub("\.","_",str(MODEL_VERSION))
    model_name = MODEL_VERSION

    ## create directory for version specific
    if not os.path.exists(os.path.join(relative,"models")):
        os.mkdir(os.path.join(relative,"models"))
    if not os.path.exists(os.path.join(relative,"models","runs",model_name)):
        os.mkdir(os.path.join(relative,"models","runs",model_name))
    if not os.path.exists(os.path.join(relative,"models","runs",model_name,"checkpoints")):
        os.mkdir(os.path.join(relative,"models","runs",model_name,"checkpoints"))

    ## OPTIMIZER
    #optimizer = optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999)
    optimizer = Adam(lr=START_LR)
    # optimizer = Adam(lr=0.005)

    ## compile model
    model.compile(
        optimizer=optimizer,
        loss=None
        # metrics=["accuracy"]
    )

    # Compute generators
    gentrain = generate_quad(train_data, train_labels, batch_size=BATCH_SIZE)
    # gentest = generate_quad(test_data, test_labels, batch_size=BATCH_SIZE)
    gentest=0 # genetest is deprecated.

    ## CALLBACKS
    ## progress
    if from_notebook:
        callbacks = []
        model_verbose = 2
    else:
        callbacks = []
        model_verbose = 2

    N, k = N_ONESHOTMETRICS, k_ONESHOTMETRICS #N_ONESHOTMETRICS not used at the moment
    callbacks.append(OneShotMetricsQuad(network, metricnetwork, N, k, gentest, \
                                        test_data, test_labels, train_data, train_labels, realistic_method=realistic_method))

    ## reduce LR
    lrate = ReduceLROnPlateau(
        monitor="loss",
        factor = 0.4,
        patience=1,
        verbose=1,
        min_lr = 0.0000001
    )
    callbacks.append(lrate)

    ## early stopping
    es = EarlyStopping(
        monitor='loss',
        mode='min',
        verbose=1,
        patience=10,
        min_delta=0.005
    )
    callbacks.append(es)

    ## save
    checkpoints = ModelCheckpoint(
        os.path.join(relative,"models","runs",model_name,"checkpoints",model_name+".h5"),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_freq='epoch',
        # period=1
    )
    # callbacks.append(checkpoints)

    ## compute step size and epoch count
    n_epochs = EPOCHS

    ## adjust parameters for test
    if test:
        n_epochs = 1

    ## save model summary
    if verbose: print("* Saving model summary...")
    with open(os.path.join(relative,"models","runs",model_name,"summary"+CUSTOM_FILE_NAME+".txt"),'w') as fh:
        # pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    if verbose: print("* Training model...")
    history = model.fit(gentrain,
                        # train_data,
                        # train_labels,
                        epochs=n_epochs,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        # batch_size=BATCH_SIZE,
                        verbose=1,
                        # validation_data=(test_data, test_labels),
                        callbacks=callbacks,
                        # class_weight=dict(enumerate(class_weights))
                        )

    ## save model
    if verbose: print("* Saving main model...")
    model.save(os.path.join(relative,"models","runs",model_name,'my_model_weights.h5'))

    if verbose: print("* Saving encoder model...")
    network.save(os.path.join(relative,"models","runs",model_name,'encoder_model_weights.h5'))

    if verbose: print("* Saving similarity model...")
    metricnetwork.save(os.path.join(relative,"models","runs",model_name,'similarity_model_weights.h5'))

    ## save history into a csv file
    if verbose: print("* Saving training history...")
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(relative,"models","runs",model_name,"history.csv"))

    ## create plots for learning rate, accuracy, and loss
    save_path = os.path.join(relative,"models","runs",model_name)
    plot_history_quad(history_df, save_path, custom_file_name=CUSTOM_FILE_NAME)

    ## compute running time
    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## All finished
    if verbose: print("* Done.")


source=r"./dataset"
dest='./some'
## Generate train set
get_what_from_full_set_with_face_crop("train", source, dest, max_samples_per_class=10, max_classes=1, mini_res=160)