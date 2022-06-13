# IMPORTS

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
import numpy.random as rng
from mtcnn.mtcnn import MTCNN
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.losses import BinaryCrossentropy
import os, gc, sys
sys.path.append('..')
from PIL import Image
import matplotlib.pyplot as plt
retval = os.getcwd()
print("[>>SLD MODEL<<] Current working directory %s" % retval, \
      "__NAME__ = ", __name__)
############################### USER INPUTS ##################################
local=True            # True if running locally
memory_issues=False   # True if GPU memory issues ("failed to allocate" / OOM)
disable_gpu=False     # True if above setting is not sufficient

## Generate train/test set
generate_train_test_sets=True  # True to gen train/test set, run only once

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
N_ONESHOTMETRICS = 3 # Parameter not used right now
IMAGE_WIDTH, IMAGE_HEIGHT = (160, 160) if MODEL_BASE_TAG=="FaceNet" else (224,224)
CUSTOM_FILE_NAME += \
    "_B"+str(BATCH_SIZE) + \
    "_E"+str(EPOCHS) + \
    "_S"+str(STEPS_PER_EPOCH) + \
    "_k"+str(k_ONESHOTMETRICS) + \
    "_lr"+str(START_LR) + \
    "_M"+str(MARGIN) + \
    "_MM"+ str(MARGIN2) + \
    "_em"+str(EMBEDDINGSIZE)
MODEL_VERSION = \
    MODEL_BASE_TAG+dataset_tag
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
# Set parameters
method="average"
model_name="FaceNet"
from_notebook=True # <<<-------------  TO CHANGE FOR AWS
online=False if __name__ == "__main__" else True
object_sizes=False        # Output the size of preloaded objects

# Chose a test image below (uncomment one)

# img_test = "sm.jpg"      # sophie marceau
# img_test = "da.jpg"      # david hasselhoff
# img_test = "eg.jpg"      # elodie gaussuin
# img_test = "yn.jpg"      # yannick noah
# img_test = "ab.jpg"      # a bouteflika (from train set!)
img_test = "1.jpg"    # ines de la fressange
# img_test = "aff.jpg"     # a fine Fenzy from _litecropped (train set!)
# img_test = "ar.jpg"      # a raja _litecropped (train set!)

########################## END OF USER INPUTS ################################
class Match:
    '''
    A simple class to store matches from the algorithm prediction resulting
    from a test image.
    Contains the name, path and distance.
    Data is used by the deployed app (in particular the carousel).

    Arguments:
        path: path of the image file.
        name: name of the image.
        distance: float, distance (similarity) betwwen the image and its match.

    Output:
        A Match object containing all arguments.
    '''


    def __init__(self, path=None, name=None, distance=None):
        self.path=path
        self.name=name
        self.distance=distance

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
        relative = "."
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

def load_models_and_data(model_name, dataset_tag, from_notebook=True, online=False):
    '''
    Load models (base, encoder and similarity), the train data and the labels,
        the map between the labels and the actuals names and the image input size.

    Arguments:
        model_name: string (FaceNet, ResNet50, InceptionV3, MobileNetV2, Xception, VGG16).
        dataset_tag: string to represent the dataset used.
        from_notebook: use True to offset relative paths.
        online: use True when deployed on AWS.

    Outputs:
        base_model: the pre-encoder model.
        network: the encoder model.
        metricnetwork: the similarity function.
        data_path: the path to the train and test set.
        train_data: an np.array containing the train set data
        train_labels_AWS: an np.array containg the train labels, optimized for AWS.
        train_filenames: the filenames corresponding to the train data.
        classes: the names of all the classes.
        target_size: the image size required for the pre-encoder.
    '''

    # Adjust relative path
    if from_notebook:
        relative = "."
    else:
        relative = "."

    if not online:
        data_path=os.path.join("./some", dataset_tag)
    else:
        data_path=os.path.join(r".\app\static\datasets", dataset_tag)

    # Get path and load base model
    try:
        base_model = get_base_model(model_name, from_notebook)
    except:
        raise Exception("No base model weights could be located")

    # Get paths and load encoder and similarity models
    network_path=os.path.join(relative, "models", "runs", model_name + dataset_tag, "encoder_model_weights.h5")
    metricnetwork_path=os.path.join(relative, "models", "runs", model_name + dataset_tag, "similarity_model_weights.h5")

    ## Check if models exists
    if not os.path.isfile(network_path):
        raise Exception("No encoder model in specified directory {}".format(network_path))

    if not os.path.isfile(metricnetwork_path):
        raise Exception("No encoder model in specified directory {}".format(metricnetwork_path))

    ## Load models
    network=load_model(network_path)
    metricnetwork=load_model(metricnetwork_path)

    # Load train data and labels
    try:
        print(os.path.join(relative, "models", 'bottlenecks', model_name + './' + 'some' + '/' + dataset_tag + '_bottleneck_features_train.npy'))
        print(os.path.join(relative, 'models', 'bottlenecks', model_name + './' + 'some' + '/' + dataset_tag + '_train_labels_AWS.npy'))
        gc.collect()
        train_data = np.load(os.path.join(relative, "models", 'bottlenecks', model_name + './' + 'some' + '/' + dataset_tag + '_bottleneck_features_train.npy'))
        gc.collect()
        test_data = np.load(os.path.join(relative, "models", 'bottlenecks', model_name + './' + 'some' + '/' + dataset_tag + '_bottleneck_features_test.npy'))
        gc.collect()
        test_labels = np.array(scipy.sparse.load_npz(os.path.join(relative, "models", 'bottlenecks', model_name + './' + 'some' + '/' + dataset_tag + '_test_labels_sparse.npz')).todense())
        train_labels_AWS = np.load(os.path.join(relative, 'models', 'bottlenecks', model_name + './' + 'some' + '/' + dataset_tag + '_train_labels_AWS.npy'))
    except:
        raise Exception('Train data or labels cannot be retrieved from given folders.')

    # Load mapping between the label columns and the label names
    # orders = np.load(os.path.join(relative, "models", 'bottlenecks', model_name + dataset_tag +  '_orders.npy'))

    # Set the target image size for the base model
    target_size=(160,160) if model_name == 'FaceNet' else (224,224)

    # Load train set filenames
    train_filenames=np.load(os.path.join(relative, "models", 'bottlenecks', model_name + './' + 'some' + '/' + dataset_tag + '_train_filenames.npy'))

    # Load classes
    classes = np.load(os.path.join(relative, "models", 'bottlenecks', model_name + './' + 'some' + '/' + dataset_tag + '_CLASSES.npy'))

    return base_model, network, metricnetwork, data_path, train_data, train_labels_AWS, train_filenames, classes, target_size, test_data, test_labels

def make_oneshot_task_realistic(train_data, train_labels, test_data=None, \
                                test_labels=None, output_labels=0, predict=1, \
                                predict_model_name=None, image=None):
    '''
    Create batch of pairs, one sample from the test set versus ALL samples in ALL classes of train set.

    Arguments:
        train_data: train data (numpy array), the reference image will be compared
            to all images contained in that array
        train_labels: train labels (numpy array)
        test_data: test data (numpy array), the reference image will be drawn
            amongst images contained in that array, unless 'image' is provided.
        test_labels: test labels (numpy array)
        output_labels: option to print out labels when testing
        predict: 0 if the function is used for in-training testing,
                 1 if the function is used for predictions
        predict_model_name: if predict is 1, name of the 1st encoder. This is
            to avoid the use of too many global variables.
        image: a np array representation of an input image, when predict is 1

    Outputs:
        pairs, actual_categories_1 if predict = 1
        pairs, targets if predict = 0 and output_labels = 1
        pairs, targets, actual_categories_1 if predict = 0 and output_labels = 0
        with:
            pairs: list of two tensors, one for the tested image (repeated as
                needed), one for the train set.
            targets: when the class of the tested image is known, target is a
                vector containing 1 where the train set is of the same class,
                0 otherwise.
            actual_categories_1: returns classes for the train set.
    '''

    # Obtain the model name differently if running for prediction or not
    if predict:
        model_name = predict_model_name
    else:
        model_name = MODEL_BASE_TAG

    if len(train_labels.shape)==1:
        # AWS one dimension format
        n_classes=np.unique(train_labels).shape[0]
    else:
        # dense matrix format
        n_classes=train_labels.shape[1]

    if model_name == 'FaceNet':
        # print(train_data.shape)
        n_examples,features=train_data.shape
    else:
        n_examples,features,t,channels=train_data.shape

    # Select the category whose sample is going to be drawn from
    category = rng.randint(0,n_classes)

    # initialize 2 empty arrays for the input image batch
    if model_name == 'FaceNet':
        pairs=[np.zeros((n_examples, features)) for i in range(2)]
    else:
        pairs=[np.zeros((n_examples, features, t, channels)) for i in range(2)]

    # initialize vector for the targets
    targets=np.zeros((n_examples,))

    # Save actually categories for information
    actual_categories_0=np.zeros((n_examples,))
    actual_categories_1=np.zeros((n_examples,))

    # Targets are one for same class.
    if len(train_labels.shape)==1:
        # AWS one dimension format
        targets[train_labels==category]=1
    else:
        # dense matrix format
        targets[train_labels[:,category]==1]=1

    # Select a random test image from the selected category
    if not predict:
        if model_name == 'FaceNet':
            subset0_test = test_data[test_labels[:,category]==1,:]
        else:
            subset0_test = test_data[test_labels[:,category]==1,:,:,:]
        nb_available_samples0_test=subset0_test.shape[0]
        idx_1_test = rng.randint(0, nb_available_samples0_test)
        sample_image = subset0_test[idx_1_test]
    elif predict:
        sample_image=image

    if model_name == 'FaceNet':
        pairs[0][:,:] = sample_image
        actual_categories_0[:] = category
        # actual_id_0[:] = idx_1_test

        pairs[1][:,:] = train_data
        if len(train_labels.shape)==1:
            # AWS one dimension format
            actual_categories_1[:] = train_labels
        else:
            # dense matrix format
            actual_categories_1[:] = np.argmax(train_labels, axis=1)
        # actual_id_1[:] =
    else:
        pairs[0][:,:,:,:] = sample_image
        actual_categories_0[:] = category
        # actual_id_0[:] = idx_1_test

        pairs[1][:,:,:,:] = train_data
        if len(train_labels.shape)==1:
            # AWS one dimension format
            actual_categories_1[:] = train_labels
        else:
            # dense matrix format
            actual_categories_1[:] = np.argmax(train_labels, axis=1)
        # actual_id_1[:] =

    if predict:
        return pairs, actual_categories_1

    if output_labels==0:
        return pairs, targets
    elif output_labels==1:
        return pairs, targets, actual_categories_1


def compute_learned_dist_one_vs_all(network, metricnetwork, k, train_data, \
                                    train_labels, test_data=None, test_labels=None, output_labels=1, \
                                    also_get_loss=0, verbose = 1, label="realistic", method="max", \
                                    predict=0, predict_model_name=None, image=None):

    '''
    This function computes the distance (similarity) between one image, either
    randomly selected from train_data or provided using the'image' argument.

    Arguments:
        network: encoder network
        metricnetwork: similarity function
        k: number of tests, should be 1 if predict is 1
        train_data: train data (numpy array), the reference image will be compared
            to all images contained in that array
        train_labels: train labels (numpy array)
        test_data: test data (numpy array), the reference image will be drawn
            amongst images contained in that array
        test_labels: test labels (numpy array)
        output_labels: option to print out labels when testing
        also_get_loss: option to also compute a classic binary cross entripy loss
        verbose: option to print out details about the execution of the code
        label: name of the type of testing done. Only use for console prints.
        method: "max" or "average". Decide how the predicted class will be computed,
            by either selecting the class corresponding to the image with the smallest
            distance ("min"), or by selectong the class whose top 3 matches have the
            smallest average ("average").
        predict: 0 if the function is used for in-training testing,
            1 if the function is used for predictions
        predict_model_name: if predict is 1, name of the 1st encoder. This is
            to avoid the use of too many global variables.
        image: a np array representation of an input image, when predict is 1


    Outputs:
        if predict = 1:
            predicted_cat: the predicted classs
            distance: the corresponding distance
            actual_image_index: the index of the matchig image in the train data
            sorted_predicted_cats: an array of all the predicted_cats, sorted
                by best match to worse match
            sorted_distances: an array of all the predicted distances, sorted
                by best match to worse match
            sorted_actual_image_index: an array of all images indexes, sorted
                by best match to worse match
        if predict = 0:
                percent_correct: a custom metrics. For each one of the k tests,
                    the ground truth class is compared to its position in the
                    sorted predicted classes of the model. For instance, if the
                    grond truth class is 3, and the model predicts 2, 3, 4, 1,
                    the percent correct will be 75%. That number is averagged
                    over all k examples. It gives an idea if a model is improving
                    or not, as it's a more granular metric that the exact_match
                    one below.
                loss: binary cross entropy loss
                exact_matches: out of the k tests, the percentage of predictions
                    that were exact.

    '''

    if predict and k!=1:
        raise Exception("Cannot predict on more than one sample.")

    n_correct = 0
    if verbose:
        print("Evaluating model with ({}) 1 test sample vs. all train samples\
              using the {} method...".format(str(k), method))

    if also_get_loss:
        bce = BinaryCrossentropy()
        loss=0

    rk_pct_total=0

    if not predict:
        print("Rounds completed:", end="\n")

    for i in range(k):
        gc.collect()
        if predict:
            pairs, actual_categories = make_oneshot_task_realistic(train_data, \
                                                                   train_labels, output_labels=1, predict=1, \
                                                                   predict_model_name=predict_model_name, image=image)
        else:
            pairs, targets, actual_categories = make_oneshot_task_realistic( \
                train_data, train_labels, test_data, test_labels, \
                output_labels=1, predict=0)
        gc.collect()

        # Get embeddings for the test image
        test_image_embeddings = network.predict(np.expand_dims(pairs[0][0], axis=0))

        # Create an array to store all embeddings
        m = pairs[0].shape[0] # number of comparison to make
        embeddingsize = test_image_embeddings.shape[1]
        embeddings = np.zeros((m, embeddingsize*2))

        train_set_embeddings=network.predict(pairs[1])
        embeddings[:,embeddingsize:]=train_set_embeddings
        embeddings[:,:embeddingsize]=test_image_embeddings

        # Get distances
        distances = metricnetwork(embeddings)
        distances=np.array(distances)
        # print(type(distances))
        # print(distances.shape)
        last_correct=False
        del embeddings
        del pairs

        if method=="min":
            if not predict:
                if np.argmin(distances) in np.argwhere(targets == np.amax(targets)):
                    n_correct+=1
                    last_correct=True
            elif predict:
                arg_min_d=np.argmin(distances)
                print('arg_min_d', arg_min_d)
                predicted_cat=int(actual_categories[arg_min_d])
                print('actual_categories', actual_categories)
                distance=np.amin(distances)
                print('distance', distance)
                actual_image_index=arg_min_d # No need to invoke ORDERS, train not shuffled for predict

                # Rank all results
                sorted_actual_image_index = np.argsort(distances)
                print(type(sorted_actual_image_index))
                print(sorted_actual_image_index)
                sorted_distances = distances[sorted_actual_image_index]
                sorted_predicted_cats = actual_categories[sorted_actual_image_index].astype(int)

        elif method=="average":
            # Compute the average per class of the smallest 3 distances
            avg_per_class=np.zeros(len(np.unique(actual_categories)))
            unsorted_actual_image_index=np.zeros(len(np.unique(actual_categories)))
            print_i=0
            s_dist = np.argsort(distances) # <--- sort only one time the whole array
            for i in range(avg_per_class.shape[0]):
                mask=actual_categories==i
                sorted_absolute_arguments_this_class=s_dist[mask[s_dist]]
                unsorted_actual_image_index[i]=int(sorted_absolute_arguments_this_class[0])
                sorted_distances_this_class=distances[s_dist][mask[s_dist]]
                avg_per_class[i]=np.average(sorted_distances_this_class[:3])
                if print_i <= 30:
                    # print(mask)
                    # print(sorted_absolute_arguments_this_class)
                    # print(sorted_distances_this_class)
                    # print(avg_per_class[i])
                    print_i+=1
            print('avg_per_class', avg_per_class)
            sorted_predicted_cats = np.argsort(avg_per_class) # <--- categories where the average is the lowest
            sorted_actual_image_index=unsorted_actual_image_index[sorted_predicted_cats].astype(int) # <--- absolute index of the image with the lowest distance for a given class
            sorted_distances = avg_per_class[sorted_predicted_cats] # <---

            predicted_cat = int(np.argmin(avg_per_class))
            distance=np.min(distances[actual_categories==predicted_cat])
            if predict:
                actual_image_index = np.where(np.logical_and(actual_categories==predicted_cat, distances==distance))[0][0] # No need to invoke ORDERS, train not shuffled for predict
            if not predict:
                rk_array = avg_per_class.argsort()
                target_cat = int(actual_categories[np.argmax(targets)])
                rk_pct=100*np.where(rk_array==target_cat)[0][0]/avg_per_class.shape[0]
                rk_pct_total+=rk_pct
                print("Rank percentage =", round(100-rk_pct,2), end=" ")
                if predicted_cat == target_cat:
                    n_correct+=1
                    last_correct=True
                else: # not correct
                    pass #no further action needed

        else:
            raise Exception("Wrong selection technique.")

        if predict:
            print('SUMMARY OF PREDICTIONS')
            print("Predicted single cat:", predicted_cat, "Predicted single distance:", \
                  distance, "Predicted single index:", actual_image_index, \
                  "##############################", "Predicted categories:", \
                  sorted_predicted_cats, "Predicted distances:", sorted_distances, \
                  "Predicted indexes:",  sorted_actual_image_index, \
                  sep="\n")
            # NOTE:
            # In case of the AVERAGE technique:
            # distance is the minimum distance within the predicted class
            # sorted_distance is the sorted AVERAGE distance per class
            # Therefore, it is normal than distance !=sorted_distance[0]
            # return predicted_cat, distance, actual_image_index, sorted_predicted_cats, sorted_distances, sorted_actual_image_index

        if also_get_loss:
            print('AAAAAAAAA')
            probs=1-distances
            print('probs', probs)
            new_loss=bce(targets, probs).numpy()
            print('new_loss', new_loss)
            loss+=new_loss
            print('loss', loss)

        # del probs, targets, actual_categories

        #During testing, this allows to quickly see how accurate the model is.
        if last_correct:
            print("o")
        else:
            print("x")
    print(" ")

    exact_matches = (100.0 * n_correct / k)
    percent_correct = 100-rk_pct_total/k

    if verbose:
        if label:
            print("Got an average of {}% realistic exact matches one-shot learning accuracy on the {} set over {} repetitions.\n".format(exact_matches,label,k))
        else:
            print("Got an average of {}% realistic exact matches one-shot learning accuracy \n".format(exact_matches))

    if method=="average":
        print("The average scoring is {}% (0% is best, 100% is worst).".format(round(percent_correct,0)))

    if also_get_loss:
        loss=loss/k
        return percent_correct, loss, exact_matches
    else:
        return percent_correct

def display_image(img):
    '''
    Simple function that allows multiple images to be outputted locally.

    Arguments:
        img: Input image as array.

    Output:
        The image is plotted in the console.
    '''
    plt.figure()
    plt.imshow(img)
    plt.axis('off')


def make_prediction_quad(model_name, base_model, network, metricnetwork, image_path, \
                         train_data, train_labels, train_filenames, classes, data_path, \
                         target_size=(224,224), filename=None, method="average", \
                         extra_matches=None, online=False, test_data=None, test_labels=None):
    '''
    Arguments:
        model_name: name of the model, e.g. "FaceNet"
        base_model: the pre-encoder model.
        network: the encoder model.
        metricnetwork: the similarity function.
        image_path: path of the image to be tested.
        train_data: an np.array containing the train set data
        train_labels_AWS: an np.array containg the train labels, optimized for AWS.
        train_filenames: the filenames corresponding to the train data.
        classes: the names of all the classes.
        data_path: path to train and test sets.
        target_size: the image size required for the pre-encoder.
        filename: filename of the tested image.
        method: "max" or "average". Decide how the predicted class will be computed,
            by either selecting the class corresponding to the image with the smallest
            distance ("min"), or by selectong the class whose top 3 matches have the
            smallest average ("average").
        extra_matches: number of extra matches to return (by order of probability).
        online: use rue when deployed on AWS.

    Outputs:
        pred_class: predicted class of the tested image.
        distance: similarity between the tested image and the matched image.
        actual_image_index: index of thee match image in the dataset.
        actual_image_path: path of the match image.
        matches_list: list of Match objects.
    '''

    # Center/cropp image using MTCNN
    print("* Starting face detection...")
    detector = MTCNN()
    img = img = cv2.imread(image_path)
    full_height, full_width, _ = img.shape

    detections = detector.detect_faces(img)
    print(detections)
    x1, y1, width, height = detections[0]['box']
    # Make image square
    w2=width//2
    xc = x1+w2
    h2=height//2
    yc = y1+h2

    d=max(height,width)//2
    print(yc-d,yc+d, xc-d,xc+d)
    Y0, Y1, X0, X1 = yc-d, yc+d, xc-d, xc+d

    # Check that nothing is outside the frame
    check = all([
        Y0>=0,
        X0>=0,
        Y1<=full_height,
        X1<=full_width,
        ])

    if not check:
        print("FACE PARTIALLY SHOWN ON ORIGINAL IMAGE (TOO ZOOMED IN): making the image square as-is.")
        dmin=min(yc, full_height-yc, xc, full_width-xc)
        Y0, Y1, X0, X1 = yc-dmin, yc+dmin, xc-dmin, xc+dmin
        print("New coords:", Y0, Y1, X0, X1)

    face = img[Y0:Y1, X0:X1, ::-1]

    # resize pixels to the model size
    face = PIL.Image.fromarray(face, mode='RGB')
    # if MODEL_BASE_TAG == 'FaceNet':
    #     target_size=(160,160)
    face = face.resize(target_size, Image.BICUBIC)
    face = np.asarray(face)

    display_image(face)
    image = np.asarray(face)

    # pre-process
    image = image / 255
    image = np.expand_dims(image, axis=0)

    # Get the embedding using the base model
    image=base_model.predict(image)

    pred_class, distance, actual_image_index, sorted_predicted_cats, \
    sorted_distances, sorted_actual_image_index = \
        compute_learned_dist_one_vs_all(network, metricnetwork, 1, train_data, \
                                        train_labels, test_data=test_data, test_labels=test_labels, output_labels=1, \
                                        also_get_loss=0, verbose = 1, label="realistic", method=method, \
                                        predict=0, predict_model_name=model_name, image=image)

    print("[xxxx] Current working directory %s" % retval, \
          "__NAME__ = ", __name__)
    actual_image_path=os.path.normpath(train_filenames[actual_image_index])
    print("actual_image_path", actual_image_path)
    full_actual_image_path=os.path.join(data_path, "train", actual_image_path).replace('\\','/')
    print("full_actual_image_path", full_actual_image_path)
    print("OPENING PREDICTED IMAGE")
    actual_image=Image.open(full_actual_image_path)

    if not online:
        display_image(actual_image)

    pred_class=int(pred_class)
    pred_name=classes[pred_class].replace("_"," ").replace('"', '')
    print("Predicted class:", pred_class, "- corresponding to:", pred_name)
    print("Distance:", distance)
    print("Path:", full_actual_image_path)

    # Save image as file if filenmae is provided
    if filename:
        if not online:
            plt.savefig(os.path.join(".", "app", "static", "prediction", "prediction_"+filename+".png"), dpi=150)
            # plt.savefig("./app/static/prediction/prediction_{}.png".format(filename), dpi=150)
        else:
            # actual_image.save("./app/static/prediction/prediction_{}.png".format(filename))
            actual_image.save(os.path.join(".", "app", "static", "prediction", "prediction_"+filename+".png"))

    if extra_matches:
        matches_list=[]
        for i in range(extra_matches+1): # We already have the first match
            if i<=len(sorted_actual_image_index):
                # path=os.path.normpath(train_filenames[sorted_actual_image_index[i]].replace('\\',"____")) # four underscore, will be split later in views:: display_train_sample
                print("normalized path", train_filenames[sorted_actual_image_index[i]])
                # path=retrieve_train_image_path(train_filenames[sorted_actual_image_index[i]], data_path)
                # convert path to html path
                # path=pathlib.Path(path).as_uri()

                # name=classes[sorted_predicted_cats[i]].replace("_"," ").replace('"', '')
                # distance=sorted_distances[i]
                # matches_list.append(Match(path, name, distance))
                # print("****")
                # print(path, name, distance)
                # image=Image.open(path)
                # display_image(image)

    if extra_matches:
        return  pred_class, distance, actual_image_index, actual_image_path, matches_list
    else:
        return  pred_class, distance, actual_image_index, actual_image_path


# LOAD EVERYTHING NEEDED TO RUN THE TEST
base_model, \
network, \
metricnetwork, \
data_path, \
train_data, \
train_labels_AWS, \
train_filenames, \
classes, \
target_size,\
test_data,\
test_labels = load_models_and_data(model_name, dataset_tag, from_notebook, online)


# OUTPUT SIZE OF OBJECTS FOR OPTIMIZATION
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

if object_sizes:
    for o in [base_model, network, metricnetwork, data_path, train_data, \
              train_labels_AWS ,train_filenames, classes, target_size]:
        print(namestr(o, globals()))
        print(round(sys.getsizeof(o)/1e6,0), "MB")


# DISPLAY INPUT IMAGE --> MAKE SURE TO TYPE %MATPLOTLIB INLINE FIRST
# img_test_path=os.path.join(r"..\app", "static", "examples", img_test)
image= Image.open(img_test)
plt.imshow(image)


# RUN FOLLOWING CODE IF GPU MEMORY ISSUES
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# PREDICT
gc.collect()
##############################################################################
pred_class, distance, actual_image_index, actual_image_path, matches_list = make_prediction_quad(
    model_name, base_model,
    network,
    metricnetwork,
    img_test,
    train_data,
    train_labels_AWS,
    train_filenames,
    classes,
    data_path,
    target_size,
    method="average",
    test_data=test_data,
    test_labels=test_labels)
##############################################################################

# END OF CODE