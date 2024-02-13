import os
import fnmatch

import numpy as np
import nibabel as nib
from collections import Counter
from sklearn.preprocessing import normalize as sklearn_normalize
from skimage.morphology import binary_opening
from matplotlib import pyplot as plt
from keras.models import *
from keras.layers import *
from keras.optimizers import *


PATH = "./"
PATH_DATA = f"{PATH}Data/MSD/"

class DatasetLoader:
    def __init__(self, task) :
        """
        Init MedicalDatasetPipeline for one task
        """
        self.task           = task
        self.DIR            = PATH_DATA
        self.DIR_TRAIN      = "/imagesTr/"
        self.DIR_TEST       = "/imagesTs/"
        self.DIR_GT         = "/labelsTr/"
        self.DIR_GT_TEST    = "/labelsTs/"
        self.DATA_TRAIN_DIR = self.DIR + self.task + self.DIR_TRAIN
        self.DATA_GT_DIR    = self.DIR + self.task + self.DIR_GT
        self.DATA_TEST_DIR  = self.DIR + self.task + self.DIR_TEST
        self.DATA_PRED_DIR  = self.DIR + self.task + self.DIR_GT_TEST
        self.img_shape      = None
        self.train_set: np.array
        self.gt_set: np.array
        self.test_set: np.array
        self.__get_train_set()
        self.__get_test_set()

    def __get_train_set(self) :
        """
        Load training file and groundtruth file
        """
        files          = [f for f in os.listdir(self.DATA_TRAIN_DIR) if fnmatch.fnmatch(f, "[!.]*.nii.gz")]
        files_gt       = [f for f in os.listdir(self.DATA_GT_DIR) if fnmatch.fnmatch(f, "[!.]*.nii.gz")]
        n_train_sample = len(files)
        n_gt_sample    = len(files_gt)
        self.train_set      = np.empty(n_train_sample, dtype=object)
        self.gt_set         = np.empty(n_gt_sample, dtype=object)

        for i in range(n_train_sample):
            img = nib.load(self.DATA_TRAIN_DIR + files[i]).get_fdata()
            self.train_set[i] = img

        for i in range(n_gt_sample):
            img = nib.load(self.DATA_GT_DIR + files[i]).get_fdata()
            self.gt_set[i] = img

    def __get_test_set(self):
        """
        Load test file
        """    
        files_test     = [f for f in os.listdir(self.DATA_TEST_DIR) if fnmatch.fnmatch(f, "[!.]*.nii.gz")]
        n_test_sample  = len(files_test)
        self.test_set  = np.empty(n_test_sample, dtype=object)

        for i in range(n_test_sample): 
            img = nib.load(self.DATA_TEST_DIR + files_test[i]).get_fdata()
            self.test_set[i] = img

    def display_train_set(self, nb_patient=10, slice_index=30):
        fig = plt.figure(figsize= (7, 50), dpi = 90)

        k = 0
        for i in range(nb_patient):

            plt.subplot(10, 2, k + 1)
            plt.imshow(self.train_set[i][:, :, slice_index])
            plt.subplots_adjust(wspace = 0)
            plt.title("original")
            plt.axis('off')

            k += 1 
            
            mask = self.gt_set[i][:, :, slice_index] == 1
            tmp = self.train_set[i][:, :, slice_index]
            tmp[mask] = 3000

            plt.subplot(10, 2, k + 1)
            plt.imshow(tmp)
            plt.title("GT")
            plt.subplots_adjust(wspace = 0)
            plt.axis('off')

            k += 1

    def display_test_set(self, nb_patient=10, slice_index=30):
        fig = plt.figure(figsize= (7, 50), dpi = 90)

        k = 0
        for i in range(nb_patient):

            plt.subplot(10, 2, k + 1)
            plt.imshow(self.test_set[i][:,:,slice_index])
            plt.subplots_adjust(wspace = 0)
            plt.title(f"{i+1}")
            plt.axis('off')

            k += 1 

    def get_max_occurence_value(self, x):
        inv_map = {v: k for k, v in Counter(x).items()}
        return inv_map[max(inv_map.keys())]

    def get_img_shape(self):
        size_list_x = []
        size_list_y = []
        
        for patient_data in self.train_set:
            size_list_x.append(patient_data.shape[0])
            size_list_y.append(patient_data.shape[1])
        
        self.img_shape = (self.get_max_occurence_value(size_list_x),
                        self.get_max_occurence_value(size_list_y))
        return self.img_shape

    def normalize(self, x):
        for i in range(x.shape[0]):
            x[i] = sklearn_normalize(x[i], norm='max', copy=True, return_norm=False)
        return x

    def get_x_train(self):
        """
        Compute x_train from training set
        """
        self.x_train_len = 0
        for patient_data in self.train_set:
            self.x_train_len += patient_data.shape[2]

        """
        Get image shape, choosing the most represented one from training set
        """   
        if self.img_shape is None:
            self.get_img_shape()
        
        self.x_train = [] #(self.x_train_len, dtype=object)

        i = 0
        for patient_data in self.train_set:
            for j in range(patient_data.shape[2]):
                if patient_data.T[j].shape == self.img_shape:
                    self.x_train.append(patient_data.T[j])
                else:
                    """
                    Resize slice if not the right shape
                    """
                    self.x_train.append(np.resize(patient_data.T[j], self.img_shape))

        self.x_train = np.array(self.x_train)
        
        """
        Normalize
        """
        self.x_train = self.normalize(self.x_train)

        return self.x_train

    def get_y_train(self):
        """
        Compute y_train from groundtruth set
        """
        self.y_train_len = 0
        for patient_data in self.gt_set:
            self.y_train_len += patient_data.shape[2]

        """
        Get image shape, choosing the most represented one from training set
        """   
        if self.img_shape is None:
            self.get_img_shape()
        
        self.y_train = []

        i = 0
        for patient_data in self.gt_set:
            for j in range(patient_data.shape[2]):
                if patient_data.T[j].shape == self.img_shape:
                    self.y_train.append(patient_data.T[j])
                else :
                    """
                    Resize slice if not the right shape
                    """
                    self.y_train.append(np.resize(patient_data.T[j], self.img_shape))

        self.y_train = np.array(self.y_train)
        
        """
        Normalize
        """
        self.y_train = self.normalize(self.y_train)

        return self.y_train

    def get_x_test(self):
        """
        Compute x_test from test set
        """
        self.x_test_len = 0
        for patient_data in self.test_set:
            self.x_test_len += patient_data.shape[2]

        """
        Get image shape, choosing the most represented one from training set
        """   
        if self.img_shape is None:
            self.get_img_shape()
        self.x_test = []
        self.x_test_size = []
        for patient_data in self.test_set:
            self.x_test_size.append(patient_data.shape[2])
            for j in range(patient_data.shape[2]):
                if patient_data.T[j].shape == self.img_shape:
                    self.x_test.append(patient_data.T[j])
                else :
                    """
                    Resize slice if not the right shape
                    """
                    self.x_test.append(np.resize(patient_data.T[j], self.img_shape))
        
        self.x_test = np.array(self.x_test)
        
        """
        Normalize
        """
        self.x_test = self.normalize(self.x_test)
        
        return self.x_test

    def postprocess(self, x, treshold = 0.5) : 
        for i in range(x.shape[0]):
            x[x >= treshold] = 1
            x[x < treshold] = 0
            x[i] = binary_opening(x[i , :, :] == 1)
        return x

    def save_predictions(self, y_pred) :
        """
        Save predictions
        """
        i = 0
        k = 0
        for patient_data in self.test_set :
            y_pred_resize = []
            for j in range(patient_data.shape[2]) :
                if patient_data.T[j].shape == self.img_shape :
                    y_pred_resize.append(y_pred[i])
                else :
                    """
                    Resize slice if not the right shape
                    """
                    y_pred_resize.append(np.resize(y_pred[i], patient_data.T[j].shape))
                i+=1
            y_pred_resize = np.array(y_pred_resize)
            tmp = y_pred_resize.astype(np.uint8)
            img = nib.Nifti1Image(tmp, np.eye(4))
            nib.save(img, self.DIR_GT_TEST + self.files_test[k])
            k+=1
