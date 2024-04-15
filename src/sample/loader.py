import os
import fnmatch

import numpy as np
from numpy.typing import NDArray
import nibabel as nib
from sklearn.preprocessing import normalize as sklearn_normalize
from skimage.morphology import binary_opening
from matplotlib import pyplot as plt
from keras.models import *
from keras.layers import *
from keras.optimizers import *

from src.utils import get_max_occurence_value
from sample.dataset import Dataset


class DatasetLoader:
    def __init__(self, train_dir: str, train_gt_dir: str, 
                 test_dir: str, test_gt_dir: str = ""):
        """
        Init MedicalDatasetPipeline for one task
        """
        self.train_set = DatasetLoader.__get_raw_dataset(train_dir)
        self.gt_set = DatasetLoader.__get_raw_dataset(train_gt_dir)
        self.test_set = DatasetLoader.__get_raw_dataset(test_dir)
        self.img_shape = DatasetLoader.__get_img_shape(self.train_set)

    def __get_dataset(self, raw_dataset: NDArray) -> Dataset:
        lengths = list()
        for data in raw_dataset:
            lengths.append(data.shape[2])
        
        dataset = [] #(self.x_train_len, dtype=object)
        for data in raw_dataset:
            for i in range(data.shape[2]):
                if data.T[i].shape == self.img_shape:
                    dataset.append(data.T[i])
                else:
                    """
                    Resize slice if not the right shape
                    """
                    dataset.append(np.resize(data.T[i], self.img_shape))

        dataset = np.array(dataset)
        dataset = DatasetLoader.__normalize(dataset)

        return Dataset(dataset, self.img_shape, lengths)

    @classmethod
    def __get_raw_dataset(cls, dir: str, ext: str = "[!.]*.nii.gz") -> NDArray:
        files = [f for f in os.listdir(dir) if fnmatch.fnmatch(f, ext)]
        n = len(files)
        res = np.empty(n, dtype=object)
        for i in range(n):
            img = nib.load(f"{dir}{files[i]}").get_fdata()
            res[i] = img
        return res

    @classmethod
    def __get_img_shape(cls, dataset: NDArray) -> tuple:
        size_list_x = []
        size_list_y = []
        
        for patient_data in dataset:
            size_list_x.append(patient_data.shape[0])
            size_list_y.append(patient_data.shape[1])
        
        img_shape = (get_max_occurence_value(size_list_x), 
                     get_max_occurence_value(size_list_y))
        return img_shape

    @classmethod
    def __normalize(cls, x: NDArray):
        for i in range(x.shape[0]):
            x[i] = sklearn_normalize(x[i], norm='max', copy=True, return_norm=False)
        return x

    @classmethod
    def postprocess(cls, x, treshold = 0.5) : 
        for i in range(x.shape[0]):
            x[x >= treshold] = 1
            x[x < treshold] = 0
            x[i] = binary_opening(x[i , :, :] == 1)
        return x

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

    def get_train_dataset(self):
        """
        Compute x_train from training set
        """
        return self.__get_dataset(self.train_set)

    def get_train_gt_dataset(self):
        """
        Compute y_train from groundtruth set
        """
        return self.__get_dataset(self.gt_set)

    def get_test_dataset(self):
        """
        Compute x_test from test set
        """
        return self.__get_dataset(self.test_set)

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
            DIR_GT_TEST    = "labelsTs"
            nib.save(img, DIR_GT_TEST + self.files_test[k])
            k+=1
