import os
from os.path import sep
from collections import Counter

import numpy as np
import tensorflow as tf
import tarfile
import fnmatch


def binarise(y_pred):
    threshold, upper, lower = 0.5, 1, 0
    y_pred[y_pred >= threshold] = 1.
    y_pred[y_pred < threshold] = 0.
    return y_pred


def extract_file(cls, dir: str, ext: str = "*.tar"):
    tar_files = [f for f in os.listdir(dir) if fnmatch.fnmatch(f, ext)]
    for file in tar_files:
        task_path = f"{dir}{sep}{file}"
        with tarfile.open(task_path, 'r') as tar:
            tar.extractall(f"{dir}{sep}")

def get_max_occurence_value(x: list):
    inv_map = {v: k for k, v in Counter(x).items()}
    return inv_map[max(inv_map.keys())]
