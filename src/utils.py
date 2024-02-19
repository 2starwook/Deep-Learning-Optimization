import os
from os.path import sep
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
