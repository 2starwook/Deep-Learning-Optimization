import os
from os.path import sep

class Config:
    PROJECT_DIR = "image_segmentation_optimization"
    DATA_DIR = "Data"
    MSD_DIR = "MSD"
    TASK_DIR = "Task02_Heart"
    DATA_TRAIN_DIR = "imagesTr"
    DATA_TEST_DIR = "imagesTs"
    DATA_TRAIN_GT_DIR = "labelsTr"
    DATA_TEST_GT_DIR = "labelsTs"
    
    def __init__(self) -> None:
        self.PROJECT_PATH = self.get_project_path()
        self.DATA_DIR_PATH = f"{self.PROJECT_PATH}{sep}{Config.DATA_DIR}{sep}"
        self.MSD_DIR_PATH = f"{self.DATA_DIR_PATH}{sep}{Config.MSD_DIR}{sep}"
        self.TASK_DIR_PATH = f"{self.MSD_DIR_PATH}{self.TASK_DIR}{sep}"
        self.DATA_TRAIN_DIR_PATH = f"{self.TASK_DIR_PATH}{self.DATA_TRAIN_DIR}{sep}"
        self.DATA_TRAIN_GT_DIR_PATH = f"{self.TASK_DIR_PATH}{self.DATA_TRAIN_GT_DIR}{sep}"
        self.DATA_TEST_DIR_PATH  = f"{self.TASK_DIR_PATH}{self.DATA_TEST_DIR}{sep}"

    def get_project_path(self) -> str:
        path = os.path.dirname(os.path.abspath(__file__))
        new_path = list()
        for dir in path.split(sep):
            new_path.append(dir)
            if dir == Config.PROJECT_DIR:
                break
        return sep.join(new_path)
