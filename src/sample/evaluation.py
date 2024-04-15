import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix


class Evaluation:

    @classmethod
    def __convert_to_bool(cls, img: NDArray) -> NDArray:
        return np.asarray(img).astype(bool)
    
    @classmethod
    def __check_shape(cls, img1: NDArray, img2: NDArray):
        if img1.shape != img2.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    @classmethod
    def sensitivity(cls, gt: NDArray, pred: NDArray, empty_score: float = 0.0) -> float:
        """
        True positive rate
        """
        gt = Evaluation.__convert_to_bool(gt)
        pred = Evaluation.__convert_to_bool(pred)
        
        tn, fp, fn, tp = confusion_matrix(gt.flatten(), pred.flatten()).ravel()
        if tp+fn == 0:
            return empty_score
        
        return tp / (tp+fn)

    @classmethod
    def specificity(cls, gt: NDArray, pred: NDArray, empty_score: float = 0.0) -> float:
        """
        True negative rate
        """
        gt = Evaluation.__convert_to_bool(gt)
        pred = Evaluation.__convert_to_bool(pred)
        
        tn, fp, fn, tp = confusion_matrix(gt.flatten(), pred.flatten()).ravel()
        if tn+fp == 0:
            return empty_score
        
        return tn / (tn+fp)

    @classmethod
    def dice(cls, img1: NDArray, img2: NDArray, empty_score: float = 1.0) -> float:
        """
        Dice Coefficient, aka F1 score
        """
        img1 = Evaluation.__convert_to_bool(img1)
        img2 = Evaluation.__convert_to_bool(img2)

        total_area = img1.sum() + img2.sum()
        if total_area == 0:
            return empty_score

        intersection = np.sum(np.logical_and(img1, img2))

        return np.mean(2.0 * intersection / total_area)

    @classmethod
    def iou(cls, img1: NDArray, img2: NDArray) -> float:
        """
        Jaccard Index, aka Intersection over Union
        """
        img1 = Evaluation.__convert_to_bool(img1)
        img2 = Evaluation.__convert_to_bool(img2)

        intersection = np.sum(img1 * img2)
        union = np.sum(img1) + np.sum(img2) - intersection

        return np.mean(intersection / union)
