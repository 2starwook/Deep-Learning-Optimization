def binarise(y_pred) : 
    threshold, upper, lower = 0.5, 1, 0
    y_pred[y_pred >= threshold] = 1.
    y_pred[y_pred < threshold] = 0.
    return y_pred
