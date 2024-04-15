from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from keras.layers import Input
from keras.optimizers import legacy
from keras.callbacks import ModelCheckpoint

from src.loader import DatasetLoader


PATH = './'
pipeline = DatasetLoader("Task02_Heart")
x_train, y_train, x_test = pipeline.get_train_dataset(), pipeline.get_train_gt_dataset(), pipeline.get_test_dataset()

class_wights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), 
                                    y=y_train.flatten())
class_wights = {i : w for i,w in enumerate(class_wights)}

from src.model import get_unet

input_img = Input((pipeline.img_shape[0], pipeline.img_shape[1], 1), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model.compile(optimizer=legacy.Adam())

epoch = 50
batch_size = 32

logdir = PATH + "Model/logs/unet-batch_size-{}-epochs-{}-loss-{}.h5".format(batch_size, epoch, 'dice_coef_loss') + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

history = model.fit(
    x = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2] ,1)),
    y = y_train.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2] ,1)),
    batch_size=batch_size, epochs = epoch,
    validation_split=0.2,
    class_weight=class_wights,
    callbacks=[
        tensorboard_callback,
        ModelCheckpoint(
            filepath=f"{PATH}Model/unet-pipeline-batch_size-{batch_size}-epochs-{epoch}-loss-dice_coef_loss.h5",
            verbose=1, save_best_only=True, save_weights_only=False)                            
])