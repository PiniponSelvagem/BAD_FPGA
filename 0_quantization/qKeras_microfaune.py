SEED = 43

import os
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import random

import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

os.environ['PYTHONHASHSEED']=str(SEED)
tf.compat.v1.reset_default_graph()
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import math



from qkeras import *

import csv
import pickle
from microfaune.audio import wav2spc
import os
import csv
import pickle
import glob
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
datasets_dir = '../../datasets'
save_dir = "model_quantized"
model_subname = "quantized_po2_81"
epochs = 100
bits = "8"
max_value = "1"
config = {
    "QConv2D": {
        "kernel_quantizer": f"quantized_po2({bits}, {max_value})",
        "bias_quantizer": f"quantized_po2({bits}, {max_value})"
    },
    "QBatchNormalization": {
        "axis": "-1",
        "momentum": "0.95"
    },
    "QActivation": {
        "activation": f"quantized_po2({bits}, {max_value})"
    },
    "QGru": {
        "kernel_quantizer": f"quantized_po2({bits}, {max_value})",
        "recurrent_quantizer": f"quantized_po2({bits}, {max_value})"
    },
    "QDense": {
        "kernel_quantizer": f"quantized_po2({bits}, {max_value})",
        "bias_quantizer": f"quantized_po2({bits}, {max_value})"
    }
}


#detector = RNNDetector()
#model = detector.model
#model.summary()
class MicrofauneAI():
    def model():
        n_filter = 64
        conv_reg = keras.regularizers.l2(1e-3)
        #
        spec = layers.Input(shape=[431, 40, 1], dtype=np.float32)
        #
        x = layers.Conv2D(n_filter, (3, 3), padding="same", kernel_regularizer=conv_reg, activation=None)(spec)
        x = layers.BatchNormalization(momentum=0.95)(x)
        x = layers.ReLU()(x)
        #
        x = layers.Conv2D(n_filter, (3, 3), padding="same", kernel_regularizer=conv_reg, activation=None)(x)
        x = layers.BatchNormalization(momentum=0.95)(x)
        x = layers.ReLU()(x)
        #
        x = layers.MaxPool2D((1, 2))(x)
        #
        x = layers.Conv2D(n_filter, (3, 3), padding="same", kernel_regularizer=conv_reg, activation=None)(x)
        x = layers.BatchNormalization(momentum=0.95)(x)
        x = layers.ReLU()(x)
        #
        x = layers.Conv2D(n_filter, (3, 3), padding="same", kernel_regularizer=conv_reg, activation=None)(x)
        x = layers.BatchNormalization(momentum=0.95)(x)
        x = layers.ReLU()(x)
        #
        x = layers.MaxPool2D((1, 2))(x)
        #
        x = layers.Conv2D(n_filter, (3, 3), padding="same", kernel_regularizer=conv_reg, activation=None)(x)
        x = layers.BatchNormalization(momentum=0.95)(x)
        x = layers.ReLU()(x)
        #
        x = layers.Conv2D(n_filter, (3, 3), padding="same", kernel_regularizer=conv_reg, activation=None)(x)
        x = layers.BatchNormalization(momentum=0.95)(x)
        x = layers.ReLU()(x)
        #
        x = layers.MaxPool2D((1, 2))(x)
        #
        x = math.reduce_max(x, axis=-2)
        #
        x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
        x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
        #
        x = layers.TimeDistributed(layers.Dense(64, activation="sigmoid"))(x)
        local_pred = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"))(x)
        pred = math.reduce_max(local_pred, axis=-2)
        #
        model = keras.Model(inputs=spec, outputs=pred)
        #
        # for predictions only
        dual_model = keras.Model(inputs=spec, outputs=[pred, local_pred])
        return model, dual_model


model_original, dual_model_original = MicrofauneAI.model()

"""
model = model_original
dual_model = dual_model_original
"""
from qkeras.utils import model_quantize
model = model_quantize(model_original, config, 4, transfer_weights=True)
dual_model = model_quantize(dual_model_original, config, 4, transfer_weights=True)
#model.summary()


def adjust_dataset(dataset):
    """
    This function adjusts the dataset by making sure all instances of it have shape=(40,431).
    Elements with shape that the 2nd dim is bigger than 431, is truncated to 431.
    Elements with shape that the 2nd dim is smaller than 431, is removed from the dataset.
    """
    subset_data = "X"
    subset_class = "Y"
    subset_uids = "uids"
    #
    desired_shape = (40, 431)
    removed_indices = []
    #
    for i in range(len(dataset[subset_data])):
        element = dataset[subset_data][i]
        #
        # If the element has more than 431 sub-elements, truncate to 431
        if element.shape[1] > desired_shape[1]:
            dataset[subset_data][i] = element[:, :desired_shape[1]]
        # If the element has fewer than 431 sub-elements, remove it and add its index to the list
        elif element.shape[1] < desired_shape[1]:
            removed_indices.append(i)
    #
    # Remove elements with fewer than 431 sub-elements from subset_data
    dataset[subset_data] = [element for i, element in enumerate(dataset[subset_data]) if i not in removed_indices]
    #
    # Remove corresponding elements from other subsets matching the same instance
    dataset[subset_class] = [element for i, element in enumerate(dataset[subset_class]) if i not in removed_indices]
    dataset[subset_uids]  = [element for i, element in enumerate(dataset[subset_uids])  if i not in removed_indices]
    #
    return dataset

def load_dataset(data_path, use_dump=True):
    mel_dump_file = os.path.join(data_path, "mel_dataset.pkl")
    if os.path.exists(mel_dump_file) and use_dump:
        with open(mel_dump_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = compute_feature(data_path)
        with open(mel_dump_file, "wb") as f:
            pickle.dump(dataset, f)
    dataset = adjust_dataset(dataset)   # This line fixed the problem of very few instances in the dataset when comparing to the microfaune_ai python notebook
    inds = [i for i, x in enumerate(dataset["X"]) if x.shape[1] == 431]
    X = np.array([dataset["X"][i].transpose() for i in inds])
    Y = np.array([int(dataset["Y"][i]) for i in inds])
    uids = [dataset["uids"][i] for i in inds]
    return X, Y, uids

def compute_feature(data_path):
    print(f"Compute features for dataset {os.path.basename(data_path)}")
    labels_file = os.path.join(data_path, "labels.csv")
    if os.path.exists(labels_file):
        with open(labels_file, "r") as f:
            reader = csv.reader(f, delimiter=',')
            labels = {}
            next(reader)  # pass fields names
            for name, y in reader:
                labels[name] = y
    else:
        print("Warning: no label file detected.")
        wav_files = glob(os.path.join(data_path, "wav/*.wav"))
        labels = {os.path.basename(f)[:-4]: None for f in wav_files}
    i = 1
    X = []
    Y = []
    uids = []
    for file_id, y in labels.items():
        print(f"{i:04d}/{len(labels)}: {file_id:20s}", end="\r")
        spc = wav2spc(os.path.join(data_path, "wav", f"{file_id}.wav"))
        X.append(spc)
        Y.append(y)
        uids.append(file_id)
        i += 1
    return {"uids": uids, "X": X, "Y": Y}

def split_dataset(X, Y, random_state=0):
    split_generator = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    ind_train, ind_test = next(split_generator.split(X, Y))
    X_train, X_test = X[ind_train, :, :], X[ind_test, :, :]
    Y_train, Y_test = Y[ind_train], Y[ind_test]
    return ind_train, ind_test


X0, Y0, uids0 = load_dataset(os.path.join(datasets_dir, "ff1010bird_wav"))
X1, Y1, uids1 = load_dataset(os.path.join(datasets_dir, "warblrb10k_public_wav"))


X = np.concatenate([X0, X1]).astype(np.float32)/255
Y = np.concatenate([Y0, Y1])
uids = np.concatenate([uids0, uids1])

del X0, X1, Y0, Y1
print(Counter(Y))






ind_train, ind_test = split_dataset(X, Y)

X_train, X_test = X[ind_train, :, :, np.newaxis], X[ind_test, :, :, np.newaxis]
Y_train, Y_test = Y[ind_train], Y[ind_test]
uids_train, uids_test = uids[ind_train], uids[ind_test]
del X, Y

X_train.shape[1:]

print("Training set: ", Counter(Y_train))
print("Test set: ", Counter(Y_test))





from datetime import datetime
class DataGenerator(keras.utils.Sequence):
    def __init__(self, X, Y, batch_size=32):
        self.X = X
        self.Y = Y
        self.n = len(Y)
        self.batch_size = batch_size
        self.shuffle()
    #
    def __len__(self):
        return int(np.floor(self.n)/self.batch_size)
    #
    def __getitem__(self, index):
        batch_inds = self.inds[self.batch_size*index:self.batch_size*(index+1)]
        self.counter += self.batch_size
        if self.counter >= self.n:
            self.shuffle()
        return self.X[batch_inds, ...], self.Y[batch_inds]
    #
    def shuffle(self):
        self.inds = np.random.permutation(self.n)
        self.counter = 0

optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.FalseNegatives()])

alpha = 0.5
batch_size = 32

data_generator = DataGenerator(X_train, Y_train, batch_size)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=5, min_lr=1e-5)

history = model.fit_generator(data_generator, steps_per_epoch=100, epochs=epochs,
                                validation_data=(X_test, Y_test),
                                class_weight={0: alpha, 1: 1-alpha},
                                callbacks=[reduce_lr], verbose=1)


if not os.path.exists(save_dir):
    os.makedirs(save_dir)
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
model.save_weights(f"{save_dir}/model_weights-{model_subname}-{date_str}.h5")


import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
plt.figure(figsize=(9, 6))
plt.title("Training / Validation loss")
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"plots/{model_subname}-TvsV_loss-{date_str}.png")

plt.figure(figsize=(9, 6))
plt.title("Training / Validation Accuracy")
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(f"plots/{model_subname}-TvsV_accuracy-{date_str}.png")




dual_model.load_weights(f"{save_dir}/model_weights-{model_subname}-{date_str}.h5")
wav_files = {os.path.basename(f)[:-4]: f for f in glob.glob(os.path.join(datasets_dir, "*/wav/*.wav"))}
scores, local_scores = dual_model.predict(X_test)

Y_hat = scores.squeeze() > 0.5
print(f"Accuracy: {np.mean(Y_hat == Y_test)*100:.2f}")
print("Accuracy: 90.18 -> Expected from 'learn_model.ipynb'")


from sklearn.metrics import roc_curve, auc
fpr, tpr, sc = roc_curve(Y_test, scores)
print(f"Area under ROC curve: {auc(fpr, tpr):f}")
print("Area under ROC curve: 0.955267 -> Expected from 'learn_model.ipynb'")




plt.figure(figsize=(9, 6))
plt.plot(1-fpr, tpr)
plt.title("Precision / Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.savefig(f"plots/{model_subname}-PR_curve-{date_str}.png")

plt.show()

