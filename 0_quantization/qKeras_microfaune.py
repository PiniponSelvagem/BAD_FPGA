SEED = 4

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

#import qkeras.utils as qutils
import six
import tensorflow as tf
import tensorflow.keras.backend as K
from qkeras.qconv2d_batchnorm import QConv2DBatchnorm
from qkeras.qdepthwiseconv2d_batchnorm import QDepthwiseConv2DBatchnorm
from qkeras.qrecurrent import QSimpleRNN
from qkeras.qrecurrent import QLSTM
from qkeras.qrecurrent import QGRU
from qkeras.qrecurrent import QBidirectional
from qkeras.quantizers import quantized_po2
import qkeras_microfaune_model as qmodel


#from model_config.config_test import ModelConfig            # model_test
#from model_config.config_0 import ModelConfig               # model_quant_411
#from model_config.config_0_noQuantState import ModelConfig  # model_quant_411_noQuantState
#from model_config.config_0_quantState_401 import ModelConfig  # model_quant_411_quantState_401
#from model_config.config_0_quantState_801 import ModelConfig  # model_quant_411_quantState_801
#from model_config.config_0_qconvbnorm import ModelConfig    # model_quant_411_qconvbnorm
#from model_config.config_0_qconvbnorm__input_relu import ModelConfig    # model_quant_411_qconvbnorm__input_relu
#from model_config.config_0_qconvbnorm__input_relu_tensorflowBGRU import ModelConfig    # model_quant_411_qconvbnorm__input_relu_tensorflowBGRU
#from model_config.config_0_qconvbnorm__input_relu_tensorflowBGRU_GRUnoBias import ModelConfig    # model_quant_411_qconvbnorm__input_relu_tensorflowBGRU_GRUnoBias
#from model_config.config_0_qconvbnorm__input_relu_tensorflowBGRU_GRUunits32 import ModelConfig    # model_quant_411_qconvbnorm__input_relu_tensorflowBGRU_GRUunits32
#from model_config.config_0_qconvbnorm__input_relu_tensorflowBGRU_GRUunits16 import ModelConfig    # model_quant_411_qconvbnorm__input_relu_tensorflowBGRU_GRUunits16
#from model_config.config_0_qconvbnorm__input_relu_tensorflowBGRU_GRUunits8 import ModelConfig    # model_quant_411_qconvbnorm__input_relu_tensorflowBGRU_GRUunits8
#from model_config.config_0_qconvbnorm__input_relu_tensorflowBGRU_GRUunits4 import ModelConfig    # model_quant_411_qconvbnorm__input_relu_tensorflowBGRU_GRUunits4
#from model_config.config_0_qconvbnorm__input_relu_tensorflowBGRU_GRUunits2 import ModelConfig    # model_quant_411_qconvbnorm__input_relu_tensorflowBGRU_GRUunits2
#from model_config.config_0_qconvbnorm__input_relu_tensorflowBGRU_GRUunits1 import ModelConfig    # model_quant_411_qconvbnorm__input_relu_tensorflowBGRU_GRUunits1
#from model_config.config_0_qconvbnorm__input_relu_tensorflowBGRU_GRUunits1_e200 import ModelConfig    # model_quant_411_qconvbnorm__input_relu_tensorflowBGRU_GRUunits1_e200
from model_config.config_0_qconvbnorm__input_relu_tensorflowBGRU_GRUunits1_timedistFilters1 import ModelConfig    # model_quant_411_qconvbnorm__input_relu_tensorflowBGRU_GRUunits1_timedistFilters1
#from model_config.config_1 import ModelConfig               # model_quant__conv-po2-81_gru-po2-81_bnorm-

import csv
import pickle
from microfaune.audio import wav2spc
import os
import csv
import pickle
import glob
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import qkeras_microfaune_model as qmodel

datasets_dir = '../../datasets'
save_dir = ModelConfig.folder
model_name = ModelConfig.name
plots_dir = "plots"
epochs = ModelConfig.epochs
steps_per_epoch = ModelConfig.steps_per_epoch
#model_original, dual_model_original = qmodel.MicrofauneAI.model()
model, dual_model = qmodel.MicrofauneAI(ModelConfig).modelQuantized()
"""
model = model_original
dual_model = dual_model_original
"""

def keep_percentage_of_array(arr, percentage):
    # Calculate the index up to which to keep elements
    index_to_keep = int(len(arr) * percentage)
    new_array = arr[:index_to_keep]
    return new_array

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

#### ModelConfig training_dataset_percentage ####
X0 = keep_percentage_of_array(X0, ModelConfig.training_dataset_percentage)
Y0 = keep_percentage_of_array(Y0, ModelConfig.training_dataset_percentage)
uids0 = keep_percentage_of_array(uids0, ModelConfig.training_dataset_percentage)

X1 = keep_percentage_of_array(X1, ModelConfig.training_dataset_percentage)
Y1 = keep_percentage_of_array(Y1, ModelConfig.training_dataset_percentage)
uids1 = keep_percentage_of_array(uids1, ModelConfig.training_dataset_percentage)
#################################################

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

history = model.fit(data_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                validation_data=(X_test, Y_test),
                                class_weight={0: alpha, 1: 1-alpha},
                                callbacks=[reduce_lr], verbose=1)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model.save_weights(f"{save_dir}/{model_name}.h5")

#
# Taken from qkeras.utils, and modified so it dosent crash on non quantized layers like GRU
#
# Model utilities: before saving the weights, we want to apply the quantizers
#
def model_save_quantized_weights(model, filename=None):
  """Quantizes model for inference and save it.
  #
  Takes a model with weights, apply quantization function to weights and
  returns a dictionary with quantized weights.
  #
  User should be aware that "po2" quantization functions cannot really
  be quantized in meaningful way in Keras. So, in order to preserve
  compatibility with inference flow in Keras, we do not covert "po2"
  weights and biases to exponents + signs (in case of quantize_po2), but
  return instead (-1)**sign*(2**round(log2(x))). In the returned dictionary,
  we will return the pair (sign, round(log2(x))).
  #
  Arguments:
    model: model with weights to be quantized.
    filename: if specified, we will save the hdf5 containing the quantized
      weights so that we can use them for inference later on.
  #
  Returns:
    dictionary containing layer name and quantized weights that can be used
    by a hardware generator.
  #
  """
  #
  saved_weights = {}
  #
  print("... quantizing model")
  for layer in model.layers:
    if hasattr(layer, "get_quantizers"):
      if any(isinstance(layer, t) for t in [QBidirectional]):   # ADDED to skip GRU and Bidirectional
        continue       # ADDED to skip GRU and Bidirectional
      weights = []
      signs = []
      #
      if any(isinstance(layer, t) for t in [QConv2DBatchnorm, QDepthwiseConv2DBatchnorm]):
        qs = layer.get_quantizers()
        ws = layer.get_folded_weights()
      elif any(isinstance(layer, t) for t in [QSimpleRNN, QLSTM, QGRU]):
        qs = layer.get_quantizers()[:-1]
        ws = layer.get_weights()
      else:
        qs = layer.get_quantizers()
        ws = layer.get_weights()
      #
      has_sign = False
      #
      for quantizer, weight in zip(qs, ws):
        if quantizer:
          weight = tf.constant(weight)
          weight = tf.keras.backend.eval(quantizer(weight))
        #
        # If quantizer is power-of-2 (quantized_po2 or quantized_relu_po2),
        # we would like to process it here.
        #
        # However, we cannot, because we will loose sign information as
        # quanized_po2 will be represented by the tuple (sign, log2(abs(w))).
        #
        # In addition, we will not be able to use the weights on the model
        # any longer.
        #
        # So, instead of "saving" the weights in the model, we will return
        # a dictionary so that the proper values can be propagated.
        #
        weights.append(weight)
        #
        has_sign = False
        if quantizer:
          if isinstance(quantizer, six.string_types):
            q_name = quantizer
          elif hasattr(quantizer, "__name__"):
            q_name = quantizer.__name__
          elif hasattr(quantizer, "name"):
            q_name = quantizer.name
          elif hasattr(quantizer, "__class__"):
            q_name = quantizer.__class__.__name__
          else:
            q_name = ""
        if quantizer and ("_po2" in q_name):
          # Quantized_relu_po2 does not have a sign
          if isinstance(quantizer, quantized_po2):
            has_sign = True
          sign = np.sign(weight)
          # Makes sure values are -1 or +1 only
          sign += (1.0 - np.abs(sign))
          weight = np.round(np.log2(np.abs(weight)))
          signs.append(sign)
        else:
          signs.append([])
      #
      saved_weights[layer.name] = {"weights": weights}
      if has_sign:
        saved_weights[layer.name]["signs"] = signs
      #
      if not any(isinstance(layer, t) for t in [QConv2DBatchnorm, QDepthwiseConv2DBatchnorm]):
        layer.set_weights(weights)
      else:
        print(layer.name, " conv and batchnorm weights cannot be seperately"
              " quantized because they will be folded before quantization.")
    else:
      if layer.get_weights():
        print(" ", layer.name, "has not been quantized")
  #
  if filename:
    model.save_weights(filename)
  #
  return saved_weights

model_save_quantized_weights(model, f"{save_dir}/{model_name}__quantweights.h5")    # cant use this if GRU not quantized using QKeras


if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')   # WSL
from matplotlib import pyplot as plt
plt.figure(figsize=(9, 6))
plt.title("Training / Validation loss")
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"{plots_dir}/{model_name}-TvsV_loss.png")

plt.figure(figsize=(9, 6))
plt.title("Training / Validation Accuracy")
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(f"{plots_dir}/{model_name}-TvsV_accuracy.png")


dual_model.load_weights(f"{save_dir}/{model_name}.h5")
#wav_files = {os.path.basename(f)[:-4]: f for f in glob.glob(os.path.join(datasets_dir, "*/wav/*.wav"))}
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
plt.savefig(f"{plots_dir}/{model_name}-PR_curve.png")
#plt.show()



import time
import keras.backend as K
for layer in model.layers:
    try:
        if layer.get_quantizers():
            q_gw_w_pairs = [(quantizer, gweight, weight) for quantizer, gweight, weight in zip(layer.get_quantizers(), layer.get_weights(), layer.weights)]
            for _, (quantizer, gweight, weight) in enumerate(q_gw_w_pairs):
                print(weight.name)
                if (quantizer != None):
                    qweight = K.eval(quantizer(gweight))
                    print(qweight)
                #time.sleep(1)
    except AttributeError:
        print("warning, the weight is not quantized in the layer", layer.name)

