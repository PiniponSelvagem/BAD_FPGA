import numpy as np
from keras.utils import to_categorical

pathToTrainFeature1 = './feature_extracted/BAD_wblr_feature.npy'
pathToTrainLabel1 = './feature_extracted/BAD_wblr_label.npy'
pathToTrainFeature2 = './feature_extracted/ff1010bird_feature.npy'
pathToTrainLabel2 = './feature_extracted/ff1010bird_label.npy'

saveModelTo = './model/' # path where to save generated model and csv file with results
modelName = 'all_convnet_BAD.h5'
resultName = 'all_convnet_BAD.csv'

pathToTestMetadata = './feature_extracted/birdvox_metadata.txt'
pathToTestFeature = './feature_extracted/birdvox_feature.npy'

#train data
feature1 = np.load(pathToTrainFeature1)
label1 = np.load(pathToTrainLabel1)
label1 = to_categorical(label1, 2)

feature2 = np.load(pathToTrainFeature2)
label2 = np.load(pathToTrainLabel2)
label2 = to_categorical(label2, 2)

feature = np.concatenate([feature1, feature2])
label = np.concatenate([label1, label2])

#test_data
test_class_file = np.loadtxt(pathToTestMetadata, delimiter=',', dtype='str') # test_class_file[:,0] -> ID, test_class_file[:,1] -> real value of hasBird
test_data = np.load(pathToTestFeature)


def trainWithSeed(seed):
    import gc
    gc.collect()

    import os
    os.environ['PYTHONHASHSEED']=str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    import random

    from keras.models import Sequential
    from keras.layers.core import Flatten, Dense, Dropout, Reshape, Activation
    from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, MaxPooling1D, Conv1D
    from tensorflow.keras.layers import BatchNormalization
    from sklearn.model_selection import train_test_split
    from keras.callbacks import ModelCheckpoint
    from keras.optimizers import Adadelta, RMSprop,SGD,Adam
    import keras
    import tensorflow as tf

    import time

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    os.environ['PYTHONHASHSEED']=str(seed)
    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    start = time.time()
    #########################################################

    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2)

    model = Sequential()

    #conv1
    model.add(ZeroPadding2D((2,2),input_shape=(40,500,1)))
    model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', kernel_initializer=initializer))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.25))

    #maxpool1
    model.add(MaxPooling2D((5,1), strides=(5,1)))
    model.add(Dropout(0.50))
    #conv2

    model.add(ZeroPadding2D((2,2)))
    model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', kernel_initializer=initializer))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.25))

    #maxpool2
    model.add(MaxPooling2D((2,1), strides=(2,1)))
    model.add(Dropout(0.50))

    #conv3
    model.add(ZeroPadding2D((2,2)))
    model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', kernel_initializer=initializer))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.25))

    #maxpool3
    model.add(MaxPooling2D((2,1), strides=(2,1)))
    model.add(Dropout(0.50))

    #conv4
    model.add(ZeroPadding2D((2,2)))
    model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', kernel_initializer=initializer))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.25))

    #maxpool4
    model.add(MaxPooling2D((2,1), strides=(2,1)))
    model.add(Dropout(0.50))

    #stacking(reshaping)
    model.add(Reshape((500, 16)))

    #temporal squeeing
    model.add(MaxPooling1D((500), strides=(1)))
    model.add(Dropout(0.50))

    #fully connected layers
    model.add(Flatten())
    model.add(Dense(196, activation='sigmoid'))
    model.add(Dropout(0.50))
    model.add(Dense(2, activation='softmax', name='predictions'))
    model.summary()

    #train_data
    classes = 2
    opt = Adam(decay = 1e-6)
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, shuffle=False)

    # compile model
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    #fit the model
    hist = model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=2)

    #save_model
    if not os.path.exists(saveModelTo):
        os.mkdir(saveModelTo)
    ###model.save(saveModelTo+modelName)

    #test_label_predict
    predict_x = model.predict(test_data) 
    classes_x = np.argmax(predict_x,axis=1)

    #save_test_labels_&_probs
    pred_probs=np.array(classes_x)
    ###np.savetxt(saveModelTo+resultName,np.c_[test_class_file[:,0],pred_probs[:]],fmt='%s', delimiter=',')

    #show accuracy
    total = len(pred_probs)
    nCorrect = 0
    real_values = test_class_file[:,1]
    for i in range(len(pred_probs)):
        nCorrect = nCorrect + (int(real_values[i]) == pred_probs[i])
    accuracy = nCorrect/total
    print("Accuracy for '"+pathToTestFeature+"' dataset: "+str(accuracy))

    with open(saveModelTo+"all_convnet_maxpool_combinedTDS_seeds.csv", "a") as myfile:
        myfile.write(str(seed)+","+str(accuracy)+"\n")

    #########################################################
    end = time.time()
    return end - start


#from datetime import timedelta

for i in range(1, 50):
    trainWithSeed(i)
