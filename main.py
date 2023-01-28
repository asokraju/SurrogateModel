import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Concatenate, Add
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.backend as K
from keras.utils.vis_utils import plot_model


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# import gym
# from gym import spaces
# from gym.utils import seeding
# import math
import os
# import argparse
# import pprint as pp

# import time
# from joblib import dump, load
# from scipy.io import savemat
# import datetime
# import json


from tools.modules import Lin_nn_k

#to reduce the tensorflow messages
# tf.get_logger().setLevel('WARNING')
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')

if __name__ == '__main__':
        
    # Checking if we are using gpu
    # assert tf.test.is_gpu_available()
    # assert tf.test.is_built_with_cuda()
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    
    # load data
    df_data = pd.read_csv("data.csv", sep = "\t", index_col=0)

    k = 5
    mask_5 = df_data.k==k
    df_5 = df_data[mask_5]
    print(df_5.head())

    df_shape = df_5.shape[0]
    print(df_shape)
    

    # data transformation
    scalar = StandardScaler()

    train_data = scalar.fit_transform(df_5)
    S0, S, U = train_data[:,1:4].astype('float32'), train_data[:,4:4+3*k].astype('float32'), train_data[:,4+3*k:].astype('float32')
    U1, U2, U3, U4, U5 = U[:,0:2], U[:,2:4], U[:,4:6], U[:,6:8], U[:,8:10]
    S1, S2, S3, S4, S5 = S[:,0:3], S[:,3:6], S[:,6:9], S[:,9:12], S[:,12:15]

    S_train = [S0, U1, U2, U3, U4, U5]
    Y_train = [S1, S2, S3, S4, S5] #train_data[:,4:].astype('float32')#[[S1, S2, S3, S4, S5], [U1, U2, U3, U4, U5]]
    # Y_train[0]


    # Training
    autoencoder = Lin_nn_k()
    # LossFunc    =     {'output_1':'mse', 'output_2':'mse', 'output_2':'mse', 'output_2':'mse', 'output_2':'mse'}
    # lossWeights =     {'output_1':0.5, 'output_2':0.5}
    autoencoder.compile(optimizer=RMSprop(learning_rate=0.0001), loss='huber')#, loss=LossFunc, loss_weights=lossWeights
    # autoencoder.build(input_shape)
    # autoencoder.summary()
    checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model",save_best_only=True, save_format="tf")
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    history = autoencoder.fit(S_train, Y_train, epochs=100, shuffle=True, validation_split=0.1,  callbacks=[checkpoint_cb, early_stopping_cb])

    # save
    history_df = pd.DataFrame(history)
    history_df.to_csv("history_df.csv", sep='\t')
