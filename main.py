import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Concatenate, Add
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from keras.utils.vis_utils import plot_model

from datetime import datetime
import numpy as np
import pandas as pd
from functools import reduce
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
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


from tools import KoopMan, state_encoder_model, state_decoder_model, linear_system

#to reduce the tensorflow messages
# tf.get_logger().setLevel('WARNING')
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')


def data_gen(X_train, U_train, k):
    # organizes the data and centers it between -1, 1
    Data = []
    x_max, x_min = [], []
    u_max, u_min = [],  []
    s_mean, u_mean = np.zeros(X_train[0].shape[1]), np.zeros(U_train[0].shape[1])
    for j in range(len(X_train)):
        x_max.append(X_train[j].max(axis=0).tolist())
        x_min.append(X_train[j].min(axis=0).tolist())

        u_max.append(U_train[j].max(axis=0).tolist())
        u_min.append(U_train[j].min(axis=0).tolist())
        u_data = U_train[j].tolist()
        s_data = X_train[j].tolist()
        for i in range(len(u_data) - k - 1):
            state_flatten = reduce(lambda a,b:a+b, s_data[i+1:i+1+k])
            x0_u_flatlist = reduce(lambda a,b:a+b, [s_data[i]]+u_data[i:i+k])
            Data.append(x0_u_flatlist + state_flatten)
    x_max_val = np.array(x_max).max(axis=0).tolist()
    x_min_val = np.array(x_min).min(axis=0).tolist()
    u_max_val = np.array(u_max).max(axis=0).tolist()
    u_min_val = np.array(u_min).min(axis=0).tolist()
    XUmax =  np.array(x_max_val + u_max_val*k + x_max_val*k)
    XUmin =  np.array(x_min_val + u_min_val*k + x_min_val*k)
    centered_data = list(map(lambda x: 2*(x - XUmin) / (XUmax - XUmin) - 1,Data))
    # print(XUmax, XUmin)
    return centered_data, (x_max_val, x_min_val), (u_max_val, u_min_val)


if __name__ == '__main__':
    INPUT_SHAPE = 4
    OUTPUT_SHAPE = 4
    DENSE_LAYERS = [4, 16, 32, 64, 128, 256]
    LATENT_DIM = 256
    X_DIM = 4
    U_DIM = 2

    BATCH_SIZE = 1000
    AUTOTUNE = tf.data.AUTOTUNE
    LEARNING_RATE = 2e-4
    PATIENCE = 10
    EPOCHS  = 50000
    TRAIN_SPLIT = 0.99
    K = 1
    LOGDIR = LOGDIR = os.path.join(r"C:\Users\kkosara\SurrogateModel\logdir", datetime.now().strftime("%Y%m%d-%H%M%S"))


    U_train = np.load('U_train.npy', mmap_mode=None, allow_pickle=True, fix_imports=True)
    X_train = np.load('X_train.npy', mmap_mode=None, allow_pickle=True, fix_imports=True)    
    
    data, s_mean, u_mean = data_gen(X_train, U_train, k=K)
    data_count = np.shape(data)[0]


    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(buffer_size=100, seed=42, reshuffle_each_iteration=True)

    # Split the dataset into training and validation sets
    train_dataset = dataset.take(int(TRAIN_SPLIT * data_count))
    train_dataset = train_dataset.shuffle(buffer_size=500, seed=42, reshuffle_each_iteration=False)
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    val_dataset = dataset.skip(int(TRAIN_SPLIT * data_count))
    val_dataset = val_dataset.shuffle(buffer_size=500, seed=42, reshuffle_each_iteration=False)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    
    
    
    # Training
    encoder = state_encoder_model(INPUT_SHAPE, DENSE_LAYERS, LATENT_DIM)
    decoder = state_decoder_model(OUTPUT_SHAPE, DENSE_LAYERS, LATENT_DIM)
    lin_sys = linear_system(LATENT_DIM, U_DIM)
    print(encoder.summary())
    print(decoder.summary())
    print(lin_sys.summary())

    koopman = KoopMan(encoder, decoder, lin_sys, X_DIM, U_DIM)
    koopman.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGDIR)
    history = koopman.fit(train_dataset, epochs=EPOCHS, callbacks=[tensorboard_callback])
    koopman.save('k')
    # # LossFunc    =     {'output_1':'mse', 'output_2':'mse', 'output_2':'mse', 'output_2':'mse', 'output_2':'mse'}
    # # lossWeights =     {'output_1':0.5, 'output_2':0.5}
    # autoencoder.compile(optimizer=RMSprop(learning_rate=0.0001), loss='huber')#, loss=LossFunc, loss_weights=lossWeights
    # # autoencoder.build(input_shape)
    # # autoencoder.summary()
    # checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model",save_best_only=True, save_format="tf")
    # early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    # history = autoencoder.fit(S_train, Y_train, epochs=100, shuffle=True, validation_split=0.1,  callbacks=[checkpoint_cb, early_stopping_cb])

    # save
    history_df = pd.DataFrame(history)
    history_df.to_csv("history_df.csv", sep='\t')