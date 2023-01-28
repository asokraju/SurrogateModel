# imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Concatenate, Add
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.backend as K


import numpy as np
import pandas as pd
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import math

class Lin_nn_k(keras.Model):
    """
    Creates a multi-input and mullti-output neural network that models a input-output dynamical system with k-step accuracy
    x_dim: dimension of state vector
    u_dim: dimension of input vector
    y_dim: dimention of lifted state vector
    w_dim: dimension of lifted input vector
    phi_hidden_layers: hidden layer neurons for forward and inverse function of state <---> lifted state
    
    psi_hidden_layers: hidden layer neurons for forward and inverse function of input+state <---> lifted input + state
    """
    def __init__(self, 
                 x_dim=3,
                 u_dim=2,
                 y_dim=6,
                 w_dim=4,
                 k=5,
                 phi_hidden_layers=[30, 20, 10],
                 psi_hidden_layers=[30, 20, 10],
                 **kwargs):
        super().__init__(**kwargs)
        self.x_dim, self.u_dim = x_dim,u_dim
        self.y_dim, self.w_dim = y_dim, w_dim # latent dim
        self.k=k
        self.phi_hidden_layers = phi_hidden_layers + [self.y_dim]
        self.psi_hidden_layers = psi_hidden_layers + [self.w_dim]
        self.phi_layers_inv = self.phi_hidden_layers[::-1] # reversing the layer dimension order
        self.psi_layers_inv = self.psi_hidden_layers[::-1] # reversing the layer dimension order

        # phi network
        self.Layers_phi_start = [Dense(units=self.phi_hidden_layers[0], activation='elu', kernel_initializer='he_uniform', name = "phi_in", kernel_regularizer=keras.regularizers.l1_l2(0.01, 0.01))]
        self.Layers_phi_mid = [Dense(units=dim, activation='elu', kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l1_l2(0.01, 0.02)) for dim in self.phi_hidden_layers[1:-1]]
        self.Layers_phi_last = [Dense(units=self.phi_hidden_layers[-1], activation='elu', kernel_initializer='he_uniform', name = "phi_out", kernel_regularizer=keras.regularizers.l1_l2(0.0, 0.0))]
        self.Layers_phi = self.Layers_phi_start + self.Layers_phi_mid + self.Layers_phi_last

        #psi network
        self.Layers_psi_start = [Dense(units=self.psi_hidden_layers[0], activation='elu', kernel_initializer='he_uniform', name = "psi_in", kernel_regularizer=keras.regularizers.l1_l2(0.0, 0.0))]
        self.Layers_psi_mid = [Dense(units=dim, activation='elu', kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l1_l2(0.01, 0.02)) for dim in self.psi_hidden_layers[1:-1]]
        self.Layers_psi_last = [Dense(units=self.psi_hidden_layers[-1], activation='elu', kernel_initializer='he_uniform',name = "psi_out", kernel_regularizer=keras.regularizers.l1_l2(0.01, 0.01))]
        self.Layers_psi = self.Layers_psi_start + self.Layers_psi_mid + self.Layers_psi_last

        # phi inverse
        self.Layers_phi_inv_start = [Dense(units=self.phi_layers_inv[1], activation='elu', kernel_initializer='he_uniform', name = "phi_inv_in", kernel_regularizer=keras.regularizers.l1_l2(0.0, 0.0))]
        self.Layers_phi_inv_mid = [Dense(units=dim, activation='elu', kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l1_l2(0.01, 0.02)) for dim in self.phi_layers_inv[2:]]
        self.Layers_phi_inv_last = [Dense(units=self.x_dim, activation=None, kernel_initializer='he_uniform',name = "phi_inv_out", kernel_regularizer=keras.regularizers.l1_l2(0.01, 0.01))]
        self.Layers_phi_inv = self.Layers_phi_inv_start + self.Layers_phi_inv_mid + self.Layers_phi_inv_last

        # psi inverse
        self.Layers_psi_inv_start = [Dense(units=self.psi_layers_inv[1], activation='elu', kernel_initializer='he_uniform', name = "psi_inv_in", kernel_regularizer=keras.regularizers.l1_l2(0.0, 0.0))]
        self.Layers_psi_inv_mid = [Dense(units=dim, activation='elu', kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l1_l2(0.01, 0.02)) for dim in self.psi_layers_inv[2:]]
        self.Layers_psi_inv_last = [Dense(units=self.u_dim, activation=None, kernel_initializer='he_uniform',name = "psi_inv_out", kernel_regularizer=keras.regularizers.l1_l2(0.01, 0.01))]
        self.Layers_psi_inv = self.Layers_psi_inv_start + self.Layers_psi_inv_mid + self.Layers_psi_inv_last

        # linear system layers
        self.A_layers = [Dense(units=self.y_dim, activation=None, kernel_initializer='he_uniform', name = "Ay", use_bias=False)]
        self.B_layers = [Dense(units=self.y_dim, activation=None, kernel_initializer='he_uniform', name = "Bw", use_bias=False)]

        # network initialization
        self.Phi_nn = keras.models.Sequential(self.Layers_phi)
        self.Psi_nn = keras.models.Sequential(self.Layers_psi)
        self.Phi_inv_nn = keras.models.Sequential(self.Layers_phi_inv)
        self.Psi_inv_nn = keras.models.Sequential(self.Layers_psi_inv)

        self.A_nn = keras.models.Sequential(self.A_layers)
        self.B_nn = keras.models.Sequential(self.B_layers)
        self.lin_sys = Add(name="linear_sys")

        self.input_x = Input(self.x_dim, name="state")
        self.input_u = Input(self.u_dim, name="action")
        self.input_xu = Concatenate(name = "state_and_action")
        self.input_yw = Concatenate(name = "y_and_w")
    
    def nn_model(self, state, action):
        state_action = self.input_xu([state, action])
        Phi = self.Phi_nn(state)
        Psi = self.Psi_nn(state_action)
        Ay = self.A_nn(Phi)
        Bw = self.B_nn(Psi)
        y_new = self.lin_sys([Ay, Bw])
        state = self.Phi_inv_nn(y_new)
        y_w = self.input_yw([Phi, Psi])
        u_pred = self.Psi_inv_nn(y_w)
        return state, u_pred, [Phi, Psi, Ay, Bw, y_new]
    
    def call(self, inputs):
        state, u1, u2, u3, u4, u5 = inputs
        predicted_action, predicted_state = [], []
        actions = [u1, u2, u3, u4, u5]
        for i in range(self.k):
            # state_action = self.input_xu([state, actions[k]])
            # Phi = self.Phi_nn(state)
            # Psi = self.Psi_nn(state_action)
            # Ay = self.A_nn(Phi)
            # Bw = self.B_nn(Psi)
            # y_new = self.lin_sys([Ay, Bw])
            # state = self.Phi_inv_nn(y_new)
            # y_w = self.input_yw([Phi, Psi])
            # u_pred = self.Psi_inv_nn(y_w)
            # print(u_pred, predicted_action, predicted_action[2*k])
            state, u_pred, _ = self.nn_model(state, actions[i])
            predicted_action.append(u_pred)
            predicted_state.append(state)
            # print(predicted_state, predicted_action)
            # print(predicted_action)
            action_loss = tf.reduce_mean(tf.square(u_pred - actions[i]))
            self.add_loss(action_loss)
        # print(predicted_state, predicted_action)
        # predicted_state = tf.reshape(predicted_state, [self.k*self.x_dim])
        # predicted_action = tf.reshape(predicted_action, [self.k*self.u_dim])
        # output = tf.concat([predicted_state, predicted_action], axis=0)
        # print(predicted_state, predicted_action, output)
        return predicted_state