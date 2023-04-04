import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Add


import numpy as np
from functools import reduce



# Sampling layer
class Sampling(layers.Layer):
    "used to sample a vector in latent space with learned mean - z_mean and (log) variance - z_log_var"
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        vec_len = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch_size, vec_len))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Define encoder model
def state_encoder_model(input_shape, dense_layers, latent_dim, nn_name = 'state'):
    # Create input layer
    encoder_inputs = keras.Input((input_shape,), name=nn_name+"_encoder_inputs")
    
    # starting layer
    x = Dense(
        units=dense_layers[0], 
        activation='elu', 
        kernel_initializer='he_uniform', 
        name = nn_name+"_phi_in", 
        kernel_regularizer=keras.regularizers.l1_l2(0.01, 0.01)
        )(encoder_inputs)
    # middle layers
    mid_layers = [
        Dense(
        units=f, 
        activation='elu', 
        kernel_initializer='he_uniform',
        kernel_regularizer=keras.regularizers.l1_l2(0.01, 0.01)
        ) 
        for f in dense_layers[1:-1]
        ]
    for mid_layer in mid_layers:
        x = mid_layer(x)

    # Add dense layer with specified number of neurons and activation function
    x = Dense(
        dense_layers[-1],
        activation='relu', 
        kernel_regularizer=keras.regularizers.l1_l2(0.0, 0.0),
        name = nn_name+"_phi_out")(x)
    
    # Add output layers for latent space (mean and variance) and sample from this space
    z_mean = layers.Dense(latent_dim, name = nn_name+"_z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name=nn_name+"_z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    
    # Create encoder model
    return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name=nn_name+'_encoder')



# Define decoder model
def state_decoder_model(ouput_shape, dense_layers, latent_dim, nn_name = 'state'):
    # Create input layer
    decoder_inputs = keras.Input((latent_dim, ), name=nn_name+"_encoder_inputs")
    dense_layers = dense_layers[::-1]
    # starting layer
    x = Dense(
        units=dense_layers[0], 
        activation='elu', 
        kernel_initializer='he_uniform', 
        name = nn_name+"_phi_inv_in", 
        kernel_regularizer=keras.regularizers.l1_l2(0.01, 0.01)
        )(decoder_inputs)
    # middle layers
    mid_layers = [
        Dense(
        units=f, 
        activation='elu', 
        kernel_initializer='he_uniform',
        kernel_regularizer=keras.regularizers.l1_l2(0.01, 0.01)
        ) 
        for f in dense_layers[1:]
        ]
    for mid_layer in mid_layers:
        x = mid_layer(x)

    # Add dense layer with specified number of neurons and activation function
    decoder_outputs = layers.Dense(
        ouput_shape,
        activation='relu', 
        kernel_regularizer=keras.regularizers.l1_l2(0.0, 0.0),
        name = nn_name+"_phi_inv_out")(x)
    
    # Create encoder model
    return keras.Model(decoder_inputs, decoder_outputs, name=nn_name+'_state_decoder')


def linear_system(latent, u_dim):
    input_y = Input(shape=(latent,))
    input_u = Input(shape=(u_dim,))
    Ay = Dense(
            units=latent, 
            activation=None, 
            kernel_initializer='he_uniform', 
            name = "Ay", 
            use_bias=False)(input_y)
    Ay = Model(inputs=input_y, outputs=Ay)

    Bw = Dense(
            units=latent, 
            activation=None, 
            kernel_initializer='he_uniform', 
            name = "Bw", 
            use_bias=False)(input_u)
    Bw = Model(inputs=input_u, outputs=Bw)
    # output = Add(name="linear_sys")([Ay, Bw])
    return Model(inputs=[Ay.input, Bw.input], outputs= Ay.output + Bw.output)

class KoopMan(keras.Model):
    def __init__(self, encoder, decoder, lin_sys, x_dim, u_dim, **kwargs):
        super().__init__(**kwargs)
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.encoder = encoder
        self.decoder = decoder
        self.lin_sys = lin_sys
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.prediction_loss_tracker = keras.metrics.Mean(name="prediction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.prediction_loss_tracker,
            self.kl_loss_tracker,
        ]
    def call(self, x):
        state, action = x[:,:self.x_dim], x[:,self.x_dim:]
        y_mean, y_log_var, y = self.encoder(state)
        reconstruction = self.decoder(y)
        reconstruction_state_new = self.lin_sys([y_mean, action])
        predicted_state_new = self.decoder(reconstruction_state_new)
        return y_mean, y_log_var, y, reconstruction, predicted_state_new

    def train_step(self, data):
        state_action, new_state = data[:,:self.x_dim+self.u_dim], data[:, self.x_dim+self.u_dim:]
        state, action = state_action[:,:self.x_dim], state_action[:,self.x_dim:]
        with tf.GradientTape() as tape:
            # z_mean, z_log_var, z = self.encoder(data)
            # reconstruction = self.decoder(z)
            y_mean, y_log_var, y, reconstruction, predicted_state_new = self(state_action)
            reconstruction_loss_state = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mse(state, reconstruction)
                )
            )
            prediction_loss_state = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mse(new_state, predicted_state_new)
                )
            )
            kl_loss = -0.5 * (1 + y_log_var - tf.square(y_mean) - tf.exp(y_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            total_loss = reconstruction_loss_state + kl_loss + prediction_loss_state
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss_state)
        self.prediction_loss_tracker.update_state(prediction_loss_state)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "prediction_loss" : self.prediction_loss_tracker.result()
        }