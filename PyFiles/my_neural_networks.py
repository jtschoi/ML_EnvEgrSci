import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from glob import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential


class dense_neural_network:
    def __init__(
        self,
        input_shape,
        output_shape,
        loss_fn,
        n_neuron,
        activation_hl,
        activation_output,
        num_epochs,
        learning_rate,
        minibatch_size,
        N_layers,
        leaky_alpha=0.01,
    ):
        act_possible = ["relu", "tanh", "leaky_relu", "linear"]
        msg = (
            "Activation function should be one of the following: "
            + ", ".join([f'"{x}"' for x in act_possible])
            + "."
        )
        assert activation_hl in act_possible, msg
        assert activation_output in act_possible, msg

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_neuron = n_neuron
        self.act_hl = activation_hl
        self.act_output = activation_output
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.N_layers = N_layers
        self.leaky_alpha = leaky_alpha
        self.loss_fn = loss_fn

        # in case the activation function is leaky ReLU
        if (self.act_hl == "leaky_relu") or (self.act_output == "leaky_relu"):

            def my_leaky_relu(x):
                return tf.nn.leaky_relu(x, alpha=0.01)

            if self.act_hl == "leaky_relu":
                self.act_hl = Activation(my_leaky_relu)
            if self.act_output == "leaky_relu":
                self.act_output = Activation(my_leaky_relu)

        # initializating the (empty) NN model
        model = Sequential()

        # adding layers; note that the input_shape is equal to the number of
        # `X_traindata` columns or variables, if I understand correctly.
        # (e.g., CO2 and CH4 as inputs --> input_shape=2)
        for n in range(self.N_layers):
            n_ = n + 1
            layername = "hidden_layer_{}".format(n_)
            if n_ == 1:
                # input layer and hidden layer 1
                to_add = Dense(
                    self.n_neuron,
                    activation=self.act_hl,
                    name=layername,
                    input_shape=(self.input_shape,),
                )
            else:
                # hidden layers 2 and beyond
                to_add = Dense(self.n_neuron, activation=self.act_hl, name=layername)

            # adding the hidden layers
            model.add(to_add)

        # adding the output layer; number of neurons is the output (y variable) shape
        model.add(
            Dense(self.output_shape, activation=activation_output, name="output_layer")
        )

        # model compiling
        model.compile(
            loss=self.loss_fn,
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
        )

        self.model = model

        return None

    def fit(
        self,
        X_train,
        y_train,
        validation_split=0.2,
        verbosity=2,
        early_stopping=True,
        es_monitor="val_loss",
        patience=20,
    ):
        self.validsplit = validation_split
        self.verbosity = verbosity
        if early_stopping:
            early_stop = keras.callbacks.EarlyStopping(
                monitor=es_monitor, patience=patience
            )
            fitted = self.model.fit(
                X_train,
                y_train,
                batch_size=self.minibatch_size,
                epochs=self.num_epochs,
                validation_split=self.validsplit,
                verbose=self.verbosity,
                callbacks=[early_stop],
            )
        else:
            fitted = self.model.fit(
                X_train,
                y_train,
                batch_size=self.minibatch_size,
                epochs=self.num_epochs,
                validation_split=self.validsplit,
                verbose=self.verbosity,
            )

        self.fitted = fitted

        return self.fitted
