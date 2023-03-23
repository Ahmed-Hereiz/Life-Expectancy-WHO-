import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras import backend as K
from keras import layers, metrics
from tensorflow.keras import regularizers
from sklearn.model_selection import KFold
from tensorflow.keras.layers import BatchNormalization, Add
from tensorflow.keras.models import Model



class ForwardfeedNN:
    """A forward-feed neural network model for regression.

    This model implements a forward-feed neural network architecture for regression
    problems using the Keras API from TensorFlow. The architecture consists of several
    dense layers with batch normalization and dropout regularization to avoid overfitting.

    Parameters:
        None

    Attributes:
        kf: An instance of the KFold class used for cross-validation.
        model: A Keras Sequential model that contains the neural network architecture.

    Methods:
        _build_model: Private method that builds the neural network architecture.
        fit: Trains the model on the training data.
        predict: Generates predictions for new input data.
    """
    def __init__(self):
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42)
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            layers.Dense(units=512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(units=256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(units=256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(units=256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(units=256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(units=256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(units=256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(units=128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(units=128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(units=64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(units=32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(units=1),
        ])

        adam = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(optimizer=adam, loss='mean_absolute_error', metrics=[metrics.MeanAbsoluteError(name='mae')])
        return model

    def fit(self, X, y, epochs=800, batch_size=256*2, verbose=1):
        for fold, (train_indices, val_indices) in enumerate(self.kf.split(X)):
            x_train_fold = X.iloc[train_indices]
            y_train_fold = y.iloc[train_indices]
            x_val_fold = X.iloc[val_indices]
            y_val_fold = y.iloc[val_indices]
            
        self.model.fit(x_train_fold, y_train_fold, epochs=epochs, batch_size=batch_size, verbose=verbose, 
                        validation_data=(x_val_fold, y_val_fold))
        
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    
    
class ResNet:
    """A fResNet network model for regression.

    This model implements a Resnet neural network architecture for regression
    problems using the Keras API from TensorFlow. The architecture consists of several
    dense layers with batch normalization and dropout regularization to avoid overfitting.

    Parameters:
        None

    Attributes:
        kf: An instance of the KFold class used for cross-validation.
        model: A Keras Sequential model that contains the neural network architecture.

    Methods:
        _build_model: Private method that builds the neural network architecture.
        fit: Trains the model on the training data.
        predict: Generates predictions for new input data.
    """
    def __init__(self):
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42)
        self.model = self._build_model()

    def _build_model(self):
        # Define input
        inputs = Input(shape=(X_train.shape[1],))

        # Initial Dense layer
        x = Dense(units=256, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        # Residual layers
        for i in range(2):
            # Inner Dense layer
            inner = Dense(units=256)(x)
            inner = BatchNormalization()(inner)
            inner = Activation('relu')(inner)
            inner = Dropout(0.2)(inner)

            # Inner Dense layer
            inner = Dense(units=256)(inner)
            inner = BatchNormalization()(inner)
            inner = Activation('relu')(inner)
            inner = Dropout(0.2)(inner)

            # Inner Dense layer
            inner = Dense(units=256)(inner)
            inner = BatchNormalization()(inner)
            inner = Activation('relu')(inner)
            inner = Dropout(0.2)(inner)

            # Add the input to the output of the residual layers
            x = Add()([x, inner])
            x = Activation('relu')(x)
            x = Dropout(0.2)(x)

        # Final Dense layers
        x = Dense(units=128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(units=64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(units=32, activation='relu')(x)
        x = BatchNormalization()(x)
        outputs = Dense(units=1)(x)

        # Define the model
        resnet = Model(inputs=inputs, outputs=outputs)

        adam = tf.keras.optimizers.Adam(learning_rate=0.001)

        resnet.compile(optimizer=adam, loss='mean_absolute_error', metrics=[metrics.MeanAbsoluteError(name='mae')])

        return resnet
    
    def fit(self, X_train, y_train):
        for fold, (train_indices, val_indices) in enumerate(self.kf.split(X_train)):
            x_train_fold = X_train.iloc[train_indices]
            y_train_fold = y_train.iloc[train_indices]
            x_val_fold = X_train.iloc[val_indices]
            y_val_fold = y_train.iloc[val_indices]

        self.model.fit(x=x_train_fold, y=y_train_fold, epochs=800, batch_size=256*2, validation_data=(x_val_fold, y_val_fold))

    def predict(self, X_test):
        return self.model.predict(X_test)