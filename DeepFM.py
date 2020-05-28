import pandas as pd
import numpy as np
from tensorflow.keras import backend as K
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf

print(tf.__version__)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 自定义损失函数rmse
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

rnames = ['userId', 'movieId', 'rating', 'timestamp']
rating = pd.read_csv("../ml-1m/ratings.dat", sep="::", names=rnames,header=None, engine='python')
rating.info()
rating = rating.sample(frac=1)
num_rating = len(rating['rating'])
max_user = rating["userId"].max()
max_movie = rating["movieId"].max()
num_user = len(rating['userId'].unique())
num_movie = len(rating['movieId'].unique())
print(pd.DataFrame({
    'num_user': num_user,
    'max_user': max_user,
    'num_movie': num_movie,
    'max_movie': max_movie,
    'num_rating': num_rating,
    '填充率': num_rating / (num_user * num_movie)
}, index=[0]))

from tensorflow.keras import Model, utils
from tensorflow.keras.layers import Embedding, Reshape, Input, Dot, Dense, Dropout, BatchNormalization, Concatenate, Add, Flatten
from tensorflow.keras import regularizers, optimizers


def Recmand_model(max_user, max_item, k):
    input_user = Input(shape=(1, ), name='user')
    model_uer = Embedding(max_user + 1, k, input_length=1,)(input_user)
    model_uer = BatchNormalization(epsilon=0.001, momentum=0.99, axis=-1)(model_uer)
    # model_uer = Dense(k, activation="relu", use_bias=True,)(model_uer)  # 激活函数
    model_uer = Dense(50, activation="relu", use_bias=True, kernel_regularizer=regularizers.l2(0.005))(model_uer)  # 激活函数
    model_uer = Flatten()(model_uer)

    input_item = Input(shape=(1, ), name='item')
    model_item = Embedding(max_item + 1, k, input_length=1)(input_item)
    model_item = BatchNormalization(epsilon=0.001, momentum=0.99, axis=-1)(model_item)
    # model_item = Dense(k, activation="relu", use_bias=True,)(model_item)
    model_item = Dense(50, activation="relu", use_bias=True, kernel_regularizer=regularizers.l2(0.005))(model_item)  # 激活函数
    model_item = Flatten()(model_item)

    FM = Dot(1)([model_uer, model_item])  # 点积运算
    FM = Dense(1, use_bias=True, kernel_regularizer=regularizers.l2(0.01))(FM)


    # Deep_user = Embedding(max_user + 1, k, input_length=1, )(input_user)
    # Deep_user = BatchNormalization(epsilon=0.001, momentum=0.99, axis=-1)(Deep_user)
    # Deep_user = Dense(50, activation="relu", use_bias=True, kernel_regularizer=regularizers.l2(0.005))(Deep_user)  # 激活函数

    # Deep_item = Embedding(max_user + 1, k, input_length=1, )(input_item)
    # Deep_item = BatchNormalization(epsilon=0.001, momentum=0.99, axis=-1)(Deep_item)
    # Deep_item = Dense(50, activation="relu", use_bias=True, kernel_regularizer=regularizers.l2(0.005))(Deep_item)  # 激活函数

    Deep_model = Concatenate()([model_uer, model_item])
    Deep_model = Dense(64, activation="relu", use_bias=True, kernel_regularizer=regularizers.l2(0.005))(Deep_model)
    Deep_model = Dense(32, activation="relu", use_bias=True, kernel_regularizer=regularizers.l2(0.000))(Deep_model)
    Deep_model = Dense(16, activation="relu", use_bias=True, kernel_regularizer=regularizers.l2(0.000))(Deep_model)
    Deep_model = Dense(8, activation="relu", use_bias=True, kernel_regularizer=regularizers.l2(0.005))(Deep_model)
    Deep_model = Dense(4, activation="relu", use_bias=True, kernel_regularizer=regularizers.l2(0.005))(Deep_model)
    Deep_model = Dense(1, activation="relu", use_bias=True, kernel_regularizer=regularizers.l2(0.000))(Deep_model)

    DeepFM = Add()([Deep_model, FM])
    # DeepFM = Dense(1, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(0.01))(DeepFM)

    model = Model(inputs=[input_user, input_item], outputs=DeepFM)
    model.compile(loss=root_mean_squared_error, optimizer=optimizers.Adam(lr=0.0005), metrics=['mae'])
    # model.summary()
    # tf.keras.utils.plot_model(model, "DeepFM.png", show_shapes=True)
    return model

model = Recmand_model(max_user, max_movie, 50)

train_user = rating['userId'].values
train_movie = rating['movieId'].values
train_x = [train_user, train_movie]
train_y = rating["rating"].values

history = model.fit(train_x, train_y, batch_size=256, epochs=12, verbose=1, validation_split=0.2)

