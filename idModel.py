import pandas as pd
import numpy as np
from tensorflow.keras import backend as K
import seaborn as sns
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 自定义损失函数rmse
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

rnames = ['userId', 'movieId', 'rating', 'timestamp']
rating = pd.read_csv("../ml-1m/ratings.dat", sep="::", names=rnames,header=None, engine='python')
rating['rating'].astype = 'float32'
rating['userId'].astype = 'float32'
rating['movieId'].astype = 'float32'
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
from tensorflow.keras.layers import Embedding, Reshape, Input, Dot, Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers, optimizers


# def Recmand_model(max_user, max_movie, k):
#     input_uer = Input(shape=[None, ], dtype="int32")
#     model_uer = Embedding(max_user + 1, k, input_length=1, embeddings_regularizer=regularizers.l2(0.001))(input_uer)
#     model_uer = Reshape((k,))(model_uer)
#
#     input_movie = Input(shape=[None, ], dtype="int32")
#     model_movie = Embedding(max_movie + 1, k, input_length=1, embeddings_regularizer=regularizers.l2(0.001))(
#         input_movie)
#     model_movie = Reshape((k,))(model_movie)
#
#     out = Dot(1)([model_uer, model_movie])
#     out = Dense(11, activation='softmax')(out)
#     model = Model(inputs=[input_uer, input_movie], outputs=out)
#     model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])
#     model.summary()
#     return model


def Recmand_model(max_user, max_item, k):
    input_uer = Input(shape=(1, ), dtype="float32")
    model_uer = Embedding(max_user + 1, k, input_length=1,)(input_uer)
    model_uer = BatchNormalization(epsilon=0.001, momentum=0.99, axis=-1)(model_uer)
    model_uer = Dense(k, activation="relu", use_bias=True,)(model_uer)  # 激活函数
    # model_uer = Dropout(0.1)(model_uer)  # Dropout 随机删去一些节点，防止过拟合
    model_uer = Dense(50, activation="relu", use_bias=True, kernel_regularizer=regularizers.l2(0.005))(model_uer)  # 激活函数
    # model_uer = Dense(50, activation="relu")(model_uer)  # 激活函数
    model_uer = Reshape((-1,))(model_uer)

    input_item = Input(shape=(1,), dtype="float32")
    model_item = Embedding(max_item + 1, k, input_length=1,)(input_item)
    model_item = BatchNormalization(epsilon=0.001, momentum=0.99, axis=-1)(model_item)
    model_item = Dense(k, activation="relu", use_bias=True,)(model_item)
    # model_item = Dropout(0.1)(model_item)
    model_item = Dense(50, activation="relu", use_bias=True, kernel_regularizer=regularizers.l2(0.005))(model_item)  # 激活函数
    # model_item = Dense(50, activation="relu")(model_item)  # 激活函数
    model_item = Reshape((-1,))(model_item)

    out = Dot(1)([model_uer, model_item])  # 点积运算

    model = Model(inputs=[input_uer, input_item], outputs=out)
    model.compile(loss=root_mean_squared_error, optimizer=optimizers.Adam(lr=0.0005), metrics=['mae'])
    # model.summary()
    return model

model = Recmand_model(max_user, max_movie, 100)

train_user = rating['userId'].values
train_movie = rating['movieId'].values
train_x = [train_user, train_movie]
train_y = rating["rating"].values
# train_y = rating["rating"].values * 2
# train_y = utils.to_categorical(train_y, num_classes=11)  # one-hot 0~5间隔为0.5，总共有11类

history = model.fit(train_x, train_y, batch_size=256, epochs=3, verbose=1, validation_split=0.2)
#
# def plot_history(history):
#   hist = pd.DataFrame(history.history)
#   hist['epoch'] = history.epoch
#
#   plt.figure()
#   plt.xlabel('Epoch')
#   plt.ylabel('RMSE [rating]')
#   plt.plot(hist['epoch'], hist['loss'],
#            label='Train Error')
#   plt.plot(hist['epoch'], hist['val_loss'],
#            label = 'Val Error')
#   plt.ylim([0,2])
#   plt.legend()
#
#   plt.figure()
#   plt.ylim(0, 2)
#   plt.xlabel('Epoch')
#   plt.ylabel('MAE [rating]')
#   plt.plot(hist['epoch'], hist['mae'],
#            label='Train Error')
#   plt.plot(hist['epoch'], hist['val_mae'],
#            label = 'Val Error')
#   plt.ylim([0,1.2])
#   plt.legend()
#   plt.show()
#
#
# plot_history(history)

