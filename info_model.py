import pandas as pd
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# import seaborn as sns
# import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Concatenate, Input, Reshape, Dropout, Flatten, Dot, BatchNormalization, Add
from tensorflow.keras import Model, regularizers, optimizers
from tensorflow.keras import backend as K

# 自定义损失函数rmse
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 240)
pd.set_option('float_format', lambda x: '%.4f' % x)

dataPath = '../ml-1m/'
dropoutRate = 0.2
idEmbeddingDimension = 100

def genres_multi_hot(genre_int_map):
    """
    电影类型使用multi-hot编码
    :param genre_int_map:genre到数字的映射字典
    :return function
    """

    def helper(genres):
        genre_int_list = [genre_int_map[genre] for genre in genres.split('|')]
        multi_hot = np.zeros(len(genre_int_map), dtype='float32')
        multi_hot[genre_int_list] = 1
        return multi_hot

    return helper


def title_encode(word_int_map):
    """
    :param word_int_map:word到数字的映射字段
    :return:
    """

    def helper(title):
        title_words = [word_int_map[word] for word in title.split()]
        if len(title_words) > 15:
            return np.array(title_words[:15])
        else:
            title_vector = np.zeros(15, dtype='int32')
            title_vector[:len(title_words)] = title_words
            return np.array(title_vector, dtype='int32')

    return helper


# 读取movies数据:

movieTitle = ['movieId', 'title', 'genres']
movies = pd.read_csv(dataPath + 'movies.dat', sep='::', header=None, names=movieTitle, engine='python')
# 生成moviesGenresMap, 对genres进行multi_hot encoding
moviesGenresSet = set()
for val in movies['genres'].str.split('|'):
    moviesGenresSet.update(val)
moviesGenresMap = {val: index for index, val in enumerate(moviesGenresSet)}
movies['genresEncoding'] = movies['genres'].map(genres_multi_hot(moviesGenresMap))
# title:将所有单词取出来生成一个map，然后用一个长度为10的向量表示，向量的每一位表示对应位的单词在set中的索引号，没有则用0代替
moviesTitleSet = set()
for val in movies['title'].str.split():
    moviesTitleSet.update(val)
moviesTitleMap = {val: index for index, val in enumerate(moviesTitleSet, start=1)}
movies['titleEncoding'] = movies['title'].map(title_encode(moviesTitleMap))
# id 将不连续的id转换为连续的id，减少计算量
moviesIdMap = {value: index+1 for index, value in enumerate(movies['movieId'].values)}
# print(movies['movieId'].values, moviesIdMap)
movies['movieId'] = movies['movieId'].map(moviesIdMap)

# x=np.reshape(tf.keras.preprocessing.sequence.pad_sequences(movies['genresEncoding'],maxlen=18), (3883, 18)

# 读取user数据:

usersTitle = ['userId', 'gender', 'age', 'jobId', 'zip-code']
users = pd.read_table(dataPath + 'users.dat', sep='::', header=None, names=usersTitle, engine='python')
genderMap = {'F': 0, 'M': 1}
users['genderIndex'] = users['gender'].map(genderMap)
ageMap = {val: index for index, val in enumerate(set(users['age']))}
users['ageIndex'] = users['age'].map(ageMap)


# 读取ratings数据:

ratingsTitle = ['userId', 'movieId', 'ratings', 'timestamps']
ratings = pd.read_table(dataPath + 'ratings.dat', sep='::', header=None, names=ratingsTitle, engine='python')
ratings['movieId'] = ratings['movieId'].map(moviesIdMap)

# data = pd.merge(pd.merge(ratings, movies, sort=False, on='movieId'), users, sort=False, on='userId')
data = pd.merge(pd.merge(ratings, users, sort=False, on='userId'), movies, sort=False, on='movieId')
train_data = data.sample(frac=0.8, random_state=0)
test_data = data.drop(train_data.index)


dataInfo = pd.DataFrame({
    'len(movies): ': [movies['movieId'].max()],
    'len(users): ': [users['userId'].max()],
    'len(ratings): ': [len(ratings['ratings'].values)],
    '矩阵填充率: ': [len(ratings['ratings'].values)/(movies['movieId'].max()*users['userId'].max())]
})
print(dataInfo.T)
print(data.info())
print(data.describe())

print(len(ratings['ratings'].values), len(data['ratings'].values))

# sns.pairplot(train_data[['genderIndex', 'ageIndex', 'jobId']], diag_kind="kde")
# plt.show()



# 开始搭建模型
userRatings = data['ratings'].values

usersId = data['userId'].values
usersGender = data["genderIndex"].values
usersAge = data['ageIndex'].values
usersJobId = data['jobId'].values

moviesId = data['movieId'].values
moviesGenres = np.array([value for value in data['genresEncoding'].values])
moviesTitle = np.array([list(value) for value in data['titleEncoding'].values])

usersIdInputDim = data['userId'].values.max()
moviesIdInputDim = data['movieId'].values.max()
moviesGenresInputDim = len(moviesGenresSet)
moviesTitleInputDim = len(moviesTitleSet)


idDropoutRate = 0.1
# ----- id process
# usersIdInput = Input(shape=(1, ), dtype="int32", name='userId')
# usersIdModel = Embedding(usersIdInputDim + 1, idEmbeddingDimension, input_length=1,
#                           embeddings_regularizer=regularizers.l2(0.001))(usersIdInput)
# usersIdModel = BatchNormalization(epsilon=0.001, momentum=0.99, axis=-1)(usersIdModel)
# usersIdModel = Dense(100, activation="relu", use_bias=True, kernel_regularizer=regularizers.l2(0.01))(usersIdModel)
# usersIdModel = Dropout(rate=idDropoutRate)(usersIdModel)
# usersIdModel = Reshape((idEmbeddingDimension,))(usersIdModel)
#
# moviesIdInput = Input(shape=(1,), dtype="int32", name='movieId')
# moviesIdModel = Embedding(moviesIdInputDim+1, idEmbeddingDimension, input_length=1,
#                           embeddings_regularizer=regularizers.l2(0.001))(moviesIdInput)
# moviesIdModel = BatchNormalization(epsilon=0.001, momentum=0.99, axis=-1)(moviesIdModel)
# moviesIdModel = Dense(100, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(0.01))(moviesIdModel)
# moviesIdModel = Dropout(rate=idDropoutRate)(moviesIdModel)
# moviesIdModel = Reshape((idEmbeddingDimension,))(moviesIdModel)
#
# idMolde = Dot(1)([usersIdModel, moviesIdModel])





denseDimension = 16
# -----------user部分
usersGenderInput = Input(shape=(1, ), dtype="int32", name='userGender')
usersGenderModel = Embedding(2+1, 2, input_length=1,
                          embeddings_regularizer=regularizers.l2(0.0001))(usersGenderInput)
usersGenderModel = BatchNormalization(epsilon=0.001, momentum=0.99, axis=-1)(usersGenderModel)
usersGenderModel = Dense(denseDimension, activation="relu", use_bias=True,kernel_regularizer=regularizers.l2(0.001))(usersGenderModel)
# usersGenderModel = Dropout(rate=dropoutRate)(usersGenderModel)
usersGenderModel = Dense(denseDimension, activation="relu", use_bias=True,)(usersGenderModel)

usersAgeInput = Input(shape=(1, ), dtype="int32", name='userAge')
usersAgeModel = Embedding(7+1, 4, input_length=1,
                          embeddings_regularizer=regularizers.l2(0.0001))(usersAgeInput)
usersAgeModel = BatchNormalization(epsilon=0.001, momentum=0.99, axis=-1)(usersAgeModel)
usersAgeModel = Dense(denseDimension, activation="relu", use_bias=True,kernel_regularizer=regularizers.l2(0.001))(usersAgeModel)
# usersAgeModel = Dropout(rate=dropoutRate)(usersAgeModel)
usersAgeModel = Dense(denseDimension, activation="relu", use_bias=True,)(usersAgeModel)

usersJobIdInput = Input(shape=(1, ), dtype="int32", name='userJob')
usersJobIdModel = Embedding(21+1, 5, input_length=1,
                          embeddings_regularizer=regularizers.l2(0.001))(usersJobIdInput)
usersJobIdModel = BatchNormalization(epsilon=0.001, momentum=0.99, axis=-1)(usersJobIdModel)
usersJobIdModel = Dense(denseDimension, activation="relu", use_bias=True,kernel_regularizer=regularizers.l2(0.001))(usersJobIdModel)
# usersJobIdModel = Dropout(rate=dropoutRate)(usersJobIdModel)
usersJobIdModel = Dense(denseDimension, activation="relu", use_bias=True,)(usersJobIdModel)

userModel = Add()([usersGenderModel, usersAgeModel, usersJobIdModel])
userModel = Flatten()(userModel)
userDense1 = Dense(16, activation='relu', kernel_regularizer = regularizers.l2(0.001))(userModel)

# ------------movie部分
moviesGenresInput = Input(shape=(moviesGenresInputDim, ), dtype="float32", name='movieGenres')
# moviesGenresModel = Reshape((1, moviesGenresInputDim))(moviesGenresInput)
# moviesGenresEmbedding = Embedding(moviesGenresInputDim+1, 16, input_length=moviesGenresInputDim)(moviesGenresInput)
moviesGenresModel = BatchNormalization(epsilon=0.001, momentum=0.99, axis=-1)(moviesGenresInput)
moviesGenresModel = Dense(16, activation='relu', use_bias=True,kernel_regularizer=regularizers.l2(0.001))(moviesGenresModel)
moviesGenresModel = Dropout(rate=dropoutRate)(moviesGenresModel)
moviesGenresModel = Dense(16, activation='relu', use_bias=True,)(moviesGenresModel)

# moviesTitleInput = Input(shape=(15,), dtype="int32", name='movieTitle')
# moviesTitleEmbedding = Embedding(moviesTitleInputDim+1, 4, input_length=15)(moviesTitleInput)
# moviesTitleDense1 = Dense(16, activation='relu')(moviesTitleEmbedding)
# moviesTitleDropout = Dropout(rate=dropoutRate)(moviesTitleDense1)

# movieModel = Concatenate(axis=1)([moviesIdDropout, moviesGenresDropout, ])
# movieDense1 = Dense(64, activation='relu', kernel_regularizer = regularizers.l2(0.001))(movieModel)

# -----------combine
combineModel = Dot(1)([userDense1, moviesGenresModel])
# combineModelDense1 = Dense(64, activation='relu')(combineModel)
# combineModelDense2 = Dense(16, activation='relu')(combineModelDense1)
combineModelDense3 = Dense(1, activation='relu')(combineModel)
combineModelReshape = Flatten()(combineModelDense3)
out = Dense(1, activation='relu')(combineModelReshape)

# combineMode2 = Dot(1)([combineModelDense4, idMolde])
# out = Dense(1, activation='relu', kernel_regularizer=regularizers.l2(0.005))(combineModelReshape)

model = Model(inputs=[usersGenderInput,usersAgeInput, usersJobIdInput,moviesGenresInput,], outputs=out)

model.compile(loss=root_mean_squared_error, optimizer='adam', metrics=['mae'])
# model.summary()
# tf.keras.utils.plot_model(model, "my_model.png", show_shapes=True)

history = model.fit([usersGender, usersAge, usersJobId, moviesGenres,], userRatings, batch_size=256, epochs=3, verbose=1, validation_split=0.2)

# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# hist.tail()

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
#   plt.ylim([0,5])
#   plt.legend()
#
#   plt.figure()
#   plt.xlabel('Epoch')
#   plt.ylabel('MAE [rating]')
#   plt.plot(hist['epoch'], hist['mae'],
#            label='Train Error')
#   plt.plot(hist['epoch'], hist['val_mae'],
#            label = 'Val Error')
#   plt.ylim([0,20])
#   plt.legend()
#   plt.show()
#
#
# plot_history(history)