import pandas as pd
import numpy as np
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Concatenate, Input, Reshape, Dropout, Flatten, Dot, BatchNormalization, Lambda, Add
from tensorflow.keras import Model, regularizers, optimizers, constraints
from tensorflow.keras import backend as K

# 自定义损失函数rmse
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 240)
pd.set_option('float_format', lambda x: '%.4f' % x)

dataPath = '../ml-1m/'


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
users = pd.read_csv(dataPath + 'users.dat', sep='::', header=None, names=usersTitle, engine='python')
genderMap = {'F': 0, 'M': 1}
users['genderIndex'] = users['gender'].map(genderMap)
ageMap = {val: index for index, val in enumerate(set(users['age']))}
users['ageIndex'] = users['age'].map(ageMap)


# 读取ratings数据:

ratingsTitle = ['userId', 'movieId', 'ratings', 'timestamps']
ratings = pd.read_csv(dataPath + 'ratings.dat', sep='::', header=None, names=ratingsTitle, engine='python')
ratings['movieId'] = ratings['movieId'].map(moviesIdMap)

data = pd.merge(pd.merge(ratings, users), movies)

# 输出
dataInfo = pd.DataFrame({
    'len(movies): ': [movies['movieId'].max()],
    'len(users): ': [users['userId'].max()],
    'len(ratings): ': [len(ratings['ratings'].values)],
    '矩阵填充率: ': [len(ratings['ratings'].values)/(movies['movieId'].max()*users['userId'].max())]
})
print(dataInfo.T)

# 划分train与dev
trainData = data.sample(frac=0.8, random_state=0)
devData = data.drop(trainData.index)

# print(trainData.info(), testData.info())
# 从总的table中获取单列数据
userRatings = trainData['ratings'].values

usersId = trainData['userId'].values
usersGender = trainData["genderIndex"].values
usersAge = trainData['ageIndex'].values
usersJobId = trainData['jobId'].values

moviesId = trainData['movieId'].values
moviesGenres = np.array([value for value in trainData['genresEncoding'].values])
moviesTitle = np.array([list(value) for value in trainData['titleEncoding'].values])

usersIdInputDim = data['userId'].values.max()
moviesIdInputDim = data['movieId'].values.max()
moviesGenresInputDim = len(moviesGenresSet)
moviesTitleInputDim = len(moviesTitleSet)

def sqrt(x):
    x = x + 0.00000001
    abs = K.abs(x)
    sign = x / abs
    s = K.sqrt(abs)
    result = s * sign
    # result = K.reshape(result, (1, -1))
    return K.tile(result, [1, 50])

def createModel(k):
    input_uid = Input(shape=(1,))
    model_uid = Embedding(usersIdInputDim + 1, k, input_length=1, )(input_uid)
    model_uid = BatchNormalization(epsilon=0.001, momentum=0.99, axis=-1)(model_uid)
    model_uid = Dense(k, activation="relu", use_bias=True, )(model_uid)  # 激活函数
    model_uid = Dense(50, activation="relu", use_bias=True, kernel_regularizer=regularizers.l2(0.005))(model_uid)  # 激活函数
    model_uid = Flatten()(model_uid)

    user_gender_input = Input(shape=(1,))
    # model_gender = Embedding(3, 2, )(user_gender_input)
    # embeddings_constraint=constraints.MinMaxNorm(min_value=0.000001, max_value=0.29)
    model_gender = BatchNormalization(epsilon=0.001, momentum=0.99, axis=-1)(user_gender_input)
    model_gender = Dense(1, activation="relu", use_bias=True, kernel_regularizer=regularizers.l2(0.1))(model_gender)
    # model_gender = Flatten()(model_gender)
    model_gender = Lambda(sqrt)(model_gender)

    model_user = Add()([model_uid, model_gender])

    input_iid = Input(shape=(1,))
    model_iid = Embedding(moviesIdInputDim + 1, k, input_length=1, )(input_iid)
    model_iid = BatchNormalization(epsilon=0.001, momentum=0.99, axis=-1)(model_iid)
    model_iid = Dense(k, activation="relu", use_bias=True, )(model_iid)
    model_iid = Dense(50, activation="relu", use_bias=True, kernel_regularizer=regularizers.l2(0.005))(model_iid)  # 激活函数
    model_item = Flatten()(model_iid)

    out = Dot(1)([model_user, model_item])  # 点积运算

    model = Model(inputs=[input_uid, input_iid, user_gender_input], outputs=out)
    model.compile(loss=root_mean_squared_error, optimizer=optimizers.Adam(lr=0.0005), metrics=['mae'])

    model2 = Model(inputs=[input_uid, input_iid, user_gender_input], outputs=out)
    return model


model = createModel(50)


train_x = [usersId, moviesId, usersGender]
train_y = userRatings

history = model.fit(train_x, train_y, batch_size=256, epochs=8, verbose=1, validation_split=0.2)