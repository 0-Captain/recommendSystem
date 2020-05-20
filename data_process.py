import pandas as pd
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Concatenate, Input, Reshape, Dropout, Flatten, Dot
from tensorflow.keras import Model
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
users = pd.read_table(dataPath + 'users.dat', sep='::', header=None, names=usersTitle, engine='python')
genderMap = {'F': 0, 'M': 1}
users['genderIndex'] = users['gender'].map(genderMap)
ageMap = {val: index for index, val in enumerate(set(users['age']))}
users['ageIndex'] = users['age'].map(ageMap)



# 读取ratings数据:

ratingsTitle = ['userId', 'movieId', 'ratings', 'timestamps']
ratings = pd.read_table(dataPath + 'ratings.dat', sep='::', header=None, names=ratingsTitle, engine='python')
ratings['movieId'] = ratings['movieId'].map(moviesIdMap)

data = pd.merge(pd.merge(ratings, users), movies)

dataInfo = pd.DataFrame({
    'len(movies): ': [movies['movieId'].max()],
    'len(users): ': [users['userId'].max()],
    'len(ratings): ': [len(ratings['ratings'].values)],
    '矩阵填充率: ': [len(ratings['ratings'].values)/(movies['movieId'].max()*users['userId'].max())]
})
print(dataInfo.T)

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

# -----------user部分
usersIdInput = Input(shape=(1, ), dtype="int32", name='userId')
usersIdEmbedding = Embedding(usersIdInputDim + 1, 32, input_length=1)(usersIdInput)
# usersIdEmbedding = Reshape((32, ))(usersIdEmbedding)
usersIdDense1 = Dense(64, activation="relu")(usersIdEmbedding)
usersIdDropOut = Dropout(rate=0.4)(usersIdDense1)

usersGenderInput = Input(shape=(1, ), dtype="int32", name='userGender')
usersGenderEmbedding = Embedding(2+1, 2, input_length=1)(usersGenderInput)
# usersGenderEmbedding = Reshape((2, ))(usersGenderEmbedding)
usersGenderDense1 = Dense(64, activation="relu")(usersGenderEmbedding)
usersGenderDropout = Dropout(rate=0.4)(usersGenderDense1)

usersAgeInput = Input(shape=(1, ), dtype="int32", name='userAge')
usersAgeEmbedding = Embedding(7+1, 4, input_length=1)(usersAgeInput)
# usersAgeEmbedding = Reshape((4, ))(usersAgeEmbedding)
usersAgeDense1 = Dense(64, activation="relu")(usersAgeEmbedding)
usersAgeDropout = Dropout(rate=0.4)(usersAgeDense1)

usersJobIdInput = Input(shape=(1, ), dtype="int32", name='userJob')
usersJobIdEmbedding = Embedding(21+1, 5, input_length=1)(usersJobIdInput)
# usersJobIdEmbedding = Reshape((5, ))(usersJobIdEmbedding)
usersJobIdDense1 = Dense(64, activation="relu")(usersJobIdEmbedding)
usersJobIdDropout = Dropout(rate=0.4)(usersJobIdDense1)

userModel = Concatenate(axis=1)([usersIdDropOut, usersGenderDropout, usersAgeDropout, usersJobIdDropout])
userDense1 = Dense(64, activation='relu')(userModel)
# userDense1 = Reshape((-1, 64))(userDense1)

# ------------movie部分
moviesIdInput = Input(shape=(1,), dtype="int32", name='movieId')
moviesIdEmbedding = Embedding(moviesIdInputDim+1, 32, input_length=1)(moviesIdInput)
moviesIdDense1 = Dense(16, activation='relu')(moviesIdEmbedding)
moviesIdDropout = Dropout(rate=0.4)(moviesIdDense1)

moviesGenresInput = Input(shape=(moviesGenresInputDim, ), dtype="float32", name='movieGenres')
moviesGenresReshape = Reshape((1, moviesGenresInputDim))(moviesGenresInput)
# moviesGenresEmbedding = Embedding(moviesGenresInputDim+1, 16, input_length=moviesGenresInputDim)(moviesGenresInput)
moviesGenresDense1 = Dense(16, activation='relu')(moviesGenresReshape)
moviesGenresDropout = Dropout(rate=0.4)(moviesGenresDense1)

moviesTitleInput = Input(shape=(15,), dtype="int32", name='movieTitle')
moviesTitleEmbedding = Embedding(moviesTitleInputDim+1, 4, input_length=15)(moviesTitleInput)
moviesTitleDense1 = Dense(16, activation='relu')(moviesTitleEmbedding)
moviesTitleDropout = Dropout(rate=0.4)(moviesTitleDense1)

movieModel = Concatenate(axis=1)([moviesIdDropout, moviesGenresDropout, moviesTitleDropout])
movieDense1 = Dense(64, activation='relu')(movieModel)

# -----------combine
combineModel = Dot(2)([userDense1, movieDense1])
# combineModelDense1 = Dense(64, activation='relu')(combineModel)
# combineModelDense2 = Dense(16, activation='relu')(combineModelDense1)
combineModelDense3 = Dense(1, activation='relu')(combineModel)
combineModelReshape = Flatten()(combineModelDense3)
out = Dense(1, activation='relu')(combineModelReshape)


model = Model(inputs=[usersIdInput, usersGenderInput, usersAgeInput, usersJobIdInput, moviesIdInput, moviesGenresInput, ], outputs=out)

model.compile(loss=root_mean_squared_error, optimizer='Adam')
model.summary()
tf.keras.utils.plot_model(model, "my_model.png", show_shapes=True)

model.fit([usersId, usersGender, usersAge, usersJobId, moviesId,  moviesGenres, ], userRatings, batch_size=256, epochs=1, validation_split=0.3)