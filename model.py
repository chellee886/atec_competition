# coding:utf8
import codecs
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Bidirectional, LSTM, Embedding, Dropout, Dot, Multiply, Concatenate, Subtract, Softmax, Add, TimeDistributed
from keras.layers.core import Lambda
from keras.layers.merge import concatenate, add, multiply
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
import jieba



np.random.seed(0)

MAX_SEQUENCE_LENGTH = 30
MIN_WORD_OCCURRENCE = 5    # 10
EMBEDDING_DIM = 300
NUM_FOLDS = 10
BATCH_SIZE = 512    # 100
EMBEDDING_FILE = './dict/sgns.baidubaike.bigram-char'

def stopwordslist(filepath):
    stopwords = [line.strip() for line in codecs.open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords
STOP_WORDS = stopwordslist('./dict/stop_word')

replace_word_index = {}
with codecs.open('./dict/repalce_words.csv', 'r', encoding='utf-8') as f:
    for line in f:
        replace_word_list = line.replace('\n', '').replace(',', ' ').split()
        replace_word_index[replace_word_list[0]] = replace_word_list[1]

def preprocess(string):
    string = string.replace(u"花被", u"花呗").replace(u"ma", u"吗").replace(u"花唄", u"花呗")\
        .replace(u"花坝", u"花呗").replace(u"花多上", u"花多少").replace(u"花百", u"花呗").replace(u"花背", u"花呗")\
        .replace(u"节清", u"结清").replace(u"花花", u"花").replace(u"花能", u"花呗能")\
        .replace(u"花裁", u"花").replace(u"花贝", u"花呗").replace(u"花能", u"花呗能")\
        .replace(u"蚂蚊", u"蚂蚁").replace(u"蚂蚱", u"蚂蚁").replace(u"蚂议", u"蚂蚁")\
        .replace(u"螞蟻", u"蚂蚁").replace(u"借唄", u"借呗").replace(u"发呗", u"花呗")\
        .replace(u"结呗", u"借呗").replace(u'戒备', u'借呗').replace(u'芝麻', u'').replace(u'压金', u'押金')

    for key, value in replace_word_index.items():
        string = string.replace(key, value)
    return string

def fenci(sentence):
    seg_list = jieba.cut(sentence, cut_all=False)
    outseg = ''
    for word in seg_list:
        if len(word) == 1:
            outseg += word + ' '
        else:
            for character in word:
                outseg += character + ' '
            outseg += word + ' '
    return outseg


def get_embedding():
    embedding_index = {}
    f = codecs.open(EMBEDDING_FILE, mode='r', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        if len(values) == EMBEDDING_DIM + 1 and word in top_words:
            coefs = np.asarray(values[1:], dtype="float32")
            embedding_index[word] = coefs
    f.close()
    return embedding_index

def prepare(q):
    new_q = []
    for w in q.split():
        if w in top_words:
            new_q.append(w)
    new = ' '.join(new_q)
    return new

def extract_features(df):
    q1s = np.array([''] * len(df), dtype=object)
    q2s = np.array([''] * len(df), dtype=object)

    for i, (q1, q2) in enumerate(list(zip(df['question1'], df['question2']))):
        q1s[i] = q1
        q2s[i] = q2
    return q1s, q2s


train1 = pd.read_csv('./data/atec_nlp_sim_train_add.csv', sep='\t', header=None, encoding='utf-8')
train1.columns = ['id', 'question1', 'question2', 'is_duplicate']

train2 = pd.read_csv('./data/atec_nlp_sim_train.csv', sep='\t', header=None, encoding='utf-8')
train2.columns = ['id', 'question1', 'question2', 'is_duplicate']

train = train1.append(train2)

train.iloc[:, 1] = train.iloc[:, 1].apply(lambda x: preprocess(x))
train.iloc[:, 2] = train.iloc[:, 2].apply(lambda x: preprocess(x))

train.iloc[:, 1] = train.iloc[:, 1].apply(lambda x: fenci(x))
train.iloc[:, 2] = train.iloc[:, 2].apply(lambda x: fenci(x))
print(train.iloc[:, 1])

print("Creating the vocabulary of words occurred more than", MIN_WORD_OCCURRENCE)
all_questions = pd.Series(train.iloc[:, 1].tolist() + train.iloc[:, 2].tolist()).unique()
vectorizer = CountVectorizer(lowercase=False, token_pattern='\S+', min_df=MIN_WORD_OCCURRENCE)
vectorizer.fit(all_questions)
top_words = set(vectorizer.vocabulary_.keys())

embeddings_index = get_embedding()
top_words = embeddings_index.keys()

print("Train questions are being prepared for LSTM...")
q1s_train, q2s_train = extract_features(train)

tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(np.append(q1s_train, q2s_train))
word_index = tokenizer.word_index

data_1 = pad_sequences(tokenizer.texts_to_sequences(q1s_train), maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(tokenizer.texts_to_sequences(q2s_train), maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(train.iloc[:, 3])
nb_words = len(word_index) + 1
print nb_words
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True)
model_count = 0

for idx_train, idx_val in skf.split(train.iloc[:, 3], train.iloc[:, 3]):
    print("MODEL:", model_count)
    data_1_train = data_1[idx_train]
    data_2_train = data_2[idx_train]
    labels_train = labels[idx_train]

    data_1_val = data_1[idx_val]
    data_2_val = data_2[idx_val]
    labels_val = labels[idx_val]

    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer1 = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.5))
    lstm_layer2 = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.5))
    Td1 = TimeDistributed(Dense(128, activation='relu'))
    Td2 = TimeDistributed(Dense(128, activation='relu'))

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
    embedded_sequence_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer1(embedded_sequence_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
    embedded_sequence_2 = embedding_layer(sequence_2_input)
    x2 = lstm_layer1(embedded_sequence_2)

    e = Dot(axes=2)([x1, x2])
    e1 = Softmax(axis=2)(e)
    e2 = Softmax(axis=1)(e)
    e1 = Lambda(K.expand_dims, arguments={'axis': 3})(e1)
    e2 = Lambda(K.expand_dims, arguments={'axis': 3})(e2)

    _x1 = Lambda(K.expand_dims, arguments={'axis': 1})(x2)
    _x1 = Multiply()([e1, _x1])
    _x1 = Lambda(K.sum, arguments={'axis': 2})(_x1)
    _x2 = Lambda(K.expand_dims, arguments={'axis': 2})(x1)
    _x2 = Multiply()([e2, _x2])
    _x2 = Lambda(K.sum, arguments={'axis': 1})(_x2)

    # _x1 = TimeDistributed(Dense(256, activation='relu'), input_shape=(MAX_SEQUENCE_LENGTH, 256))(_x1)
    # _x2 = TimeDistributed(Dense(256, activation='relu'), input_shape=(MAX_SEQUENCE_LENGTH, 256))(_x2)

    m1 = Concatenate()([x1, _x1, Subtract()([x1, _x1]), Multiply()([x1, _x1])])
    m2 = Concatenate()([x2, _x2, Subtract()([x2, _x2]), Multiply()([x2, _x2])])

    y1 = lstm_layer2(m1)
    y2 = lstm_layer2(m2)

    mx1 = Lambda(K.max, arguments={'axis': 1})(y1)
    av1 = Lambda(K.mean, arguments={'axis': 1})(y1)
    mx2 = Lambda(K.max, arguments={'axis': 1})(y2)
    av2 = Lambda(K.mean, arguments={'axis': 1})(y2)
    y = Concatenate()([av1, mx1, av2, mx2])

    y3 = BatchNormalization()(y)
    y3 = Dense(512, activation='relu')(y3)
    # y3 = Dropout(0.5)(y3)

    y3 = BatchNormalization()(y3)
    y3 = Dense(1024, activation='relu')(y3)
    # y3 = Dropout(0.5)(y3)

    y3 = BatchNormalization()(y3)
    y3 = Add()([y, y3])
    # y4 = Dropout(0.5)(y3)

    y4 = BatchNormalization()(y3)
    y4 = Dense(512, activation='relu')(y4)
    # y4 = Dropout(0.5)(y4)

    y4 = BatchNormalization()(y4)
    y4 = Dense(1024, activation='relu')(y4)
    # y4 = Dropout(0.5)(y4)

    y4 = BatchNormalization()(y4)
    y4 = Add()([y3, y4])

    out = Dense(1, activation='sigmoid')(y4)

    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=out)
    model.compile(loss="binary_crossentropy", optimizer='nadam')
    early_stoping = EarlyStopping(monitor='val_loss', patience=5)
    best_model_path = "./model/" + "best_model" + str(model_count) + ".h5"
    model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True,
                                       save_weights_only=True)
    hist = model.fit([data_1_train, data_2_train], labels_train,
                     validation_data=([data_1_val, data_2_val], labels_val),
                     epochs=10, batch_size=BATCH_SIZE,shuffle=True,
                     callbacks=[early_stoping, model_checkpoint])
    print('validation loss:', min(hist.history['val_loss']))

    model_count += 1
