from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Dropout, LSTM, GRU, Flatten, Embedding, RepeatVector
from keras.layers import Merge, Reshape, RepeatVector, BatchNormalization, Lambda, TimeDistributed, Permute
from keras.layers import GlobalMaxPooling2D, Convolution2D, merge, Bidirectional
from keras.regularizers import l2
from keras.optimizers import *

from keras import backend as K
#import tensorflow as tf


# Baseline model
def Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate):
    print("Creating text model...")
    model = Sequential()
    model.add(Embedding(num_words, embedding_dim,
                        weights=[embedding_matrix], input_length=seq_length, trainable=False))
    model.add(LSTM(output_dim=512, return_sequences=True, input_shape=(seq_length, embedding_dim)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(output_dim=512, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1024, activation='tanh'))
    return model


def img_model(dropout_rate):
    print("Creating image model...")
    model = Sequential()
    model.add(Dense(1024, input_dim=4096, activation='tanh'))
    return model


def vqa_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    vgg_model = img_model(dropout_rate)
    lstm_model = Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate)
    print("Merging final model...")
    fc_model = Sequential()
    fc_model.add(Merge([vgg_model, lstm_model], mode='mul'))
    fc_model.add(Dropout(dropout_rate))
    fc_model.add(Dense(1000, activation='tanh'))
    fc_model.add(Dropout(dropout_rate))
    fc_model.add(Dense(num_classes, activation='softmax'))
    fc_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                     metrics=['accuracy'])
    return fc_model


# Model used last semester
def Word2VecModel_nofc(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate):
    print("Creating text model...")
    model = Sequential()
    model.add(Embedding(num_words, embedding_dim,
                        weights=[embedding_matrix], input_length=seq_length, trainable=False))
    model.add(LSTM(output_dim=512, return_sequences=True, input_shape=(seq_length, embedding_dim)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(output_dim=512, return_sequences=True, input_shape=(seq_length, embedding_dim)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(output_dim=512, return_sequences=False))
    return model


def img_model_nofc(dropout_rate):
    print("Creating image model...")
    model = Sequential()
    model.add(Reshape(target_shape=(4096,), input_shape=(4096,)))
    return model


def vqa_model_1(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    vgg_model = img_model_nofc(dropout_rate)
    lstm_model = Word2VecModel_nofc(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate)
    print("Merging final model...")
    fc_model = Sequential()
    fc_model.add(Merge([vgg_model, lstm_model], mode='concat'))
    fc_model.add(Dropout(dropout_rate))
    fc_model.add(Dense(1000, activation='tanh'))
    fc_model.add(Dropout(dropout_rate))
    fc_model.add(Dense(num_classes, activation='softmax'))
    fc_model.compile(optimizer='adam', loss='categorical_crossentropy',
                     metrics=['accuracy'])
    return fc_model


# Let's try something new
def Word2VecModel_nofc_ret_seq(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate):
    print("Creating text model...")
    model = Sequential()
    model.add(Embedding(num_words, embedding_dim,
                        weights=[embedding_matrix], input_length=seq_length, trainable=False))
    model.add(GRU(output_dim=1024, return_sequences=True, input_shape=(seq_length, embedding_dim)))
    model.add(Dropout(dropout_rate))
    model.add(GRU(output_dim=1024, return_sequences=True))
    return model


def vqa_model_2(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    vgg_model = img_model(dropout_rate)
    vgg_model.add(RepeatVector(seq_length))
    lstm_model = Word2VecModel_nofc_ret_seq(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate)

    print("Merging final model...")
    fc_model = Sequential()
    fc_model.add(Merge([vgg_model, lstm_model], mode='mul', concat_axis=-1))
    fc_model.add(BatchNormalization())
    fc_model.add(GRU(1024, return_sequences=False))
    fc_model.add(Dropout(dropout_rate))
    fc_model.add(BatchNormalization())
    fc_model.add(Dense(1000, activation='tanh'))
    fc_model.add(Dropout(dropout_rate))
    fc_model.add(BatchNormalization())
    fc_model.add(Dense(num_classes, activation='softmax'))
    fc_model.compile(optimizer='adam', loss='categorical_crossentropy',
                     metrics=['accuracy'])
    return fc_model


# Get image candidates & shit
def img_model_notop(dropout_rate):
    print("Creating image model...")
    model = Sequential()
    model.add(Flatten(input_shape=(512, 49)))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dropout_rate))
    return model


def vqa_model_3(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    vgg_model = img_model_notop(dropout_rate)
    lstm_model = Word2VecModel_nofc(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate)
    print("Merging final model...")
    fc_model = Sequential()
    fc_model.add(Merge([vgg_model, lstm_model], mode='concat'))
    fc_model.add(Dense(1000, activation='tanh'))
    fc_model.add(Dropout(dropout_rate))
    fc_model.add(Dense(1000, activation='tanh'))
    fc_model.add(Dropout(dropout_rate))
    fc_model.add(Dense(num_classes, activation='softmax'))
    fc_model.compile(optimizer='adam', loss='categorical_crossentropy',
                     metrics=['accuracy'])
    return fc_model


# Big-ass soft-attention-with-alignment-mega-model
def split_sum(x):
    v1 = K.sum(x[:, 0:26, :], axis=1)
    v2 = K.sum(x[:, 26:75, :], axis=1)
    return K.concatenate([v1, v2], axis=1)


def collapse_sum(x):
    return K.sum(x, axis=1)


def img_model_notop_2(img_input, dropout_rate):
    # This model uses the include_top=False version of VGG19
    print("Creating image model...")
    model = Reshape(target_shape=(512, 49), input_shape=(512, 49))(img_input)
    return model


def BLSTM(text_input, embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate):
    emb = Embedding(num_words, embedding_dim, weights=[embedding_matrix],
                    input_length=seq_length, trainable=False)(text_input)

    return Bidirectional(
        LSTM(512, U_regularizer=l2(0.01), return_sequences=True, consume_less='gpu'),
        merge_mode="sum")(emb)

# 53.25%
def vqa_model_4(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    text_input = Input(shape=(26,), dtype='int32')
    img_input = Input(shape=(512, 49))
    a = BLSTM(text_input, embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate / 2)
    b = img_input  # img_model_notop_2(img_input, dropout_rate/2)
    # a - tensor containing word embeddings after Bi-LSTM - size (batch_size, 26, 512)
    # b - tensor containing image embeddings after VGG19 include_top=False - size (batch_size, 512, 49)

    print("Building Alignment Matrix")
    sa_model = merge([a, b], mode='dot', dot_axes=(2, 1), output_shape=(26, 49))
    sa_model = Activation('relu')(sa_model)

    # Apply softmax per each line of the alignment matrix
    # This and e2 are later used to obtain beta and alpha - meaning the soft aligned image per each word and vice-versa
    e1 = Activation('softmax')(sa_model)

    # Apply softmax per each column of the alignment matrix
    e2 = Permute((2, 1))(sa_model)
    e2 = Activation('softmax')(e2)

    print("Building Soft Attentions")
    beta = merge([e1, b], mode='dot', dot_axes=(2, 2), output_shape=(26, 512))
    alpha = merge([e2, a], mode='dot', dot_axes=(2, 1), output_shape=(49, 512))

    # Since b is (512 x 49) we need to transpose this to align with alpha, so we get an output of size (49 x 1024)
    b_tr = Permute((2, 1))(b)

    print("Building v1, v2, G and H")
    v1 = merge([a, beta], mode='concat', concat_axis=2, output_shape=(26, 1024))
    v2 = merge([b_tr, alpha], mode='concat', concat_axis=2, output_shape=(49, 1024))

    # Concatenate so we don't have to split a shared NN (G) for both, instead, run it on the concatenated input
    # and split the result
    v12 = merge([v1, v2], mode='concat', concat_axis=1, output_shape=(75, 1024))
    # G equivalent
    v12 = TimeDistributed(Dense(512, input_shape=(1024,), activation='relu'))(v12)
    v12 = TimeDistributed(Dropout(dropout_rate))(v12)

    # Split back into v1,i and v2,j vectors + apply sum for each one and obtain v1 and v2, which are concatenated
    v12 = Lambda(split_sum, output_shape=(1024,))(v12)

    H = Dense(num_classes, activation='softmax')(v12)

    model = Model(input=[text_input, img_input], output=H)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model

# also 53.25% but makes more sense
def vqa_model_5(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    text_input = Input(shape=(26,), dtype='int32')
    img_input = Input(shape=(512, 49))
    a = BLSTM(text_input, embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate / 2)
    b = img_input  # img_model_notop_2(img_input, dropout_rate/2)
    # a - tensor containing word embeddings after Bi-LSTM - size (batch_size, 26, 512)
    # b - tensor containing image embeddings after VGG19 include_top=False - size (batch_size, 512, 49)

    # Since b is (512 x 49) we need to transpose this to align with alpha, so we get an output of size (49 x 512)
    b_tr = Permute((2, 1))(b)

    print("Building Alignment Matrix")
    sa_model = merge([a, b], mode='dot', dot_axes=(2, 1), output_shape=(26, 49))
    sa_model = Activation('relu')(sa_model)

    # Apply softmax per each line of the alignment matrix
    # This and e2 are later used to obtain beta and alpha - meaning the soft aligned image per each word and vice-versa
    e1 = Activation('softmax')(sa_model)

    # Apply softmax per each column of the alignment matrix
    e2 = Permute((2, 1))(sa_model)
    e2 = Activation('softmax')(e2)

    print("Building Soft Attentions")
    beta = merge([e1, b], mode='dot', dot_axes=(2, 2), output_shape=(26, 512))
    alpha = merge([e2, a], mode='dot', dot_axes=(2, 1), output_shape=(49, 512))

    print("Building v1, v2, G and H")
    v1 = merge([a, beta], mode='concat', concat_axis=2, output_shape=(26, 1024))
    v2 = merge([b_tr, alpha], mode='concat', concat_axis=2, output_shape=(49, 1024))

    # Concatenate so we don't have to split a shared NN (G) for both, instead, run it on the concatenated input
    # and split the result
    v12 = merge([v1, v2], mode='concat', concat_axis=1, output_shape=(75, 1024))
    # G equivalent
    v12 = Dense(512, input_shape=(1024,), activation='relu')(v12)
    v12 = Dropout(dropout_rate)(v12)

    # Split back into v1,i and v2,j vectors + apply sum for each one and obtain v1 and v2, which are concatenated
    v12 = Lambda(split_sum, output_shape=(1024,))(v12)

    H = Dense(num_classes, activation='softmax')(v12)

    model = Model(input=[text_input, img_input], output=H)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model

# 53.1% but is the closest to the paper implementation
def vqa_model_6(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    text_input = Input(shape=(26,), dtype='int32')
    img_input = Input(shape=(512, 49))
    a = Embedding(num_words, embedding_dim, weights=[embedding_matrix],
                    input_length=seq_length, trainable=False)(text_input)
    #BLSTM(text_input, embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate / 2)
    b = img_input  # img_model_notop_2(img_input, dropout_rate/2)
    # a - tensor containing word embeddings after Bi-LSTM - size (batch_size, 26, 512)
    # b - tensor containing image embeddings after VGG19 include_top=False - size (batch_size, 512, 49)

    # Since b is (512 x 49) we need to transpose this to align with alpha, so we get an output of size (49 x 512)
    b_tr = Permute((2, 1))(b)

    F_a = Dense(512, activation='relu')(a)
    F_a = Dropout(dropout_rate / 2)(F_a)

    F_b = Dense(512, activation='relu')(b_tr)
    F_b = Dropout(dropout_rate / 2)(F_b)
    F_b = Permute((2, 1))(F_b)

    print("Building Alignment Matrix")
    sa_model = merge([F_a, F_b], mode='dot', dot_axes=(2, 1), output_shape=(26, 49))
    sa_model = Activation('relu')(sa_model)

    # Apply softmax per each line of the alignment matrix
    # This and e2 are later used to obtain beta and alpha - meaning the soft aligned image per each word and vice-versa
    e1 = Activation('softmax')(sa_model)

    # Apply softmax per each column of the alignment matrix
    e2 = Permute((2, 1))(sa_model)
    e2 = Activation('softmax')(e2)

    print("Building Soft Attentions")
    beta = merge([e1, b], mode='dot', dot_axes=(2, 2), output_shape=(26, 512))
    alpha = merge([e2, a], mode='dot', dot_axes=(2, 1), output_shape=(49, 512))

    print("Building v1, v2, G and H")
    v1 = merge([a, beta], mode='concat', concat_axis=2, output_shape=(26, 1024))
    v2 = merge([b_tr, alpha], mode='concat', concat_axis=2, output_shape=(49, 1024))

    # Concatenate so we don't have to split a shared NN (G) for both, instead, run it on the concatenated input
    # and split the result
    v12 = merge([v1, v2], mode='concat', concat_axis=1, output_shape=(75, 1024))
    # G equivalent
    v12 = Dense(512, input_shape=(1024,), activation='relu')(v12)
    v12 = Dropout(dropout_rate)(v12)

    # Split back into v1,i and v2,j vectors + apply sum for each one and obtain v1 and v2, which are concatenated
    v12 = Lambda(split_sum, output_shape=(1024,))(v12)

    H = Dense(num_classes, activation='softmax')(v12)

    model = Model(input=[text_input, img_input], output=H)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model


def collapse_max(x):
    return K.max(x, axis=1)

# Ask attend and answer 1 hop ish 51.5%
def vqa_model_7(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    text_input = Input(shape=(26,), dtype='int32')

    a = BLSTM(text_input, embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate / 2)
    b = Input(shape=(512, 49))

    V = a
    S = Permute((2, 1))(b)  # (49, 512)
    SWa = Dense(512)(S)  # (49, 512)
    SWa = Dropout(dropout_rate)(SWa)

    C = merge([V, SWa], mode='dot', dot_axes=(2, 2), output_shape=(26, 49))

    Watt = Lambda(collapse_max, output_shape=(1, 49))(C)
    Watt = Activation('softmax')(Watt)  # (49,)
    Watt = Reshape((1, 49))(Watt)  # (1, 49)

    SWe = Dense(512)(S)  # (49 x 512)
    SWe = Dropout(dropout_rate)(SWe)
    Satt = merge([Watt, SWe], mode='dot', dot_axes=(2, 1), output_shape=(1, 512))

    Vt = Permute((2, 1))(V)  # (512, 26)
    Qbow = Dense(1)(Vt)  # (512, 1)
    Qbow = Dropout(dropout_rate)(Qbow)
    Qbow = Permute((2, 1))(Qbow)  # (1, 512)

    O = merge([Satt, Qbow], mode='sum', output_shape=(1, 512))
    O = Activation('relu')(O)
    P = Dense(num_classes, activation='softmax')(O)
    P = Reshape((1000,))(P)

    model = Model(input=[text_input, b], output=P)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model

# Ask attend and answer 2 hops ish
def vqa_model_8(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    text_input = Input(shape=(26,), dtype='int32')

    a = BLSTM(text_input, embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate / 2)
    b = Input(shape=(512, 49))

    V = a
    S = Permute((2, 1))(b)  # (49, 512)

    # 1st hop
    Wa = Dense(512)
    SWa = Wa(S)  # (49, 512)

    C = merge([V, SWa], mode='dot', dot_axes=(2, 2), output_shape=(26, 49))

    Watt = Lambda(collapse_max, output_shape=(1, 49))(C)
    Watt = Activation('softmax')(Watt)  # (49,)
    Watt = Reshape((1, 49))(Watt)  # (1, 49)

    We = Dense(512)
    SWe = We(S)  # (49, 512)
    Satt = merge([Watt, SWe], mode='dot', dot_axes=(2, 1), output_shape=(1, 512))

    Vt = Permute((2, 1))(V)  # (512, 26)
    Qbow = Dense(1)(Vt)  # (512, 1)
    Qbow = Permute((2, 1))(Qbow)  # (1, 512)

    O = merge([Satt, Qbow], mode='sum', output_shape=(1, 512))
    O = Activation('relu')(O)

    # 2nd hop
    C2 = merge([O, SWe], mode='dot', dot_axes=(2, 2), output_shape=(1, 49))
    Watt2 = Activation('softmax')(C2)  # (1, 49)

    We2 = Dense(512)
    SWe2 = We2(S)  # (49, 512)
    Satt2 = merge([Watt2, SWe2], mode='dot', dot_axes=(2, 1), output_shape=(1, 512))

    O2 = merge([Satt2, O], mode='sum', dot_axes=(2, 2), output_shape=(1, 512))
    O2 = Activation('relu')(O2)

    P = Dense(num_classes, activation='softmax')(O2)
    P = Reshape((1000,))(P)

    #sgd = SGD(lr=0.01, momentum=0.9, decay=0.1)
    model = Model(input=[text_input, b], output=P)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model