from keras import backend as K
from keras.layers import Input, Dense, Activation, Dropout, LSTM, Embedding
from keras.layers import Reshape, Lambda, Permute
from keras.layers import merge, Bidirectional
from keras.models import Model
from keras.regularizers import l2


def split_sum(x):
    v2 = K.sum(x[:, 0:49, :], axis=1)
    v1 = K.sum(x[:, 49:, :], axis=1)
    return K.concatenate([v1, v2], axis=1)


def collapse_sum(x):
    return K.sum(x, axis=1)


def BLSTM(text_input):
    return Bidirectional(
        LSTM(512, U_regularizer=l2(0.01), return_sequences=True),
        merge_mode="sum")(text_input)


def model_2(embedding_dim, dropout_rate, num_classes):
    text_input = Input(shape=(None, embedding_dim))
    img_input = Input(shape=(512, 49))

    a = BLSTM(text_input)
    b = img_input
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
    sa_model = merge([F_a, F_b], mode='dot', dot_axes=(2, 1), output_shape=(None, 49))
    sa_model = Activation('relu')(sa_model)

    # Apply softmax per each line of the alignment matrix
    # This and e2 are later used to obtain beta and alpha - meaning the soft aligned image per each word and vice-versa
    e1 = Activation('softmax')(sa_model)

    # Apply softmax per each column of the alignment matrix
    e2 = Permute((2, 1))(sa_model)
    e2 = Activation('softmax')(e2)

    print("Building Soft Attentions")
    beta = merge([e1, b], mode='dot', dot_axes=(2, 2), output_shape=(None, 512))
    alpha = merge([e2, a], mode='dot', dot_axes=(2, 1), output_shape=(49, 512))

    print("Building v1, v2, G and H")
    v1 = merge([a, beta], mode='concat', concat_axis=2, output_shape=(None, 1024))
    v2 = merge([b_tr, alpha], mode='concat', concat_axis=2, output_shape=(49, 1024))

    # Concatenate so we don't have to split a shared NN (G) for both, instead, run it on the concatenated input
    # and split the result
    v12 = merge([v2, v1], mode='concat', concat_axis=1, output_shape=(None, 1024))
    # G equivalent
    v12 = Dense(512, input_shape=(1024,), activation='relu')(v12)
    v12 = Dropout(dropout_rate)(v12)

    # Split back into v1,i and v2,j vectors + apply sum for each one and obtain v1 and v2, which are concatenated
    v12 = Lambda(split_sum, output_shape=(1024,))(v12)

    H = Dense(num_classes, activation='softmax')(v12)

    model = Model(input=[img_input, text_input], output=H)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model
