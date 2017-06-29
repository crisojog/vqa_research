from keras.layers import Dense, Dropout, LSTM, Activation, Permute, RepeatVector, Flatten, Bidirectional, Embedding, \
    Conv1D, merge, Input, Lambda, concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.optimizers import RMSprop, Nadam, Adagrad, Adam
from keras.utils import plot_model
from keras import backend as K


def img_model(img_input, dropout_rate):
    print("Creating image model...")
    model = Dropout(0.3)(img_input)
    model = Dense(1024, activation='tanh', kernel_initializer='he_normal', kernel_constraint=maxnorm(3))(model)
    model = Dropout(dropout_rate)(model)
    model = Permute((2, 1))(model)
    return model


def BLSTM(text_input, embedding_matrix, num_tokens, embedding_dim, dropout_rate):
    emb = Embedding(num_tokens, embedding_dim, weights=[embedding_matrix], trainable=False)(text_input)
    model = Activation('tanh')(emb)
    model = Dropout(0.3)(model)
    model = Bidirectional(LSTM(1024, U_regularizer=l2(0.015), kernel_initializer='he_normal', return_sequences=True),
                          merge_mode="ave")(model)
    model = Dropout(dropout_rate)(model)
    model = Conv1D(filters=1024, kernel_size=3, padding='valid',
                   kernel_initializer='he_normal', kernel_constraint=maxnorm(3))(model)
    model = Activation('relu')(model)
    model = Dropout(dropout_rate)(model)
    model = Dense(1024, activation='tanh', kernel_constraint=maxnorm(3))(model)
    model = Dropout(dropout_rate)(model)
    return model


def collapse_avg(x):
    return K.mean(x, axis=1)


def dual_att(embedding_matrix, num_tokens, embedding_dim, dropout_rate, regularization_rate, num_classes):
    text_input = Input(shape=(None,), dtype='int32')
    img_input = Input(shape=(49, 2048))

    a = BLSTM(text_input, embedding_matrix, num_tokens, embedding_dim, dropout_rate)
    b = img_model(img_input, dropout_rate)

    sa_model = merge([a, b], mode='dot', dot_axes=(2, 1), output_shape=(None, 49))

    e1 = Activation('softmax')(sa_model)

    e2 = Permute((2, 1))(sa_model)
    e2 = Activation('softmax')(e2)

    beta = merge([e1, b], mode='dot', dot_axes=(2, 2), output_shape=(None, 1024))
    alpha = merge([e2, a], mode='dot', dot_axes=(2, 1), output_shape=(49, 1024))

    b_tr = Permute((2, 1))(b)

    v1 = merge([a, beta], mode='mul', output_shape=(None, 1024))
    v1 = Lambda(collapse_avg, output_shape=(1024,))(v1)
    v1 = Dropout(0.5)(v1)

    v2 = merge([b_tr, alpha], mode='mul', output_shape=(49, 1024))
    v2 = Lambda(collapse_avg, output_shape=(1024,))(v2)
    v2 = Dropout(0.5)(v2)

    fc_model = concatenate([v1, v2], axis=1)  # (2048)
    fc_model = Dropout(0.5)(fc_model)
    fc_model = Dense(1024, activation='tanh', kernel_constraint=maxnorm(3),
                     kernel_initializer='he_normal')(fc_model)
    fc_model = Dropout(0.5)(fc_model)
    fc_model = Dense(num_classes, activation='softmax', kernel_constraint=maxnorm(3),
                     kernel_initializer='he_normal')(fc_model)

    model = Model(input=[img_input, text_input], output=fc_model)
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipvalue=0.1)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    plot_model(model, to_file='model_plots/model_dual_attention.png')
    return model
