from keras.layers import Dense, Dropout, LSTM, Activation, Conv1D, GlobalMaxPooling1D
from keras.layers import merge, Input, Embedding, concatenate, BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.utils import plot_model
from keras.optimizers import RMSprop, Nadam, Adagrad, Adam, Adamax
from keras.initializers import glorot_uniform, glorot_normal


# Model 2 - Baseline-CNN
def text_model(embedding_matrix, num_tokens, embedding_dim, text_input, dropout_rate, regularization_rate):
    model = Embedding(num_tokens, embedding_dim, weights=[embedding_matrix], trainable=False)(text_input)
    model_unigram = Conv1D(filters=512, kernel_size=1, padding='valid')(model)
    model_unigram = Activation('relu')(model_unigram)
    model_unigram = GlobalMaxPooling1D()(model_unigram)

    model_bigram = Conv1D(filters=512, kernel_size=2, padding='valid')(model)
    model_bigram = Activation('relu')(model_bigram)
    model_bigram = GlobalMaxPooling1D()(model_bigram)

    model_trigram = Conv1D(filters=1024, kernel_size=3, padding='valid')(model)
    model_trigram = Activation('relu')(model_trigram)
    model_trigram = GlobalMaxPooling1D()(model_trigram)

    model_merge = concatenate([model_unigram, model_bigram, model_trigram])
    model_merge = BatchNormalization()(model_merge)
    model_merge = Activation('tanh')(model_merge)
    model_merge = Dropout(dropout_rate)(model_merge)
    return model_merge


def img_model(img_input, regularization_rate):
    print("Creating image model...")
    model = Dense(2048, activation='tanh', W_constraint=maxnorm(3),
                  kernel_initializer=glorot_normal(),
                  kernel_regularizer=l2(regularization_rate))(img_input)
    return model


def baseline_cnn(embedding_matrix, num_tokens, embedding_dim, dropout_rate, regularization_rate, num_classes):
    img_input = Input(shape=(2048,))
    text_input = Input(shape=(None,))

    vgg_model = img_model(img_input, regularization_rate)
    lstm_model = text_model(embedding_matrix, num_tokens, embedding_dim, text_input, dropout_rate, regularization_rate)

    print("Merging final model...")
    fc_model = merge([vgg_model, lstm_model], mode='mul')
    fc_model = Dropout(dropout_rate)(fc_model)
    fc_model = Dense(1000, activation='tanh', W_constraint=maxnorm(3),
                     kernel_initializer=glorot_normal(),
                     kernel_regularizer=l2(regularization_rate))(fc_model)
    fc_model = Dropout(dropout_rate)(fc_model)
    fc_model = Dense(num_classes, activation='softmax', W_constraint=maxnorm(3),
                     kernel_initializer=glorot_normal(),
                     kernel_regularizer=l2(regularization_rate))(fc_model)

    model = Model(inputs=[img_input, text_input], outputs=fc_model)
    opt = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    print (model.summary())
    plot_model(model, to_file='model_plots/model_baseline_cnn.png')
    return model
