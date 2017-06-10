from keras.layers import Dense, Dropout, LSTM, Activation
from keras.layers import merge, Input, Embedding
from keras.models import Model
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.utils import plot_model
from keras.optimizers import RMSprop, Nadam, Adagrad, Adam, Adamax
from keras.initializers import glorot_uniform, glorot_normal


# Model 1 - Baseline -- 56.68% Val Acc -- 60.07% Test-dev Acc
def text_model(embedding_matrix, num_tokens, embedding_dim, text_input, dropout_rate, regularization_rate):
    print("Creating text model...")
    model = Embedding(num_tokens, embedding_dim, weights=[embedding_matrix], trainable=False)(text_input)
    model = Activation('tanh')(model)
    model = LSTM(units=1024, return_sequences=True, U_regularizer=l2(0.1), kernel_initializer=glorot_normal())(model)
    model = Dropout(dropout_rate)(model)
    model = LSTM(units=1024, U_regularizer=l2(0.1), kernel_initializer=glorot_normal())(model)
    model = Dropout(dropout_rate)(model)
    model = Dense(1024, activation='tanh', W_constraint=maxnorm(3),
                  kernel_initializer=glorot_normal(),
                  kernel_regularizer=l2(regularization_rate))(model)
    return model


def img_model(img_input, regularization_rate):
    print("Creating image model...")
    model = Dense(1024, activation='tanh', W_constraint=maxnorm(3),
                  kernel_initializer=glorot_normal(),
                  kernel_regularizer=l2(regularization_rate))(img_input)
    return model


def model_1(embedding_matrix, num_tokens, embedding_dim, dropout_rate, regularization_rate, num_classes):
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
    # opt = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # opt = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    print (model.summary())
    plot_model(model, to_file='model_plots/model1.png')
    return model
