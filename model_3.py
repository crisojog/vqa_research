from keras.layers import Dense, Dropout, LSTM, Activation, Input, Lambda
from keras import backend as K
from keras.layers.merge import multiply, dot
from keras.models import Model
from keras import regularizers
from numpy import newaxis

# Define our model
def text_model(text_input, dropout_rate, regularization_rate):
    print("Creating text model...")
    model = Activation('tanh')(text_input)
    model = LSTM(units=1024, return_sequences=True)(model)
    model = Dropout(dropout_rate)(model)
    model = LSTM(units=1024)(model)
    model = Dropout(dropout_rate)(model)
    model = Dense(256, activation='tanh', kernel_regularizer=regularizers.l2(regularization_rate))(model)
    return model


def img_model(img_input, regularization_rate):
    print("Creating image model...")
    model = Dense(256, activation='tanh', kernel_regularizer=regularizers.l2(regularization_rate))(img_input)
    return model


def outer_product(inputs):
    """
    inputs: list of two tensors (of equal dimensions, 
        for which you need to compute the outer product
    """
    x, y = inputs
    batchSize = K.shape(x)[0]
    outerProduct = x[:, :, newaxis] * y[:, newaxis, :]
    outerProduct = K.reshape(outerProduct, (batchSize, -1))
    # returns a flattened batch-wise set of tensors
    return outerProduct


def model_3(embedding_dim, dropout_rate, regularization_rate, num_classes):
    img_input = Input(shape=(2048,))
    text_input = Input(shape=(None, embedding_dim))

    vgg_model = img_model(img_input, regularization_rate)
    lstm_model = text_model(text_input, dropout_rate, regularization_rate)

    print("Merging final model...")
    fc_model = Lambda(outer_product, output_shape=(256**2, ))([vgg_model, lstm_model])
    fc_model = Dropout(dropout_rate)(fc_model)
    fc_model = Dense(1000, activation='tanh', kernel_regularizer=regularizers.l2(regularization_rate))(fc_model)
    fc_model = Dropout(dropout_rate)(fc_model)
    fc_model = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(regularization_rate))(fc_model)
    model = Model(inputs=[img_input, text_input], outputs=fc_model)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                     metrics=['accuracy'])
    print (model.summary())
    return model
