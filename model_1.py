from keras.layers import Dense, Dropout, LSTM, Activation
from keras.layers import merge, Input
from keras.models import Model


# Define our model
def text_model(text_input, embedding_dim, dropout_rate):
    print("Creating text model...")
    model = Activation('tanh')(text_input)
    model = LSTM(units=1024, return_sequences=True)(model)
    model = Dropout(dropout_rate)(model)
    model = LSTM(units=1024)(model)
    model = Dropout(dropout_rate)(model)
    model = Dense(1024, activation='tanh')(model)
    return model


def img_model(img_input):
    print("Creating image model...")
    model = Dense(1024, activation='tanh')(img_input)
    return model


def model_1(embedding_dim, dropout_rate, num_classes):
    img_iput = Input(shape=(2048,))
    text_input = Input(shape=(None, embedding_dim))

    vgg_model = img_model(img_iput)
    lstm_model = text_model(text_input, embedding_dim, dropout_rate)
    print("Merging final model...")
    fc_model = merge([vgg_model, lstm_model], mode='mul')
    fc_model = Dropout(dropout_rate)(fc_model)
    fc_model = Dense(1000, activation='tanh')(fc_model)
    fc_model = Dropout(dropout_rate)(fc_model)
    fc_model = Dense(num_classes, activation='softmax')(fc_model)
    model = Model(inputs=[img_iput, text_input], outputs=fc_model)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                     metrics=['accuracy'])
    print (model.summary())
    return model
