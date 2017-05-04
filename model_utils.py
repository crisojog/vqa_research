from model_1 import model_1
from model_2 import model_2
from model_3 import model_3

def get_model(
        dropout_rate,
        regularization_rate,
        embedding_size,
        num_classes,
        model_name):
    if model_name == 'model_1':
        return model_1(embedding_size, dropout_rate, regularization_rate, num_classes)
    elif model_name == 'model_2':
        return model_2(embedding_size, dropout_rate, num_classes)
    elif model_name == 'model_3':
        return model_3(embedding_size, dropout_rate, regularization_rate, num_classes)