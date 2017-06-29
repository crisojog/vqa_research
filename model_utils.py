from baseline import baseline
from baseline_cnn import baseline_cnn
from dual_att import dual_att


def get_model(
        dropout_rate,
        regularization_rate,
        embedding_size,
        num_classes,
        model_name,
        embedding_matrix=None):
    if model_name == 'baseline':
        return baseline(embedding_matrix, len(embedding_matrix),
                        embedding_size, dropout_rate, regularization_rate, num_classes)
    elif model_name == 'baseline_cnn':
        return baseline_cnn(embedding_matrix, len(embedding_matrix),
                            embedding_size, dropout_rate, regularization_rate, num_classes)
    elif model_name == 'dual_att':
        return dual_att(embedding_matrix, len(embedding_matrix),
                        embedding_size, dropout_rate, regularization_rate, num_classes)
