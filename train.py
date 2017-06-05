import argparse
import os
import math

from utils import *
from preprocess import get_most_common_answers
from train_utils import *
from model_utils import get_model

dataDir = 'VQA'
taskType = 'OpenEnded'
dataType = 'mscoco'  # 'mscoco' for real and 'abstract_v002' for abstract

dataSubType_train = 'train2014'
annFile_train     = '%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, dataSubType_train)
quesFile_train    = '%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, dataSubType_train)
imgDir_train      = '%s/Images/%s/%s/' %(dataDir, dataType, dataSubType_train)
vqa_train = VQA(annFile_train, quesFile_train)

dataSubType_val   = 'val2014'
annFile_val       = '%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, dataSubType_val)
quesFile_val      = '%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, dataSubType_val)
imgDir_val        = '%s/Images/%s/%s/' %(dataDir, dataType, dataSubType_val)
vqa_val = VQA(annFile_val, quesFile_val)

dataSubType_test  = 'test-dev2015'  # Hardcoded for test-dev
quesFile_test     = '%s/Questions/%s_%s_%s_questions.json' % (dataDir, taskType, dataType, dataSubType_test)
imgDir_test       = '%s/Images/%s/%s/' % (dataDir, dataType, 'test2015')


# Train our model
def train_model(ques_train_map, ans_train_map, img_train_map, ques_train_ids, ques_to_img_train,
                ques_val_map, ans_val_map, img_val_map, ques_val_ids, ques_to_img_val,
                id_to_ans, train_dim, val_dim, ans_types, params, embedding_matrix):

    # training parameters
    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    num_batches_train = int(math.ceil(float(train_dim) / batch_size))
    num_batches_val = int(math.ceil(float(val_dim) / batch_size))
    eval_every = params['eval_every']
    start_from = params['start_from_epoch']
    use_embedding_matrix = not params['not_use_embedding_matrix']

    train_loss, train_acc = [], []
    val_loss, val_acc = [], []
    eval_acc = []

    print "Loading model"

    model = get_model(
        dropout_rate=float(params['dropout_rate']),
        regularization_rate=float(params['regularization_rate']),
        embedding_size=int(params['embedding_size']),
        num_classes=int(params['num_answers']),
        model_name=params['model'],
        embedding_matrix=embedding_matrix)

    if not ans_types:
        savedir = "models/%s_%s" % (params['model'], str(params['num_answers']))
    else:
        savedir = "models/%s_%s_%s" % (params['model'], ans_types.replace("/", ""), str(params['num_answers']))

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    if start_from > 0:
        weights_filename = "%s/%s_epoch_%d_weights.h5" % (savedir, params['model'], start_from)
        if os.path.exists(weights_filename):
            model.load_weights(weights_filename)
        else:
            raise Exception('The weight file %s does not exist' % weights_filename)

    word_embedding_size = int(params['embedding_size'])
    use_first_words = params['use_first_words']
    for k in range(start_from, start_from + num_epochs):
        loss, acc = train_epoch(k + 1, model, num_batches_train, batch_size, ques_train_map, ans_train_map,
                                img_train_map, ques_train_ids, ques_to_img_train, word_embedding_size,
                                use_first_words, use_embedding_matrix, int(params['num_answers']))
        train_loss.append(loss)
        train_acc.append(acc)
        if not params['use_test']:
            loss, acc = val_epoch(k + 1, model, num_batches_val, batch_size, ques_val_map, ans_val_map,
                                  img_val_map, ques_val_ids, ques_to_img_val, word_embedding_size,
                                  use_first_words, use_embedding_matrix, int(params['num_answers']))
            val_loss.append(loss)
            val_acc.append(acc)
        if (k + 1) % eval_every == 0:
            model.save_weights("%s/%s_epoch_%d_weights.h5" % (savedir, params['model'], (k + 1)), overwrite=True)
            if not params['use_test']:
                eval_accuracy = evaluate(model, vqa_val, batch_size, ques_val_map, img_val_map,
                                         id_to_ans, word_embedding_size, use_first_words, use_embedding_matrix)
                print ("Eval accuracy: %.2f" % eval_accuracy)
                eval_acc.append(eval_accuracy)
    plot_loss(train_loss, val_loss, savedir)
    plot_accuracy(train_acc, val_acc, savedir)

    if not params['use_test']:
        plot_eval_accuracy(eval_acc, savedir)

        best_epoch = (1 + np.argmax(np.array(eval_acc))) * eval_every
        print "Best accuracy %.02f on epoch %d" % (max(eval_acc), best_epoch)

        model.load_weights("%s/%s_epoch_%d_weights.h5" % (savedir, params['model'], best_epoch))
        evaluate(model, vqa_val, batch_size, ques_val_map, img_val_map, id_to_ans,
                 word_embedding_size, use_first_words, use_embedding_matrix, verbose=True)


def main(params):
    use_embedding_matrix = not params['not_use_embedding_matrix']
    ans_to_id, id_to_ans = get_most_common_answers(vqa_train, vqa_val, int(params['num_answers']), params['ans_types'],
                                                   show_top_ans=False, use_test=params['use_test'])

    embedding_matrix, ques_train_map, ans_train_map, img_train_map, ques_to_img_train = \
        get_train_data(params['ans_types'], params['use_test'], use_embedding_matrix)
    _, ques_val_map, ans_val_map, img_val_map, ques_to_img_val = get_val_data(params['ans_types'],
                                                                              params['use_test'],
                                                                              use_embedding_matrix)

    normalize_image_embeddings([img_train_map, img_val_map])

    if not params['use_test']:
        filtered_ann_ids_train = set(vqa_train.getQuesIds(ansTypes=params['ans_types']))
        filtered_ann_ids_val = set(vqa_val.getQuesIds(ansTypes=params['ans_types']))
    else:
        filtered_ann_ids_train = set(vqa_train.getQuesIds(ansTypes=params['ans_types']) +
                                     vqa_val.getQuesIds(ansTypes=params['ans_types']))

    ques_train_ids = ques_train_map.keys()
    ques_val_ids = ques_val_map.keys()

    ques_train_ids = np.array([i for i in ques_train_ids if i in filtered_ann_ids_train])
    if not params['use_test']:
        ques_val_ids = np.array([i for i in ques_val_ids if i in filtered_ann_ids_val])

    train_dim, val_dim = len(ques_train_ids), len(ques_val_ids)
    print "Loaded dataset with train size %d and val size %d" % (train_dim, val_dim)

    if not params['eval_only']:
        train_model(ques_train_map, ans_train_map, img_train_map, ques_train_ids, ques_to_img_train,
                    ques_val_map, ans_val_map, img_val_map, ques_val_ids, ques_to_img_val,
                    id_to_ans, train_dim, val_dim, params['ans_types'], params, embedding_matrix)
    else:
        savedir = "models/%s_%s" % (params['model'], str(params['num_answers']))
        print "Loading model"
        model = get_model(
            dropout_rate=float(params['dropout_rate']),
            regularization_rate=float(params['regularization_rate']),
            embedding_size=int(params['embedding_size']),
            num_classes=int(params['num_answers']),
            model_name=params['model'],
            embedding_matrix=embedding_matrix)
        model.load_weights("%s/%s_epoch_%d_weights.h5" % (savedir, params['model'], params['eval_epoch']))
        evaluate_for_test(
            quesFile_test,
            model,
            params['batch_size'],
            ques_val_map,
            img_val_map,
            id_to_ans,
            params['embedding_size'],
            params['use_first_words'],
            use_embedding_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='model_1', help='model to use for training')
    parser.add_argument('--ans_types', default=[], help='filter questions with specific answer types')
    parser.add_argument('--num_answers', default=1000, type=int, help='number of top answers to classify')
    parser.add_argument('--num_epochs', default=80, type=int, help='number of epochs to train')
    parser.add_argument('--batch_size', default=1000, type=int, help='batch size for training')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate for the dropout layers')
    parser.add_argument('--regularization_rate', default=0., type=float, help='regularization rate for the FC layers')
    parser.add_argument('--embedding_size', default=300, type=int, help='length of the word embedding')
    parser.add_argument('--eval_every', default=5, type=int, help='how often to run the model evaluation')
    parser.add_argument('--start_from_epoch', default=0, type=int, help='start with weights from a previous epoch and '
                                                                        'go on from there')
    parser.add_argument('--use_test', dest='use_test', action='store_true',
                        help='use test set (which also means training on train+val')
    parser.set_defaults(use_test=False)
    parser.add_argument('--eval_only', dest='eval_only', action='store_true',
                        help='used to only evaluate a specific model (specify which epoch as well)')
    parser.set_defaults(eval_only=False)
    parser.add_argument('--eval_epoch', default=40, type=int, help='epoch to evaluate')
    parser.add_argument('--use_first_words', default=0, type=int, help='use the first X words of the question as a model parameter')
    parser.add_argument('--not_use_embedding_matrix', dest='not_use_embedding_matrix', action='store_true',
                        help='do not use a word embedding matrix to store the word embeddings,'
                             ' otherwise the models will use an Embedding layer')
    parser.set_defaults(not_use_embedding_matrix=False)

    args = parser.parse_args()
    params = vars(args)
    main(params)

