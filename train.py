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
                id_to_ans, train_dim, val_dim, ans_types, params):

    # training parameters
    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    num_batches_train = int(math.ceil(float(train_dim) / batch_size))
    num_batches_val = int(math.ceil(float(val_dim) / batch_size))
    eval_every = params['eval_every']

    train_loss, train_acc = [], []
    val_loss, val_acc = [], []
    eval_acc = []

    print "Loading model"
    model = get_model(
        dropout_rate=float(params['dropout_rate']),
        regularization_rate=float(params['regularization_rate']),
        embedding_size=int(params['embedding_size']),
        num_classes=int(params['num_answers']),
        model_name=params['model'])

    if not ans_types:
        savedir = "models/%s_%s" % (params['model'], str(params['num_answers']))
    else:
        savedir = "models/%s_%s_%s" % (params['model'], ans_types.replace("/", ""), str(params['num_answers']))

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    word_embedding_size = int(params['embedding_size'])
    for k in range(num_epochs):
        loss, acc = train_epoch(k + 1, model, num_batches_train, batch_size, ques_train_map, ans_train_map,
                                img_train_map, ques_train_ids, ques_to_img_train, word_embedding_size)
        train_loss.append(loss)
        train_acc.append(acc)
        if not params['use_test']:
            loss, acc = val_epoch(k + 1, model, num_batches_val, batch_size, ques_val_map, ans_val_map, img_val_map,
                                  ques_val_ids, ques_to_img_val, word_embedding_size)
            val_loss.append(loss)
            val_acc.append(acc)
        if (k + 1) % eval_every == 0:
            model.save_weights("%s/%s_epoch_%d_weights.h5" % (savedir, params['model'], (k + 1)), overwrite=True)
            if not params['use_test']:
                eval_accuracy = evaluate(model, vqa_val, batch_size, ques_val_map, img_val_map,
                                         id_to_ans, word_embedding_size)
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
                 word_embedding_size, verbose=True)


def main(params):
    ans_to_id, id_to_ans = get_most_common_answers(vqa_train, vqa_val, int(params['num_answers']), params['ans_types'],
                                                   show_top_ans=False, use_test=params['use_test'])

    ques_train_map, ans_train_map, img_train_map, ques_to_img_train = get_train_data(params['ans_types'],
                                                                                     params['use_test'])
    ques_val_map, ans_val_map, img_val_map, ques_to_img_val = get_val_data(params['ans_types'],
                                                                           params['use_test'])

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
                    id_to_ans, train_dim, val_dim, params['ans_types'], params)
    else:
        savedir = "models/%s_%s" % (params['model'], str(params['num_answers']))
        print "Loading model"
        model = get_model(
            dropout_rate=float(params['dropout_rate']),
            regularization_rate=float(params['regularization_rate']),
            embedding_size=int(params['embedding_size']),
            num_classes=int(params['num_answers']),
            model_name=params['model'])
        model.load_weights("%s/%s_epoch_%d_weights.h5" % (savedir, params['model'], params['eval_epoch']))
        evaluate_for_test(
            quesFile_test,
            model,
            params['batch_size'],
            ques_val_map,
            img_val_map,
            id_to_ans,
            params['embedding_size'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='model_1', help='model to use for training')
    parser.add_argument('--ans_types', default=[], help='filter questions with specific answer types')
    parser.add_argument('--num_answers', default=1000, type=int, help='number of top answers to classify')
    parser.add_argument('--num_epochs', default=80, type=int, help='number of epochs to train')
    parser.add_argument('--batch_size', default=1000, type=int, help='batch size for training')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate for the dropout layers')
    parser.add_argument('--regularization_rate', default=0., type=float, help='regularization rate for the FC layers')
    parser.add_argument('--embedding_size', default=300, type=int, help='length of the a word embedding')
    parser.add_argument('--eval_every', default=5, type=int, help='how often to run the model evaluation')
    parser.add_argument('--use_test', dest='use_test', action='store_true',
                        help='use test set (which also means training on train+val')
    parser.set_defaults(use_test=False)
    parser.add_argument('--eval_only', dest='eval_only', action='store_true',
                        help='used to only evaluate a specific model (specify which epoch as well)')
    parser.set_defaults(eval_only=False)
    parser.add_argument('--eval_epoch', default=40, type=int, help='epoch to evaluate')

    args = parser.parse_args()
    params = vars(args)
    main(params)
