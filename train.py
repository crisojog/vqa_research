import argparse
import cPickle as pickle
import os

from preprocess import get_most_common_answers
from utils import plot_accuracy, plot_eval_accuracy, plot_loss
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


def get_train_data(ans_types):
    if ans_types:
        extra = "_%s" % ans_types.replace("/", "")
    else:
        extra = ""
    print "Loading train questions"
    ques_train_map = pickle.load(open("data/train_questions.pkl", "r"))
    print "Loading train answers"
    ans_train_map = pickle.load(open("data/train_answers%s.pkl" % extra, "r"))
    print "Loading train images"
    img_train_map = pickle.load(open("data/train_images.pkl", "r"))
    print "Loading ques_to_img map"
    ques_to_img_train = pickle.load(open("data/train_ques_to_img.pkl", "r"))
    print "Done"
    return ques_train_map, ans_train_map, img_train_map, ques_to_img_train


def get_val_data(ans_types):
    if ans_types:
        extra = "_%s" % ans_types.replace("/", "")
    else:
        extra = ""
    print "Loading validation questions"
    ques_val_map = pickle.load(open("data/val_questions.pkl", "r"))
    print "Loading validation answers"
    ans_val_map = pickle.load(open("data/val_answers%s.pkl" % extra, "r"))
    print "Loading validation images"
    img_val_map = pickle.load(open("data/val_images.pkl", "r"))
    print "Loading ques_to_img map"
    ques_to_img_val = pickle.load(open("data/val_ques_to_img.pkl", "r"))
    print "Done"
    return ques_val_map, ans_val_map, img_val_map, ques_to_img_val


# Train our model
def train_model(ques_train_map, ans_train_map, img_train_map, ques_train_ids, ques_to_img_train,
                ques_val_map, ans_val_map, img_val_map, ques_val_ids, ques_to_img_val,
                id_to_ans, train_dim, val_dim, ans_types, params):

    # training parameters
    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    num_batches_train = train_dim // batch_size
    num_batches_val = val_dim // batch_size
    eval_every = 5

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

    for k in range(num_epochs):
        loss, acc = train_epoch(k + 1, model, num_batches_train, batch_size, ques_train_map, ans_train_map,
                                img_train_map, ques_train_ids, ques_to_img_train)
        train_loss.append(loss)
        train_acc.append(acc)
        loss, acc = val_epoch(k + 1, model, num_batches_val, batch_size, ques_val_map, ans_val_map, img_val_map,
                              ques_val_ids, ques_to_img_val)
        val_loss.append(loss)
        val_acc.append(acc)
        if (k + 1) % eval_every == 0:
            model.save_weights("%s/%s_epoch_%d_weights.h5" % (savedir, params['model'], (k + 1)), overwrite=True)
            eval_accuracy = evaluate(
                model, vqa_val, batch_size, ques_val_map, img_val_map, id_to_ans, params['ans_types'])
            print ("Eval accuracy: %.2f" % eval_accuracy)
            eval_acc.append(eval_accuracy)

    plot_loss(train_loss, val_loss, savedir)
    plot_accuracy(train_acc, val_acc, savedir)
    plot_eval_accuracy(eval_acc, savedir)

    best_epoch = (1 + np.argmax(np.array(eval_acc))) * eval_every
    print "Best accuracy %.02f on epoch %d" % (max(eval_acc), best_epoch)

    model.load_weights("%s/%s_epoch_%d_weights.h5" % (savedir, params['model'], best_epoch))
    evaluate(model, vqa_val, batch_size, ques_val_map, img_val_map, id_to_ans, params['ans_types'], verbose=True)


def main(params):
    ans_to_id, id_to_ans = get_most_common_answers(vqa_train, int(params['num_answers']), params['ans_types'])

    ques_train_map, ans_train_map, img_train_map, ques_to_img_train = get_train_data(params['ans_types'])
    ques_val_map, ans_val_map, img_val_map, ques_to_img_val = get_val_data(params['ans_types'])

    filtered_ann_ids_train = set(vqa_train.getQuesIds(ansTypes=params['ans_types']))
    filtered_ann_ids_val = set(vqa_val.getQuesIds(ansTypes=params['ans_types']))

    ques_train_ids = ques_train_map.keys()
    ques_val_ids = ques_val_map.keys()

    ques_train_ids = np.array([i for i in ques_train_ids if i in filtered_ann_ids_train])
    ques_val_ids = np.array([i for i in ques_val_ids if i in filtered_ann_ids_val])

    train_dim, val_dim = len(ques_train_ids), len(ques_val_ids)
    print "Loaded dataset with train size %d and val size %d" % (train_dim, val_dim)

    train_model(ques_train_map, ans_train_map, img_train_map, ques_train_ids, ques_to_img_train,
                ques_val_map, ans_val_map, img_val_map, ques_val_ids, ques_to_img_val,
                id_to_ans, train_dim, val_dim, params['ans_types'], params)


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

    args = parser.parse_args()
    params = vars(args)
    main(params)
