import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
from numpy import linalg as LA


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def plot_loss(train_loss, val_loss, savedir):
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(['Train loss', 'Validation loss'], loc='upper left')
    #plt.show()
    plt.savefig('%s/loss_plot.png' % savedir)
    plt.clf()


def plot_accuracy(train_acc, val_acc, savedir):
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.legend(['Train accuracy', 'Validation accuracy'], loc='upper left')
    #plt.show()
    plt.savefig('%s/acc_plot.png' % savedir)
    plt.clf()


def plot_eval_accuracy(eval_acc, savedir):
    plt.plot(eval_acc)
    plt.legend(['Eval accuracy on validation'], loc='lower right')
    #plt.show()
    plt.savefig('%s/eval_acc_plot.png' % savedir)
    plt.clf()


def get_train_data(ans_types, use_test, use_embedding_matrix):
    if ans_types:
        extra = "_%s" % ans_types.replace("/", "")
    else:
        extra = ""

    if use_embedding_matrix:
        print "Loading token embedding matrix"
        embedding_matrix = pickle.load(open("data/tokens_embedding.pkl", "r"))
        tokens = "_tokens"
    else:
        embedding_matrix = None
        tokens = ""

    if not use_test:
        print "Loading train questions"
        ques_train_map = pickle.load(open("data/train%s_questions.pkl" % tokens, "r"))
        print "Loading train answers"
        ans_train_map = pickle.load(open("data/train_answers%s.pkl" % extra, "r"))
        print "Loading train images"
        img_train_map = pickle.load(open("data/train_images.pkl", "r"))
        print "Loading ques_to_img map"
        ques_to_img_train = pickle.load(open("data/train_ques_to_img.pkl", "r"))
        print "Done"
        return embedding_matrix, ques_train_map, ans_train_map, img_train_map, ques_to_img_train
    else:
        print "Loading train_val questions"
        ques_train_map = pickle.load(open("data/train_val%s_questions.pkl" % tokens, "r"))
        print "Loading train_val answers"
        ans_train_map = pickle.load(open("data/train_val_answers%s.pkl" % extra, "r"))
        print "Loading train_val images"
        img_train_map = pickle.load(open("data/train_val_images.pkl", "r"))
        print "Loading ques_to_img map"
        ques_to_img_train = pickle.load(open("data/train_val_ques_to_img.pkl", "r"))
        print "Done"
        return embedding_matrix, ques_train_map, ans_train_map, img_train_map, ques_to_img_train


def get_val_data(ans_types, use_test, use_embedding_matrix):
    if ans_types:
        extra = "_%s" % ans_types.replace("/", "")
    else:
        extra = ""

    if use_embedding_matrix:
        print "Loading token embedding matrix"
        embedding_matrix = pickle.load(open("data/tokens_embedding.pkl", "r"))
        tokens = "_tokens"
    else:
        embedding_matrix = None
        tokens = ""

    if not use_test:
        print "Loading validation questions"
        ques_val_map = pickle.load(open("data/val%s_questions.pkl" % tokens, "r"))
        print "Loading validation answers"
        ans_val_map = pickle.load(open("data/val_answers%s.pkl" % extra, "r"))
        print "Loading validation images"
        img_val_map = pickle.load(open("data/val_images.pkl", "r"))
        print "Loading ques_to_img map"
        ques_to_img_val = pickle.load(open("data/val_ques_to_img.pkl", "r"))
        print "Done"
        return embedding_matrix, ques_val_map, ans_val_map, img_val_map, ques_to_img_val
    else:
        print "Loading test questions"
        ques_val_map = pickle.load(open("data/test%s_questions.pkl" % tokens, "r"))
        print "Loading test images"
        img_val_map = pickle.load(open("data/test_images.pkl", "r"))
        print "Loading ques_to_img map"
        ques_to_img_val = pickle.load(open("data/test_ques_to_img.pkl", "r"))
        print "Done"
        return embedding_matrix, ques_val_map, None, img_val_map, ques_to_img_val


def normalize_image_embeddings(img_map_list):
    for img_map in img_map_list:
        for k, features in img_map.iteritems():
            features /= LA.norm(features, 2, axis=0)
            img_map[k] = features