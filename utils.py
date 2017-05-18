import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle

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