import matplotlib.pyplot as plt

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
