import numpy as np
import argparse
import skimage.io as io
import os
import matplotlib.pyplot as plt
import scipy
import spacy

from model_utils import get_model
from utils import get_val_data, normalize_image_embeddings
from preprocess import get_most_common_answers
from VQA.PythonHelperTools.vqaTools.vqa import VQA

from keras import backend as K

K.set_learning_phase(0)


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


def to_plot(img):
    if K.image_dim_ordering() == 'tf':
        return np.rollaxis(img, 0, 1).astype(np.uint8)
    else:
        return np.rollaxis(img, 0, 3).astype(np.uint8)


def plot(img):
    plt.imshow(to_plot(img))


def verify(params):

    # get dataset -----------------------------------------------------------------------------------------------------

    ans_to_id, id_to_ans = get_most_common_answers(vqa_train, vqa_val, int(params['num_answers']), [])
    embedding_matrix, ques_val_map, ans_val_map, img_val_map, ques_to_img_val = get_val_data([], False, True)

    normalize_image_embeddings([img_val_map])
    nlp = spacy.load('en_vectors_glove_md')

    # get model -------------------------------------------------------------------------------------------------------

    model = get_model(
        dropout_rate=0,
        regularization_rate=0,
        embedding_size=int(params['embedding_size']),
        num_classes=int(params['num_answers']),
        model_name=params['model'],
        embedding_matrix=embedding_matrix)
    savedir = "models/%s_%s" % (params['model'], str(params['num_answers']))
    model.load_weights("%s/%s_epoch_%d_weights.h5" % (savedir, params['model'], params['epoch']))
    layers = model.layers

    # get info about the layer weights---------------------------------------------------------------------------------
    print "-------------------------------------------------"
    print "Layer weights:"
    print "-------------------------------------------------"
    for layer in layers:
        try:
            if layer.trainable:
                weights = layer.get_weights()
                print "Layer %s" % layer.name
                if len(weights) > 0:
                    weights = weights[0]
                    print "Shape {}".format(weights.shape)
                    print "max %f" % np.max(weights)
                    print "min %f" % np.min(weights)
                    print "mean %f" % np.mean(weights)
                    print "std %f" % np.std(weights)
                    print "sum %f" % np.sum(weights)
                print "--------------------------"
        except:
            print "Layer %s - no weights" % layer.name
            print "--------------------------"
            pass

    # get some input image and question -------------------------------------------------------------------------------

    vqa_ann_ids = vqa_val.getQuesIds()
    val_anns = vqa_val.loadQA(vqa_ann_ids)

    ann = val_anns[params['question_no']]
    question_prepro, image_prepro = ques_val_map[ann['question_id']], img_val_map[ann['image_id']]

    batch = [np.expand_dims(np.array(image_prepro), 0), np.expand_dims(np.array(question_prepro), 0)]

    pred = model.predict_on_batch(batch)
    pred_ans = np.argmax(pred[0])
    print (id_to_ans[pred_ans])

    print "-------------------------------------------------"
    print "Layer outputs:"
    print "-------------------------------------------------"
    for i in range(len(layers)):
        try:
            model_fn = K.function([layers[params['image_layer']].input, layers[params['text_layer']].input], [layers[i].output])
            layer_output = model_fn(batch)[0]
            print "Layer %s" % layers[i].name
            print "Shape {}".format(layer_output.shape)
            print "max %f" % np.max(layer_output)
            print "min %f" % np.min(layer_output)
            print "mean %f" % np.mean(layer_output)
            print "std %f" % np.std(layer_output)
            print "sum %f" % np.sum(layer_output)
            print "--------------------------"
        except:
            print "Layer %s -- cannot show output" % layers[i].name
            print "--------------------------"
            pass

    fig = plt.figure()

    vqa_val.showQA([ann])
    question = nlp(vqa_val.qqa[ann['question_id']]['question'])

    imgFilename = 'COCO_' + dataSubType_val + '_' + str(ann['image_id']).zfill(12) + '.jpg'
    if os.path.isfile(imgDir_val + imgFilename):
        I = io.imread(imgDir_val + imgFilename)
        I = scipy.misc.imresize(I, (2048, 2048), interp='nearest')

        fig.add_subplot(2, len(question), 1)
        plt.imshow(I)
        plt.xlabel(vqa_val.qqa[ann['question_id']]['question'])
        plt.xticks([])
        plt.yticks([])
    else:
        print(imgDir_val + imgFilename)
    print ("Multiple choice answer: %s" % ann['multiple_choice_answer'])

    model_fn = K.function([layers[params['image_layer']].input, layers[params['text_layer']].input],
                          [layers[params['attention_layer']].output])
    layer_output = model_fn(batch)
    layer_output = layer_output[0][0]

    layer_output_sum = np.sum(layer_output, axis=0)
    layer_output_img = scipy.misc.imresize(np.resize(layer_output_sum, (7, 7)), (2048, 2048), interp='bicubic')

    fig.add_subplot(2, len(question), 2)
    plot(I)
    plt.imshow(layer_output_img, cmap='gray', alpha=0.6)
    plt.xlabel("Ans: %s \nAns prob: %.2f" % (id_to_ans[pred_ans], np.max(pred)))
    plt.xticks([])
    plt.yticks([])

    for word_id in range(len(question) - 2):
        layer_output_img = scipy.misc.imresize(np.resize(layer_output[word_id], (7, 7)), (2048, 2048), interp='bicubic')

        print (layer_output[word_id], layer_output[word_id].shape)

        print "max %f" % np.max(layer_output[word_id])
        print "min %f" % np.min(layer_output[word_id])
        print "mean %f" % np.mean(layer_output[word_id])
        print "std %f" % np.std(layer_output[word_id])
        print "sum %f" % np.sum(layer_output[word_id])

        fig.add_subplot(2, len(question), word_id + 3)

        plot(I)
        plt.imshow(layer_output_img, cmap='gray', alpha=0.6)
        plt.xlabel("%s %s %s" % (question[word_id].text, question[word_id + 1].text, question[word_id + 2].text))
        plt.xticks([])
        plt.yticks([])

    model_fn = K.function([layers[params['image_layer']].input, layers[params['text_layer']].input],
                          [layers[params['attention_layer'] + 1].output])
    layer_output = model_fn(batch)
    layer_output = layer_output[0][0]
    layer_output_sum = np.sum(layer_output, axis=0) / len(layer_output)

    labels = ["%s\n%s\n%s\n" % (question[i].text, question[i+1].text, question[i+2].text) for i in range(len(question) - 2)]

    xlocations = np.array(range(len(layer_output_sum))) + 0.5
    width = 0.5

    fig.add_subplot(2, len(question) // 2, len(question) // 2 + 1)
    plt.bar(xlocations, layer_output_sum, width=width)
    plt.xticks(xlocations + width / 2, labels)
    plt.xlim(0, xlocations[-1] + width * 2)

    plt.show()
    # plt.savefig('attention_plot.png')
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='model_1', help='model to use for training')
    parser.add_argument('--num_answers', default=1000, type=int, help='number of top answers to classify')
    parser.add_argument('--embedding_size', default=300, type=int, help='length of the a word embedding')
    parser.add_argument('--epoch', default=40, type=int, help='epoch to evaluate')
    parser.add_argument('--question_no', default=0, type=int, help='question number to evaluate')
    parser.add_argument('--image_layer', default=0, type=int, help='index of the image layer input')
    parser.add_argument('--text_layer', default=0, type=int, help='index of the text layer input')
    parser.add_argument('--attention_layer', default=0, type=int, help='index of the attention layer output')

    args = parser.parse_args()
    params = vars(args)
    verify(params)
