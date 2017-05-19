import numpy as np
import argparse
import skimage.io as io
import os
import matplotlib.pyplot as plt
import scipy
import spacy

from model_utils import get_model
from utils import get_train_data, get_val_data
from preprocess import get_most_common_answers
from VQA.PythonHelperTools.vqaTools.vqa import VQA

from keras import backend as K


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


def verify(params):

    # get dataset -----------------------------------------------------------------------------------------------------

    #ans_to_id, id_to_ans = get_most_common_answers(vqa_train, int(params['num_answers']), [])
    #ques_train_map, ans_train_map, img_train_map, ques_to_img_train = get_train_data([])
    ques_val_map, ans_val_map, img_val_map, ques_to_img_val = get_val_data([])
    nlp = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')

    # get model -------------------------------------------------------------------------------------------------------

    model = get_model(
        dropout_rate=0,
        regularization_rate=0,
        embedding_size=int(params['embedding_size']),
        num_classes=int(params['num_answers']),
        model_name=params['model'])

    layers = model.layers
    model_fn = K.function([layers[2].input, layers[0].input], [layers[10].output])

    # get info about the layer weights---------------------------------------------------------------------------------
    weights = layers[-1].get_weights()[0]

    print "max %f" % np.max(weights)
    print "min %f" % np.min(weights)
    print "mean %f" % np.mean(weights)
    print "std %f" % np.std(weights)

    '''
    plt.hist(weights, bins='auto')
    plt.title("Weights Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    '''
    # get some input image and question -------------------------------------------------------------------------------

    vqa_ann_ids = vqa_val.getQuesIds()
    val_anns = vqa_val.loadQA(vqa_ann_ids)

    fig = plt.figure()
    ann = val_anns[8] #choose randomish question
    word_id = 2

    question_prepro, image_prepro = ques_val_map[ann['question_id']], img_val_map[ann['image_id']]
    question_nlp = nlp(vqa_val.qqa[ann['question_id']]['question'])

    vqa_val.showQA([ann])
    imgFilename = 'COCO_' + dataSubType_val + '_' + str(ann['image_id']).zfill(12) + '.jpg'
    if os.path.isfile(imgDir_val + imgFilename):
        I = io.imread(imgDir_val + imgFilename)
        I = scipy.misc.imresize(I, (700, 700), interp='nearest')
        fig.add_subplot(2,1,1)
        plt.imshow(I)
        plt.axis('off')
    else:
        print(imgDir_val + imgFilename)

    input = [np.expand_dims(np.array(question_prepro), 0), np.expand_dims(np.array(image_prepro), 0)]
    layer_output = model_fn(input)
    layer_output_img = scipy.misc.imresize(np.resize(layer_output[0][0][word_id], (7, 7)), (700, 700), interp='nearest')

    fig.add_subplot(2, 1, 2)
    plt.imshow(layer_output_img, cmap="cool")
    plt.xlabel(question_nlp[word_id].text)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='model_1', help='model to use for training')
    parser.add_argument('--weights', default='models/model_1_epoch_45_weights.h5',
                        help='path towards the saved model weights')
    parser.add_argument('--num_answers', default=1000, type=int, help='number of top answers to classify')
    parser.add_argument('--embedding_size', default=300, type=int, help='length of the a word embedding')

    args = parser.parse_args()
    params = vars(args)
    verify(params)
