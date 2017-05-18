import argparse
import cPickle as pickle
import os
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import spacy
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.preprocessing import image
from numpy import linalg as LA
from tqdm import tqdm

from VQA.PythonHelperTools.vqaTools.vqa import VQA
from imagenet_utils import preprocess_input
from utils import softmax


def get_img_model(img_model_type):
    if img_model_type == "vgg19":
        print ("Loading VGG19 model")
        base_model = VGG19(weights='imagenet', include_top=True)
        return Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
    elif img_model_type == "vgg19_multi":
        print ("Loading VGG19-early-cut model")
        return VGG19(weights='imagenet', include_top=False)
    elif img_model_type == "resnet50":
        print ("Loading ResNet50 model")
        return ResNet50(weights='imagenet', include_top=False)
    elif img_model_type == "resnet50_multi":
        print ("Loading ResNet50-early-cut model")
        base_model = ResNet50(weights='imagenet', include_top=False)
        return Model(inputs=base_model.input, outputs=base_model.layers[-2].output)


def get_most_common_answers(vqa_train, num_answers, ans_types, show_top_ans=False):
    ans_dict = {}
    annIds = vqa_train.getQuesIds(ansTypes=ans_types)
    anns = vqa_train.loadQA(annIds)
    for ann in anns:
        for ans in ann['answers']:
            answer = ans['answer'].lower()
            if answer in ans_dict:
                ans_dict[answer] += 1
            else:
                ans_dict[answer] = 1
    sorted_ans_dict = sorted(ans_dict.items(), key=itemgetter(1), reverse=True)

    if show_top_ans:
        # Some bar plots
        num_ans_plot = 20
        total_ans = 0
        for (x, y) in sorted_ans_dict: total_ans += y
        plt.bar(range(1, num_ans_plot + 1), [float(y) / total_ans * 100 for (x, y) in sorted_ans_dict[0:num_ans_plot]], 0.9,
                color='b')
        plt.xticks(range(1, num_ans_plot + 1), [x for (x, y) in sorted_ans_dict[0:num_ans_plot]])
        plt.title("Most Common Answer Frequencies")
        plt.show()

    sorted_ans_dict = [x for (x, y) in sorted_ans_dict]
    sorted_ans_dict = sorted_ans_dict[0:num_answers]

    ans_to_id = dict((a, i) for i, a in enumerate(sorted_ans_dict))
    id_to_ans = dict((i, a) for i, a in enumerate(sorted_ans_dict))
    return ans_to_id, id_to_ans


def process_question(vqa, ann, nlp, question_word_vec_map):
    quesId = ann['question_id']
    if quesId in question_word_vec_map:
        return question_word_vec_map[quesId]
    question = nlp(vqa.qqa[quesId]['question'])
    question_word_vec = [w.vector for w in question]

    return np.array(question_word_vec)


def process_answer(ann, data, ans_map, ans_to_id, id_to_ans, use_all_ans=False):
    quesId = ann['question_id']
    if quesId in ans_map:
        return ans_map[quesId]
    encoding = np.zeros(len(id_to_ans))
    if not use_all_ans:
        answer = ann['multiple_choice_answer'].lower()
        if answer in ans_to_id:
            encoding[ans_to_id[answer]] = 1
            return encoding
        elif data == "val":
            return np.zeros(len(id_to_ans))
        else:
            return None
    else:
        for ans in ann['answers']:
            answer = ans['answer'].lower()
            if answer in ans_to_id:
                encoding[ans_to_id[answer]] += 1
        if np.sum(encoding) > 0:
            #encoding /= np.sum(encoding)
            encoding = softmax(encoding)
            return encoding
        elif data == "val":
            return np.zeros(len(id_to_ans))
        else:
            return None


def process_img(img_model, ann, dataSubType, imgDir, input_shape=(224, 224), output_shape=(4096,)):
    imgId = ann['image_id']
    imgFilename = 'COCO_' + dataSubType + '_' + str(imgId).zfill(12) + '.jpg'
    if os.path.isfile(imgDir + imgFilename):
        img = image.load_img(imgDir + imgFilename, target_size=input_shape)
        x = image.img_to_array(img)
        x = preprocess_input(x)
        features = img_model.predict(np.array([x]))
        features = np.reshape(features[0], output_shape)
        features /= LA.norm(features, 2)
        return features
    else:
        return None


def get_input_shape(img_model_name):
    if img_model_name == 'inception_v3':
        return (299, 299)
    return (224, 224)


def get_output_shape(img_model_name):
    if img_model_name == 'vgg19':
        return (4096,)
    elif img_model_name == 'vgg19_multi':
        return (512, 49)
    elif img_model_name == 'resnet50' or img_model_name == 'inception_v3':
        return (2048,)
    elif img_model_name == 'resnet50_multi':
        return (2048, 49)


def process_questions(vqa, data, nlp, overwrite):
    filename = "data/%s_questions.pkl" % data
    if not os.path.exists(filename) or overwrite:
        question_word_vec_map = {}
        annIds = vqa.getQuesIds()
        anns = vqa.loadQA(annIds)

        for ann in tqdm(anns):
            quesId = int(ann['question_id'])
            if quesId in question_word_vec_map:
                continue

            question = process_question(vqa, ann, nlp, question_word_vec_map)
            if question is None:
                continue

            question_word_vec_map[quesId] = question

        f = open(filename, "w")
        pickle.dump(question_word_vec_map, f, pickle.HIGHEST_PROTOCOL)
        f.close()


def process_answers(vqa, data, ans_types, ans_to_id, id_to_ans, overwrite, use_all_ans):
    if not ans_types:
        filename = "data/%s_answers.pkl" % data
    else:
        filename = "data/%s_answers_%s.pkl" % (data, ans_types.replace("/", ""))

    if not os.path.exists(filename) or overwrite:
        ans_map = {}
        annIds = vqa.getQuesIds(ansTypes=ans_types)
        anns = vqa.loadQA(annIds)

        for ann in tqdm(anns):
            quesId = int(ann['question_id'])
            if quesId in ans_map:
                continue

            answer = process_answer(ann, data, ans_map, ans_to_id, id_to_ans, use_all_ans)
            if answer is None:
                continue

            ans_map[quesId] = answer.tolist()

        f = open(filename, "w")
        pickle.dump(ans_map, f, pickle.HIGHEST_PROTOCOL)
        f.close()


def process_images(img_model, vqa, data, data_sub_type, img_dir, img_model_name, overwrite,
                   avg=None, use_translation=False):
    filename = "data/%s_images.pkl" % data

    if not os.path.exists(filename) or overwrite:
        img_map = {}
        annIds = vqa.getQuesIds()
        anns = vqa.loadQA(annIds)

        input_shape = get_input_shape(img_model_name)
        output_shape = get_output_shape(img_model_name)

        rolling_sum = None
        num_images = 0
        for ann in tqdm(anns):
            imgId = int(ann['image_id'])
            if imgId in img_map:
                continue

            img = process_img(img_model, ann, data_sub_type, img_dir, input_shape, output_shape)
            if img is None:
                continue
            if rolling_sum is None:
                rolling_sum = img
            else:
                rolling_sum += img
            num_images += 1

            img_map[imgId] = img

        if avg is None:
            avg = rolling_sum / num_images

        if use_translation:
            for ann in anns:
                imgId = int(ann['image_id'])
                img_map[imgId] = np.array(img_map[imgId]) - avg
                img_map[imgId] /= LA.norm(img_map[imgId], 2)
                img_map[imgId] = img_map[imgId].tolist()

        f = open(filename, "w")
        pickle.dump(img_map, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        return avg


def process_ques_to_img(vqa, data, overwrite):
    filename = "data/%s_ques_to_img.pkl" % data

    if not os.path.exists(filename) or overwrite:
        ques_to_img = {}
        annIds = vqa.getQuesIds()
        anns = vqa.loadQA(annIds)

        for ann in tqdm(anns):
            quesId = int(ann['question_id'])
            imgId = int(ann['image_id'])
            ques_to_img[quesId] = imgId

        f = open(filename, "w")
        pickle.dump(ques_to_img, f, pickle.HIGHEST_PROTOCOL)
        f.close()


def process_data(vqa_train, dataSubType_train, imgDir_train, vqa_val, dataSubType_val, imgDir_val,
                 nlp, img_model, ans_types, ans_to_id, id_to_ans, only, img_model_name, overwrite,
                 use_all_ans, use_translation):

    avg = None
    if only == 'all' or only == 'ques':
        print "Processing train questions"
        process_questions(vqa_train, "train", nlp, overwrite)
    if only == 'all' or only == 'ans':
        print "Processing train answers"
        process_answers(vqa_train, "train", ans_types, ans_to_id, id_to_ans, overwrite, use_all_ans)
    if only == 'all' or only == 'img':
        print "Processing train images"
        avg = process_images(img_model, vqa_train, "train", dataSubType_train, imgDir_train, img_model_name, overwrite,
                             use_translation=use_translation)
    if only == 'all' or only == 'ques_to_img':
        print "Processing train question id to image id mapping"
        process_ques_to_img(vqa_train, "train", overwrite)
    print "Done"


    if only == 'all' or only == 'ques':
        print "Processing validation questions"
        process_questions(vqa_val, "val", nlp, overwrite)
    if only == 'all' or only == 'ans':
        print "Processing validation answers"
        process_answers(vqa_val, "val", ans_types, ans_to_id, id_to_ans, overwrite, use_all_ans)
    if only == 'all' or only == 'img':
        print "Processing validation images"
        process_images(img_model, vqa_val, "val", dataSubType_val, imgDir_val, img_model_name, overwrite, avg,
                       use_translation=use_translation)
    if only == 'all' or only == 'ques_to_img':
        print "Processing validation question id to image id mapping"
        process_ques_to_img(vqa_val, "val", overwrite)
    print "Done"


def main(params):
    dataDir = 'VQA'
    taskType = 'OpenEnded'
    dataType = 'mscoco'

    dataSubType_train = 'train2014'
    annFile_train = '%s/Annotations/%s_%s_annotations.json' % (dataDir, dataType, dataSubType_train)
    quesFile_train = '%s/Questions/%s_%s_%s_questions.json' % (dataDir, taskType, dataType, dataSubType_train)
    imgDir_train = '%s/Images/%s/%s/' % (dataDir, dataType, dataSubType_train)
    vqa_train = VQA(annFile_train, quesFile_train)

    dataSubType_val = 'val2014'
    annFile_val = '%s/Annotations/%s_%s_annotations.json' % (dataDir, dataType, dataSubType_val)
    quesFile_val = '%s/Questions/%s_%s_%s_questions.json' % (dataDir, taskType, dataType, dataSubType_val)
    imgDir_val = '%s/Images/%s/%s/' % (dataDir, dataType, dataSubType_val)
    vqa_val = VQA(annFile_val, quesFile_val)

    nlp = spacy.load('en_vectors_glove_md') #spacy.load('en', vectors='en_glove_cc_300_1m_vectors')

    ans_to_id, id_to_ans = get_most_common_answers(vqa_train, int(params['num_answers']), params['ans_types'],
                                                   params['show_top_ans'])
    img_model = get_img_model(params['img_model'])
    process_data(vqa_train, dataSubType_train, imgDir_train, vqa_val, dataSubType_val, imgDir_val,
                 nlp, img_model, params['ans_types'], ans_to_id, id_to_ans, params['only'], params['img_model'],
                 params['overwrite'], params['use_all_ans'], params['use_translation'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--ans_types', default=[], help='filter questions with specific answer types')
    parser.add_argument('--num_answers', default=1000, type=int, help='number of top answers to classify')
    parser.add_argument('--img_model', default='resnet50', help='which image model to use for embeddings')
    parser.add_argument('--only', default='all', help='which data to preprocess (all, ques, ans, img, ques_to_img)')
    parser.add_argument('--show_top_ans', default=False, help='show plot with top answers')
    parser.add_argument('--overwrite', default=False, type=bool, help='force overwrite')
    parser.add_argument('--use_all_ans', default=False, type=bool, help='use all answers for training or only multiple'
                                                                        ' choice answer')
    parser.add_argument('--use_translation', default=False, type=bool, help='translate images by subtracting average')

    args = parser.parse_args()
    params = vars(args)
    main(params)
