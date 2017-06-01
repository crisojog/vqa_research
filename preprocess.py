import argparse
import cPickle as pickle
import os
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import spacy
import json

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


def get_most_common_answers(vqa_train, vqa_val, num_answers, ans_types, show_top_ans=False, use_test=False):
    ans_dict = {}
    annIds_train = vqa_train.getQuesIds(ansTypes=ans_types)
    anns = vqa_train.loadQA(annIds_train)
    if use_test:
        annIds_val = vqa_val.getQuesIds(ansTypes=ans_types)
        anns += vqa_val.loadQA(annIds_val)
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
        plt.bar(range(1, num_ans_plot + 1), [float(y) / total_ans * 100 for (x, y) in sorted_ans_dict[0:num_ans_plot]],
                0.9, color='b')
        plt.xticks(range(1, num_ans_plot + 1), [x for (x, y) in sorted_ans_dict[0:num_ans_plot]])
        plt.title("Most Common Answer Frequencies")
        plt.show()

    sorted_ans_dict = [x for (x, y) in sorted_ans_dict]
    sorted_ans_dict = sorted_ans_dict[0:num_answers]

    ans_to_id = dict((a, i) for i, a in enumerate(sorted_ans_dict))
    id_to_ans = dict((i, a) for i, a in enumerate(sorted_ans_dict))
    return ans_to_id, id_to_ans


def process_question(vqa, ann, nlp, question_word_vec_map, tokens_dict, question_tokens_map):
    quesId = ann['question_id']
    if quesId in question_word_vec_map:
        return question_word_vec_map[quesId], question_tokens_map[quesId]
    question = nlp(vqa.qqa[quesId]['question'])
    question_word_vec = [w.vector for w in question]

    question_len = len(question)
    question_tokens = [0] * question_len
    for i in range(question_len):
        token = question[i]
        token_l = token.lower_
        if token.has_vector and token_l in tokens_dict:
            question_tokens[i] = tokens_dict[token_l]

    return np.array(question_word_vec), np.array(question_tokens)


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


def process_img(img_model, imgId, dataSubType, imgDir, input_shape=(224, 224), output_shape=(4096,)):
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


def process_questions(vqa, data, nlp, overwrite, tokens_dict,
                      question_word_vec_map={}, question_tokens_map={}):
    filename = "data/%s_questions.pkl" % data
    filename_tokens = "data/%s_tokens_questions.pkl" % data

    if os.path.exists(filename) and os.path.exists(filename_tokens) and not overwrite:
        return question_word_vec_map, question_tokens_map

    annIds = vqa.getQuesIds()
    anns = vqa.loadQA(annIds)

    for ann in tqdm(anns):
        quesId = int(ann['question_id'])
        if quesId in question_word_vec_map:
            continue

        question, question_tokens = process_question(vqa, ann, nlp, question_word_vec_map,
                                                     tokens_dict, question_tokens_map)
        if question is None:
            continue

        question_word_vec_map[quesId] = question
        question_tokens_map[quesId] = question_tokens

    f = open(filename, "w")
    pickle.dump(question_word_vec_map, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    f = open(filename_tokens, "w")
    pickle.dump(question_tokens_map, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    return question_word_vec_map, question_tokens_map


def process_answers(vqa, data, ans_types, ans_to_id, id_to_ans, overwrite, use_all_ans, ans_map={}):
    if not ans_types:
        filename = "data/%s_answers.pkl" % data
    else:
        filename = "data/%s_answers_%s.pkl" % (data, ans_types.replace("/", ""))

    if not os.path.exists(filename) or overwrite:
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
    return ans_map


def process_images(img_model, vqa, data, data_sub_type, img_dir, img_model_name, overwrite, img_map={}):
    filename = "data/%s_images.pkl" % data

    if not os.path.exists(filename) or overwrite:
        annIds = vqa.getQuesIds()
        anns = vqa.loadQA(annIds)

        input_shape = get_input_shape(img_model_name)
        output_shape = get_output_shape(img_model_name)

        for ann in tqdm(anns):
            imgId = int(ann['image_id'])
            if imgId in img_map:
                continue

            img = process_img(img_model, ann['image_id'], data_sub_type, img_dir, input_shape, output_shape)
            if img is None:
                continue

            img_map[imgId] = img

        f = open(filename, "w")
        pickle.dump(img_map, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    return img_map


def process_ques_to_img(vqa, data, overwrite, ques_to_img={}):
    filename = "data/%s_ques_to_img.pkl" % data

    if not os.path.exists(filename) or overwrite:
        annIds = vqa.getQuesIds()
        anns = vqa.loadQA(annIds)

        for ann in tqdm(anns):
            quesId = int(ann['question_id'])
            imgId = int(ann['image_id'])
            ques_to_img[quesId] = imgId

        f = open(filename, "w")
        pickle.dump(ques_to_img, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    return ques_to_img


def process_questions_test(dataFile, data, nlp, overwrite, tokens_dict,
                           question_word_vec_map={}, question_tokens_map={}):
    filename = "data/%s_questions.pkl" % data
    filename_tokens = "data/%s_tokens_questions.pkl" % data

    if os.path.exists(filename) and os.path.exists(filename_tokens) and not overwrite:
        return

    dataset = json.load(open(dataFile, 'r'))
    for question in tqdm(dataset['questions']):
        quesId = question['question_id']
        questext = question['question']

        ques_nlp = nlp(questext)
        question_word_vec = [w.vector for w in ques_nlp]
        question_word_vec_map[quesId] = question_word_vec

        question_len = len(ques_nlp)
        question_tokens = [0] * question_len
        for i in range(question_len):
            token = ques_nlp[i]
            token_l = token.lower_
            if token.has_vector and token_l in tokens_dict:
                question_tokens[i] = tokens_dict[token_l]
        question_tokens_map[quesId] = question_tokens

    f = open(filename, "w")
    pickle.dump(question_word_vec_map, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    f = open(filename_tokens, "w")
    pickle.dump(question_tokens_map, f, pickle.HIGHEST_PROTOCOL)
    f.close()


def process_images_test(img_model, data, dataFile, dataSubType, imgDir, img_model_name, overwrite, img_map={}):
    filename = "data/%s_images.pkl" % data

    if not os.path.exists(filename) or overwrite:
        dataset = json.load(open(dataFile, 'r'))

        input_shape = get_input_shape(img_model_name)
        output_shape = get_output_shape(img_model_name)

        for question in tqdm(dataset['questions']):
            imgId = question['image_id']

            if imgId in img_map:
                continue
            img = process_img(img_model, imgId, dataSubType, imgDir, input_shape, output_shape)
            if img is None:
                continue
            img_map[imgId] = img

        f = open(filename, "w")
        pickle.dump(img_map, f, pickle.HIGHEST_PROTOCOL)
        f.close()


def process_ques_to_img_test(dataFile, data, overwrite, ques_to_img={}):
    filename = "data/%s_ques_to_img.pkl" % data

    if not os.path.exists(filename) or overwrite:
        dataset = json.load(open(dataFile, 'r'))
        for question in tqdm(dataset['questions']):
            quesId = question['question_id']
            imgId = question['image_id']
            ques_to_img[quesId] = imgId

        f = open(filename, "w")
        pickle.dump(ques_to_img, f, pickle.HIGHEST_PROTOCOL)
        f.close()


def get_most_common_tokens(vqa, nlp, tokens_dict, dataFile=None):
    if not dataFile:
        annIds = vqa.getQuesIds()
        anns = vqa.loadQA(annIds)
        for ann in tqdm(anns):
            quesId = int(ann['question_id'])
            question = nlp(vqa.qqa[quesId]['question'])
            question_tokens = [w.lower_ for w in question]
            for token in question_tokens:
                if token in tokens_dict:
                    tokens_dict[token] += 1
                else:
                    tokens_dict[token] = 1
        return

    # get tokens from the test set
    dataset = json.load(open(dataFile, 'r'))
    for question in tqdm(dataset['questions']):
        questext = question['question']

        ques_nlp = nlp(questext)
        question_tokens = [w.lower_ for w in ques_nlp]
        for token in question_tokens:
            if token in tokens_dict:
                tokens_dict[token] += 1
            else:
                tokens_dict[token] = 1


def get_tokens_dict(vqa_train, vqa_val, dataFile_test, nlp, word_embedding_dim):
    tokens_dict = {}
    get_most_common_tokens(vqa_train, nlp, tokens_dict)
    get_most_common_tokens(vqa_val, nlp, tokens_dict)
    get_most_common_tokens(None, nlp, tokens_dict, dataFile=dataFile_test)
    tokens_dict = sorted(tokens_dict.items(), key=lambda x: x[1])
    tokens_with_embedding = [(key, value) for (key, value) in tokens_dict if (nlp(key)).has_vector]

    # index 0 will be for unknown tokens or for tokens without word vectors
    index = 1
    tokens_dict = {}
    tokens_embedding = [np.array([0.] * word_embedding_dim)]
    for (key, _) in tokens_with_embedding:
        tokens_dict[key] = index
        tokens_embedding.append(nlp(key).vector)
        index += 1

    f = open("data/tokens_embedding.pkl", "w")
    pickle.dump(np.array(tokens_embedding), f, pickle.HIGHEST_PROTOCOL)
    f.close()

    return tokens_dict


def process_data(vqa_train, dataSubType_train, imgDir_train,
                 vqa_val, dataSubType_val, imgDir_val,
                 dataSubType_test, dataFile_test, imgDir_test,
                 nlp, img_model, ans_to_id, id_to_ans, params):
    ans_types = params['ans_types']
    only = params['only']
    img_model_name = params['img_model']
    overwrite = params['overwrite']
    use_all_ans = params['use_all_ans']
    use_tests = params['use_test']
    word_embedding_dim = params['word_embedding_dim']

    if only == 'all' or only == 'ques':
        print "Obtaining tokens from all datasets"
        tokens_dict = get_tokens_dict(vqa_train, vqa_val, dataFile_test, nlp, word_embedding_dim)
        print "Processing train questions"
        if not use_tests:
            process_questions(vqa_train, "train", nlp, overwrite, tokens_dict)
        else:
            ques_train_map, ques_tokens_train_map = process_questions(vqa_train, "train_val", nlp,
                                                                      overwrite, tokens_dict)
            process_questions(vqa_val, "train_val", nlp, overwrite, tokens_dict, ques_train_map, ques_tokens_train_map)
    if only == 'all' or only == 'ans':
        print "Processing train answers"
        if not use_tests:
            process_answers(vqa_train, "train", ans_types, ans_to_id, id_to_ans, overwrite, use_all_ans)
        else:
            ans_map = process_answers(vqa_train, "train_val", ans_types, ans_to_id, id_to_ans, overwrite, use_all_ans)
            process_answers(vqa_val, "train_val", ans_types, ans_to_id, id_to_ans, overwrite, use_all_ans, ans_map)
    if only == 'all' or only == 'img':
        print "Processing train images"
        if not use_tests:
            process_images(img_model, vqa_train, "train", dataSubType_train, imgDir_train, img_model_name, overwrite)
        else:
            img_map = process_images(img_model, vqa_train, "train_val", dataSubType_train, imgDir_train, img_model_name,
                                     overwrite)
            process_images(img_model, vqa_val, "train_val", dataSubType_val, imgDir_val, img_model_name, overwrite,
                           img_map)
    if only == 'all' or only == 'ques_to_img':
        print "Processing train question id to image id mapping"
        if not use_tests:
            process_ques_to_img(vqa_train, "train", overwrite)
        else:
            ques_to_img = process_ques_to_img(vqa_train, "train_val", overwrite)
            process_ques_to_img(vqa_val, "train_val", overwrite, ques_to_img)
    print "Done"

    # -------------------------------------------------------------------------------------------------

    if only == 'all' or only == 'ques':
        print "Processing validation questions"
        if not use_tests:
            process_questions(vqa_val, "val", nlp, overwrite, tokens_dict)
        else:
            process_questions_test(dataFile_test, "test", nlp, overwrite, tokens_dict)
    if only == 'all' or only == 'ans':
        print "Processing validation answers"
        if not use_tests:
            process_answers(vqa_val, "val", ans_types, ans_to_id, id_to_ans, overwrite, use_all_ans)
        else:
            print "Skipping answers for test set"
    if only == 'all' or only == 'img':
        print "Processing validation images"
        if not use_tests:
            process_images(img_model, vqa_val, "val", dataSubType_val, imgDir_val, img_model_name, overwrite)
        else:
            process_images_test(img_model, "test", dataFile_test, "test2015", imgDir_test,
                                img_model_name, overwrite)
    if only == 'all' or only == 'ques_to_img':
        print "Processing validation question id to image id mapping"
        if not use_tests:
            process_ques_to_img(vqa_val, "val", overwrite)
        else:
            process_ques_to_img_test(dataFile_test, "test", overwrite)
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

    dataSubType_test = 'test-dev2015'  # Hardcoded for test-dev
    quesFile_test = '%s/Questions/%s_%s_%s_questions.json' % (dataDir, taskType, dataType, dataSubType_test)
    imgDir_test = '%s/Images/%s/%s/' % (dataDir, dataType, 'test2015')

    nlp = spacy.load('en_vectors_glove_md')

    ans_to_id, id_to_ans = get_most_common_answers(vqa_train, vqa_val, int(params['num_answers']), params['ans_types'],
                                                   params['show_top_ans'], params['use_test'])
    img_model = get_img_model(params['img_model'])

    process_data(vqa_train, dataSubType_train, imgDir_train,
                 vqa_val, dataSubType_val, imgDir_val,
                 dataSubType_test, quesFile_test, imgDir_test,
                 nlp, img_model, ans_to_id, id_to_ans, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--ans_types', default=[], help='filter questions with specific answer types')
    parser.add_argument('--num_answers', default=1000, type=int, help='number of top answers to classify')
    parser.add_argument('--word_embedding_dim', default=300, type=int, help='word embedding dimension for one word')
    parser.add_argument('--img_model', default='resnet50', help='which image model to use for embeddings')
    parser.add_argument('--only', default='all', help='which data to preprocess (all, ques, ans, img, ques_to_img)')
    parser.add_argument('--use_test', dest='use_test', action='store_true',
                        help='use test set (which also means training on train+val')
    parser.set_defaults(use_test=False)
    parser.add_argument('--show_top_ans', dest='show_top_ans', action='store_true', help='show plot with top answers')
    parser.set_defaults(show_top_ans=False)
    parser.add_argument('--overwrite', dest='overwrite', action='store_true', help='force overwrite')
    parser.set_defaults(overwrite=False)
    parser.add_argument('--use_all_ans', dest='use_all_ans', action='store_true',
                        help='use all answers for training, otherwise use only the multiple choice answer')
    parser.set_defaults(use_all_ans=False)

    args = parser.parse_args()
    params = vars(args)
    main(params)
