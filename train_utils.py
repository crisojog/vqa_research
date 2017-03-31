import numpy as np
import json

from tqdm import tqdm
from VQA.PythonHelperTools.vqaTools.vqa import VQA
from VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval

dataDir = 'VQA'
taskType = 'OpenEnded'
dataType = 'mscoco'  # 'mscoco' for real and 'abstract_v002' for abstract

# constants for evaluation

dataSubType = 'val2014'
annFile = '%s/Annotations/%s_%s_annotations.json' % (dataDir, dataType, dataSubType)
quesFile = '%s/Questions/%s_%s_%s_questions.json' % (dataDir, taskType, dataType, dataSubType)
imgDir = '%s/Images/%s/%s/' % (dataDir, dataType, dataSubType)
resultType = 'eval'
fileTypes = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType']

[resFile, accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = \
    ['%s/Results/%s_%s_%s_%s_%s.json' % (dataDir, taskType, dataType, dataSubType,
                                         resultType, fileType) for fileType in fileTypes]


def get_batch(batch, batch_size, ques_map, ans_map, img_map, ques_ids, ques_to_img):
    # get ids in the current batch
    batch_ids = ques_ids[batch * batch_size: min((batch + 1) * batch_size, len(ques_ids))]
    # filter out ids which don't have question, answer or image
    batch_ids = [batch_id for batch_id in batch_ids if batch_id in ques_map and batch_id in ans_map and ques_to_img[batch_id] in img_map]

    # add questions to batch
    batch_questions = [ques_map[batch_id] for batch_id in batch_ids]
    batch_answers = [ans_map[batch_id] for batch_id in batch_ids]
    batch_images = [img_map[ques_to_img[batch_id]] for batch_id in batch_ids]

    # find out maximum length of a question in this batch
    max_len = max([len(ques) for ques in batch_questions])
    # ... and pad all questions in the batch to that length (more efficient than padding all questions to a single maximum length)
    batch_ques_aligned = []
    for question in batch_questions:
        if len(question) < max_len:
            batch_ques_aligned.append(np.append(question, np.zeros((max_len - len(question), 300)), axis=0))
        else:
            batch_ques_aligned.append(question)
    # finally, construct train_X, and train_y
    train_X = [np.array(batch_images), np.array(batch_ques_aligned)]
    train_y = np.array(batch_answers)
    return train_X, train_y


def train_epoch(
        epoch_no,
        model,
        num_batches,
        batch_size,
        ques_map,
        ans_map,
        img_map,
        ques_ids,
        ques_to_img):
    # shuffle all question ids on each epoch
    np.random.shuffle(ques_ids)

    loss, accuracy, total = .0, .0, .0
    for batch in tqdm(range(num_batches), desc="Train epoch %d" % epoch_no):
        train_X, train_y = get_batch(batch, batch_size, ques_map, ans_map, img_map, ques_ids, ques_to_img)
        total += len(train_y)
        # ... and train model with the batch
        l, a = model.train_on_batch(train_X, train_y)
        loss += l * len(train_y)
        accuracy += a * len(train_y)
    loss /= total
    accuracy /= total
    print("Train loss: {}\tAccuracy: {}".format(loss, accuracy))
    return loss, accuracy


def val_epoch(
        epoch_no,
        model,
        num_batches,
        batch_size,
        ques_map,
        ans_map,
        img_map,
        ques_ids,
        ques_to_img):
    loss, accuracy, total = .0, .0, .0
    for batch in tqdm(range(num_batches), desc="Val epoch %d" % epoch_no):
        val_X, val_y = get_batch(batch, batch_size, ques_map, ans_map, img_map, ques_ids, ques_to_img)
        total += len(val_y)
        l, a = model.test_on_batch(val_X, val_y)
        loss += l * len(val_y)
        accuracy += a * len(val_y)
    loss /= total
    accuracy /= total
    print("Val loss: {}\tAccuracy: {}".format(loss, accuracy))
    return loss, accuracy


def process_question_batch(model, questions, question_ids, id_to_ans, images, results):
    # find out maximum length of a question in this batch
    max_len = max([len(ques) for ques in questions])
    # ... and pad all questions in the batch to that length (more efficient than padding all questions to a single maximum length)
    ques_aligned = []
    for question in questions:
        if len(question) < max_len:
            ques_aligned.append(np.append(question, np.zeros((max_len - len(question), 300)), axis=0))
        else:
            ques_aligned.append(question)
    val_X = [np.array(images), np.array(ques_aligned)]

    predicted_y = model.predict_on_batch(val_X)
    # add results to map
    for ans, question_id in zip(predicted_y, question_ids):
        res = {}
        res['question_id'] = int(question_id)
        # Get the best answer via argmax
        res['answer'] = id_to_ans[np.argmax(ans)]
        # Get the best answer via sampling
        # res['answer'] = id_to_ans[np.random.choice(range(len(ans)), p=ans)]
        results.append(res)


def print_accuracies(vqaEval):
    print "\n"
    print "Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall'])
    print "Per Question Type Accuracy is the following:"
    for quesType in vqaEval.accuracy['perQuestionType']:
        print "%s : %.02f" % (quesType, vqaEval.accuracy['perQuestionType'][quesType])
    print "\n"
    print "Per Answer Type Accuracy is the following:"
    for ansType in vqaEval.accuracy['perAnswerType']:
        print "%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType])
    print "\n"


def evaluate(
        model,
        vqa,
        batch_size,
        ques_map,
        img_map,
        id_to_ans,
        ans_types,
        verbose=False):
    annIds = vqa.getQuesIds()
    anns = vqa.loadQA(annIds)

    questions = []
    question_ids = []
    images = []

    results = []
    for ann in tqdm(anns):
        questions.append(ques_map[ann['question_id']])
        question_ids.append(ann['question_id'])
        images.append(img_map[ann['image_id']])
        if len(questions) == batch_size:
            process_question_batch(model, questions, question_ids, id_to_ans, images, results)
            # clear arrays
            questions, question_ids, images = [], [], []
    if len(questions) > 0:
        process_question_batch(model, questions, question_ids, id_to_ans, images, results)

    # save results as a json
    with open(resFile, "w") as outfile:
        json.dump(results, outfile)

    # create vqa object and vqaRes object
    vqa_ann = VQA(annFile, quesFile)
    vqaRes = vqa_ann.loadRes(resFile, quesFile)

    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa_ann, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2

    vqaEval.evaluate()

    if verbose:
        print_accuracies(vqaEval)

    return vqaEval.accuracy['overall']