# Visual QA

## Dependencies
This code was run on Anaconda Python 2.7.3

To run the code, you will need the following:
* Keras 2.0 (with either Theano or Tensorflow)
* Numpy
* Scikit-learn
* Scikit-image
* Matplotlib
* Spacy (after installing run ```python -m spacy download en_vectors_glove_md```)
* Tqdm
* Pickle
(I sincerely hope I covered all dependencies)

## How to run
Start by getting the dataset and VQA API and unpacking it via
```bash
chmod a+x get_started.sh
./get_started.sh
```

Before running, we need to preprocess the data
```
python preprocess.py --img_model resnet152_multi --num_answers 3000
```
The image models can be vgg19, resnet50, resnet152 or inception for the baseline models and the \_multi variants for the dual_att model. 

If you want to preprocess the data in order to train on the train + val sets and test on the test-dev set, then use the ```--use_test``` parameter when preprocessing, training and testing.

To train, run
```
python train.py --model dual_att --num_answers 3000 --eval_every 2 --decay_every 15 --decay -0.025 --batch_size 128
```
in order to achieve the best results.

To evaluate a specific model trained on train + val, in order to upload the test-dev results on codalab run
```
python train.py --model dual_att --num_answers 3000 --eval_every 2 --decay_every 15 --decay -0.025 --use_test --eval_only --eval_epoch 120 --batch_size 128
```
where the ```--eval_epoch``` parameter should tell the epoch to use when evaluating. This snapshot must exist in the models directory.

## To run preprocessing with ResNet152
Create a folder called ```imagenet_models```<br/>
Download the appropriate pre-trained weights from https://github.com/lef-fan/cnn_finetune and place the file in ```imagenet_models```<br/>

## Models
### Baseline
![Alt text](baseline_model.png?raw=true "Baseline model")

### Baseline CNN
![Alt text](baseline_model_cnn.png?raw=true "Baseline CNN model")

### Dual ATT
![Alt text](dual-attention-model-v2.png?raw=true "Dual Attention model")

## Results
![Alt text](results.png?raw=true "Results on val and test-dev sets")

## Examples
![Alt text](examples.png?raw=true "Examlpe outputs from the dual-att model")
