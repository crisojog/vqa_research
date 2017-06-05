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

Run the notebook VQA_Keras_Models_and_Prepro.ipynb via
```bash
ipython notebook
```
And navigate to the mentioned file in your browser
