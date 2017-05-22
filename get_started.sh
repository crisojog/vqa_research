#!/bin/bash

git clone https://github.com/VT-vision-lab/VQA.git

mkdir data
mkdir models

cd VQA
mkdir Annotations
mkdir Images
mkdir Questions

touch __init__.py
touch PythonEvaluationTools/__init__.py
touch PythonHelperTools/__init__.py

cd Images
mkdir mscoco; cd mscoco

wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
unzip -q train2014.zip
rm train2014.zip

wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
unzip -q val2014.zip
rm val2014.zip

wget http://msvocds.blob.core.windows.net/coco2015/test2015.zip
unzip -q test2015.zip
rm test2015.zip

cd ../../Questions

wget http://visualqa.org/data/mscoco/vqa/Questions_Train_mscoco.zip
unzip -q Questions_Train_mscoco.zip
rm Questions_Train_mscoco.zip

wget http://visualqa.org/data/mscoco/vqa/Questions_Val_mscoco.zip
unzip -q Questions_Val_mscoco.zip
rm Questions_Val_mscoco.zip

wget http://visualqa.org/data/mscoco/vqa/Questions_Test_mscoco.zip
unzip -q Questions_Test_mscoco.zip
rm Questions_Test_mscoco.zip

cd ../Annotations

wget http://visualqa.org/data/mscoco/vqa/Annotations_Train_mscoco.zip
unzip -q Annotations_Train_mscoco.zip
rm Annotations_Train_mscoco.zip

wget http://visualqa.org/data/mscoco/vqa/Annotations_Val_mscoco.zip
unzip -q Annotations_Val_mscoco.zip
rm Annotations_Val_mscoco.zip

cd ../..
