# RSNA Pneumonia Detection Challenge
## A Kaggle competition
This project is an attempt to solve the Pneumonia Detection Challenge by RSNA on Kaggle. For more details, check the official competition page - [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

## Description
The goal is to detect regions of <b>chest radiograph</b> images that could suggest visual signal for pneumonia.

## Course of action
The task sounds very similar to object detection so it may be a good idea to experiment with state of the art object detection models.

### EDA
Of course, it's best to start with some exploratory data analysis. Some graphs and stats could be found in `eda.py` - it's a simple python file. It's run in `Atom` with [`Hydrogen`](https://atom.io/packages/hydrogen) for simplicity and faster iteration as well as history tracking.

#### Input
The input is an archive of `DICOM` files. Once extracted, the files can be easily read with e.g. `pydicom` and have their image data extracted. By default, the `pydicom` module returns a `numpy` array. In this case it's of `uint8` greyscale values (a single channel image).

#### Bounding boxes
The distribution of the bounding boxes' coordinates has roughly the shape of lungs, which is probably expected.

<p align="center">
  <img width="600" src="https://github.com/freespirit/RSNA-Pneumonia-Detection-Challenge/blob/master/screenshots/box_coordinates.png"/>
</p>

Width and height distributions are not very interesting but they may give an idea of what to expect from the model (as an output):

<p align="center">
  <img width="600" src="https://github.com/freespirit/RSNA-Pneumonia-Detection-Challenge/blob/master/screenshots/box_sides.png"/>
</p>

The distribution of the ratios (`height / width`) - it looks interesting that a lot of the boxes are of square shape. Most of the boxes have aspect ratio in the range [1, 2].

<p align="center">
  <img width="600" src="https://github.com/freespirit/RSNA-Pneumonia-Detection-Challenge/blob/master/screenshots/box_shapes.png"/>
</p>

Exploring the number of positive and negative samples shows that there aren't as many positive samples, which could probably make it harder for the model to predict the bounding boxes.

<p align="center">
  <img width="480" src="https://github.com/freespirit/RSNA-Pneumonia-Detection-Challenge/blob/master/screenshots/samples.png"/>
</p>

Positive: 8964
Negative: 20025

### Model
The precision is more important than speed in this task. Probably even in a real-world application of the model for similar task higher frames per second wouldn't be as important as the precision. Therefore it's better to use more accurate than faster model. As such, `RetinaNet` would be the first choice.

The implementation of the model is based on "Focal Loss for Dense Object Detection" (arXiv:1708.02002v2 \[cs.CV\] 7 Feb 2018)

### Results
By the end of the challenge the model can't be used for meaningful predictions. The main reason to fail is the predominant focus on the model’s architecture. This lead to spending less time on the pipeline, predictions and loss. By the time the architecture was ready it turned out it’s too heavy and slow to iterate, which made experiments a lot harder.

Yet, the main reason to join the challenge was the practical experience with Tensorflow. The plan was to get in the depths of TF by implementing a SOTA model. From this perspective, the work on the challenge was very useful and productive and fulfilled its goal.

Key __takeaway__:
It's better to start with the pipeline, e.g. input and output functions, as well as the output (predictions) of the model and the loss function. Those should be "connected" with a small and fast model used just to confirm the structure of the whole project and the proper flow of data. Then, the model can be extended or replaced with a more complicated one. Of course, subsets of the input can (and should) be used for faster iterations and experiments with the proper model.


To better understand the model (RetinaNet):
 1. “Focal Loss for Dense Object Detection” - arXiv:1708.02002v2 [cs.CV] 7 Feb 2018
 2. “Feature Pyramid Networks for Object Detection" - arXiv:1612.03144v2 [cs.CV] 19 Apr 2017
 3. “Deep Residual Learning for Image Recognition” - arXiv:1512.03385v1 [cs.CV] 10 Dec 2015
