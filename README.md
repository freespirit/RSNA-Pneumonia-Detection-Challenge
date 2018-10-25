# RSNA Pneumonia Detection Challenge
## A Kaggle competition
This project is an attempt to solve the Pneumonia Detection Challenge by RSNA on Kaggle. For more details, check the official competition page - [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

## Description
The goal is to detect regions of <b>chest radiograph</b> images that could suggest visual signal for pneumonia.

## Course of action
The task sounds very similar to object detection so it may be a good idea to experiment with state of the art object detection models.

### EDA
Of course, it's best to start with some exploratory data analysis. Some graphs and stats could be found in `eda.py` - it's a simple python file. It's run in `Atom` with [`Hydrogen`](https://atom.io/packages/hydrogen) for simplicity and faster iteration as well as history tracking.

### Model
The precision is more important than speed in this task. Probably even in a real-world application of the model for similar task higher frames per second wouldn't be as important as the precision. Therefore it's better to use more accurate than faster model. As such, `RetinaNet` would be the first choice.

The implementation of the model is based on "Focal Loss for Dense Object Detection" (arXiv:1708.02002v2 \[cs.CV\] 7 Feb 2018)
