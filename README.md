# RSNA Pneumonia Detection Challenge
## A Kaggle competition
This project is an attempt to solve the Pneumonia Detection Challenge by RSNA on Kaggle. For more details, check the official competition page - [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

## Description
The goal is to detect regions of <b>chest radiograph</b> images that could suggest visual signal for pneumonia.

## Course of action
The task sounds very similar to object detection so it may be a good idea to experiment with state of the art object detection models.

### EDA
Of course, it's best to start with some exploratory data analysis. Some graphs and stats could be found in `eda.py` - it's a simple python file. It's run in `Atom` with `Hydrogen` for simplicity and faster iteration as well as history tracking.

### YOLO
The first approach could be using/implementing YOLO and use it "out of the box".

Second stage would be an attempt of transfer learning (again with YOLO, for example) where a pre-trained model is used but it's final (couple of) layers are re-trained again with the available data from the competition.

### RetinaNet
An alternative approach would be to skip the YOLO implementation altogether and go for [RetinaNet-101](https://arxiv.org/pdf/1708.02002.pdf).
According to both the RetinaNet-101 and [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) papers the RetinaNet-101 model is noticeable more accurate although slower. In this competition accuracy may be more important than speed.
