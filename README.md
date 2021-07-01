# Alzheimer’s disease detection using Convolutional Neural Networks
Project done by Fernando Herrán Albelda (Master in Data Science for KSchool). 

An [interactive application](https://alzheimer-detection.herokuapp.com/) using Streamlit has been build for this thesis and can be used by the user to predict if a patient has signs of Alzheimer uploading a MRI of his brain. Github repository of this application can be found [here](https://github.com/fernandoherran/thesis-streamlit-app).

## Introduction

### Objective
The goal of this thesis is to build a 3D - Convolutional Neural Network to classify patients with Alzheimer's disease (AD) or cognitively normal (CN) using Magnetic Resonance Images (MRI) of their brains. The problem consists of a binary classification and is justified by the need to learn about the use of Deep Learning to solve problems where images are related, and to understand the impact of the different layers in the performance of the neural network.

### Requirements
Next libraries are used in the code:
- [nibabel](https://nipy.org/nibabel/)
- [deepbrain](https://pypi.org/project/deepbrain/)
- [cv2](https://docs.opencv.org/master/d0/de3/tutorial_py_intro.html)
- [tensorflow](https://www.tensorflow.org/tutorials)
- Visualization: matplotlib, seaborn
- Others: os, sys, time, numpy, pandas, random, scipy, shutil, sklearn,tqdm, functools, skimage, gzip

### Set-up virtual environment
In order to run the code without having dependencies problems, user can create a virtual environment with all needed packages. To do that, please follow process below: 
- Download `requirements.txt` file found in this master branch.
- Open terminal, and run below command (tt will create a virtual environment in your computer):

```
python -m venv virtual_env
```
- Once created, user can activate the virtual environment running the following command:
```
 source virtual_env/bin/activate
```

- Finally, user can install all needed dependencies in the virtual environment running the following command:
```
pip install -r requirements.txt
```

## Repository

This repository has 3 main folders (Dataset, Notebooks and Results), together with the requirement.txt file and the README.md .

### Dataset

Data needed to carry out this project has been obtained from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) organization . This organization unites researchers which objective is to study and define the progression of Alzheimer’s disease (AD). ADNI organization collects different type data, including MRI and PET images, genetics, cognitive tests, blood biomarkers, etc. The access to this data is not opened to the public in general, and an application must be sent in order to get access to the data (here).

ADNI includes data from 800 subjects, where 200 are normal controls, 400 are individuals with MCI and 200 present signs of Alzheimer. From all the data accessible in ADNI, only MRIs have been used in this thesis. 

### Notebooks

This project contains 5 jupyter notebooks, which can be found in the folder Notebooks.
- 1_Capture_data
- 2_MRI_preprocessing
- 3_TFR_preprocessing
- 4_CNN_creation
- 5_Alzheimer_prediction

In the folder Notebooks, a subfolder called aux_functions contains some python files, which contains functions used in the jupyter notebooks.

## Results & Conclusions

Below it is presented a summary of the results obtained. As it can be seen, the CNN model trained in this project has a recall of 92% and an accuracy of 84%.

![alt text](https://github.com/fernandoherran/master-thesis/blob/4ca06d851737e0d65e047c6430bdca1b0b8725cc/Results/figures/test_cm.png)

Looking at the roc curve, even the model fits very well with the training dataset, it also give good results with the testing dataset.

![alt text](https://github.com/fernandoherran/master-thesis/blob/4ca06d851737e0d65e047c6430bdca1b0b8725cc/Results/figures/roc_curve.png)
