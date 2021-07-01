# Alzheimer’s disease detection using Convolutional Neural Networks
Project done by Fernando Herrán Albelda (Master in Data Science for KSchool). 

An [interactive application](https://alzheimer-detection.herokuapp.com/) using Streamlit has been build for this thesis and can be used by the user to predict if a patient has signs of alzheimer uploading a MRI of his brain. Github repository of this application can be found [here](https://github.com/fernandoherran/thesis-streamlit-app).

## Introduction

### Objective
The goal of this thesis is to build a Convolutional Neural Network to classify patients with Alzheimer's disease (AD) or cognitively normal (CN) using their magnetic resonance images (MRI).

### Requirements
Next libraries are used in the code:
- [nibabel](https://nipy.org/nibabel/)
- [deepbrain](https://pypi.org/project/deepbrain/)
- [cv2](https://docs.opencv.org/master/d0/de3/tutorial_py_intro.html)
- [tensorflow](https://www.tensorflow.org/tutorials)
- Visualization: matplotlib, seaborn
- Others: os, sys, time, numpy, pandas, random, scipy, shutil, sklearn,tqdm, functools, skimage, gzip

### Set-up virtual environment
In order to run the code without having dependencies problems, user can create a virtual conda environment with all needed packages. To do that, please follow process below: 
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

### Notebooks

## Results & Conclusions

![alt text](https://github.com/fernandoherran/master-thesis/blob/89ef925d6e779f7a7894781591c73cab8cfb228a/Results/figures/test.png)
