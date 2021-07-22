# Alzheimer’s disease detection using a Convolutional Neural Network
Project by Fernando Herrán Albelda (Master in Data Science for KSchool). 

An [interactive application](https://alzheimer-prediction.azurewebsites.net) using Streamlit has been built for this thesis and can be used by the user to predict if a patient has signs of Alzheimer or not. The only input needed to run the application is to upload a Magnetic Resonance Image (MRI) of his brain, which must be in NIfTI format (.nii or .nii.gz). The Github repository of this application can be found [here](https://github.com/fernandoherran/thesis-streamlit-app).

## Introduction

### Objective
The goal of this thesis is to build a 3D - Convolutional Neural Network to classify patients with Alzheimer's disease (AD) or cognitively normal (CN) using MRIs of their brains. The problem consists of a binary classification and is justified by the need to learn about the use of Deep Learning to solve problems where images are related, and to understand the impact of the network structure on the performance.

### Requirements
Next libraries are used in the code:
- [nibabel](https://nipy.org/nibabel/)
- [deepbrain](https://pypi.org/project/deepbrain/)
- [cv2](https://docs.opencv.org/master/d0/de3/tutorial_py_intro.html)
- [tensorflow](https://www.tensorflow.org/tutorials)
- Visualization: matplotlib, seaborn
- Others: os, sys, time, numpy, pandas, random, scipy, shutil, sklearn,tqdm, functools, skimage, gzip

### Set-up virtual environment
In order to run the code without having dependencies problems, the user can create a virtual environment with all the needed packages. This virtual environment has a total size of 1.31 GB. To create the virtual environment, please follow process below: 

- Clone Github repository in your computer. Open terminal, and run comand below:
```
git clone https://github.com/fernandoherran/master-thesis.git
```
- Once cloned, run below command to create the virtual environment in your computer:
```
python -m venv virtual_env
```
- Once created, the user can activate the virtual environment running the following command (first, the user must deactivate the current environment):
```
 source virtual_env/bin/activate
```
- Finally, the user can install all the needed dependencies in the virtual environment running the following command (file requirements.txt has been downloaded when cloning the repository):
```
pip install -r requirements.txt
```

## Repository

In order to run the code without having issues of directories or files missing, the user should create a folder in his computer or Google Drive called 'master-thesis' (or other name), and place inside the following three folders: Datasets, Notebooks and Results.

In this Github repository, the user can get the folders Notebooks and Results, whilst the folder Datasets can be found in this Google Drive [folder](https://drive.google.com/drive/folders/1oNPSc0m6J8Acot32bvU4BDOPVDlgfeut?usp=sharing). 

So, the directory structure should be as follows:

```
master-thesis
├── Datasets
├── Notebooks
└── Results
```

### Dataset

All data needed to carry out this project has been obtained from the [Alzheimer’s Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu/about/) organization. This organization unites researchers which objective is to study and define the progression of Alzheimer’s disease (AD). ADNI organization collects different types of data, including MRI and PET images, genetics, cognitive tests, blood biomarkers, etc. Access to this data is not opened to the public in general, and an application must be sent in order to get access to the data (more information [here](https://ida.loni.usc.edu/collaboration/access/appLicense.jsp)).

ADNI dataset includes data from 800 subjects, where 200 are normal controls, 400 are individuals with MCI and 200 present signs of Alzheimer. In this thesis, it has worked with MRIs from patients under normal controls (CN) and Alzheimer's disease (AD).

As explained in ***Notebook 1_Capture_data***, once the raw data is downloaded from ADNI, it is presented in multiple folders and with a total size of 121 GB. The Notebook 1_Capture_data is used to reorganize all these folders, zip the MRIs files and stored them in a unique folder called *Extracted_files*, which has a total size of 72GB.

Below it can be seen the files presented in the folder Datasets:

```
master-thesis
└── Datasets
    ├── Extracted_files: folder which contains all the MRIs files. The folder size is 72 GB.
    ├── Image_files: folder which contains the images needed to train the CNN in a numpy’s compressed format (.npz). The folder size is 1.32 GB.
    ├── TFRecords: folder which contains the images needed to train the CNN in TFRecords format. The folder size is 5.24 GB.
    ├── ADNI1_Complete_1Yr_1.5T.csv
    ├── ADNI1_Complete_2Yr_1.5T.csv
    ├── ADNI1_Complete_3Yr_1.5T.csv
    └── list_titles.npz: list which contains the Image IDs of the files that must be used to train the CNN.
```

TFRecords can also be found in this Google Cloud Storage [bucket](https://console.cloud.google.com/storage/browser/tfm-kschool-bucket?project=tfm-kschool&pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))\&prefix=\&forceOnObjectsSortingFiltering=false), as to work with the Notebook ***4_CNN_creation*** via Google Colab, it is needed to load the TFRecords from a Google bucket.

### Notebooks

This project contains five main jupyter Notebooks, which are used to carry out all the process (from capturing the raw data, preprocess it, convert them to TFRecords and create the CNN). Inside the folder Notebooks, there is a folder called *aux_functions*, which contains some python files with functions used in the Notebooks.

- 1_Capture_data
- 2_MRI_preprocessing
- 3_TFR_creation
- 4_CNN_creation
- 5_Alzheimer_prediction

Below it can be seen the Notebooks directory structure:

```
master-thesis
└── Notebooks
    ├── 1_Capture_data.ipynb
    ├── 2_MRI_preprocessing.ipynb
    ├── 3_TFR_creation.ipynb
    ├── 4_CNN_creation.ipynb
    ├── 5_Alzheimer_prediction.ipynb
    └── aux_functions
         ├── aux_functions_visualization.py
         ├── aux_functions_cnn.py
         ├── deepbrain_package
         ├── app_functions
         └── ad_detection.py
```

## Results & Conclusions


Below it is presented a summary of the results obtained. As it can be seen, the CNN model trained in this project has a recall of 92% and an accuracy of 84%.

![alt text](https://github.com/fernandoherran/master-thesis/blob/4ca06d851737e0d65e047c6430bdca1b0b8725cc/Results/figures/test_cm.png)

Looking at the roc curve, even the model fits very well with the training dataset, it also give good results with the testing dataset.

![alt text](https://github.com/fernandoherran/master-thesis/blob/4ca06d851737e0d65e047c6430bdca1b0b8725cc/Results/figures/roc_curve.png)

These figures, together with the CNN model in .h5 format can be found in the folder Results. This folder contains the following files:

```
master-thesis
└── Results
    ├── cnn_model.h5
    ├── cnn_model_history.npy
    ├── figures
    └── optuna_studies
```
