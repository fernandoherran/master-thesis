path = "../"

import sys
sys.path.append(path)

import os
import time
import numpy as np
import nibabel as nib
import tensorflow as tf
from deepbrain import Extractor
from scipy import ndimage
from tensorflow.keras.models import load_model 
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import BinaryAccuracy
from Notebooks.aux_functions.aux_functions_cnn import *
from skimage.transform import resize


def read_nifti_file(file):
    """
    Read and load nifti file.
    """
    
    # Read file
    volume = nib.load(file)
    
    # Get raw data
    volume = volume.get_fdata()
    
    # Exchange axis 0 and 2
    if volume.shape[1] == volume.shape[2]:
        print(f"{file} has a shape incompatible")
    
    return volume


def remove_skull(volume):
    """
    Extract only brain mass from volume.
    """
    
    # Initialize brain tissue extractor
    ext = Extractor()

    # Calculate probability of being brain tissue
    prob = ext.run(volume) 

    # Extract mask with probability higher than 0.5
    mask = prob > 0.5
    
    # Detect only pixels with brain mass
    volume [mask == False] = 0
    volume = volume.astype("float32")
    
    return volume


def normalize(volume):
    """
    Normalize the volume intensity.
    """
    
    I_min = np.amin(volume)
    I_max = np.amax(volume)
    new_min = 0.0
    new_max = 1.0
    
    volume_nor = (volume - I_min) * (new_max - new_min)/(I_max - I_min)  + new_min
    volume_nor = volume_nor.astype("float32")
    
    return volume_nor


def cut_volume(volume):
    """
    Cut size of 3D volume.
    """
    
    if volume.shape[0] == 256:
        volume_new = volume[20:220,30:,:]
    
    if volume.shape[0] == 192:
        volume_new = volume[20:180,20:180,:]
    
    return volume_new


def resize_volume(volume):
    """
    Resize across z-axis
    """
    
    # Set the desired depth
    desired_height = 180
    desired_width = 180
    desired_depth = 110
    
    # Get current depth
    current_height = volume.shape[0]
    current_width = volume.shape[1]
    current_depth = volume.shape[2]
    
    # Compute depth factor
    height = current_height / desired_height
    width = current_width / desired_width
    depth = current_depth / desired_depth

    height_factor = 1 / height
    width_factor = 1 / width
    depth_factor = 1 / depth
    
    # Rotate
    #img = ndimage.rotate(img, 90, reshape=False)
    
    # Resize across z-axis
    volume = ndimage.zoom(volume, (height_factor, width_factor, depth_factor), order=1)
    
    return volume
    

def process_scan(file):
    """
    Read, skull stripping and resize Nifti file.
    """
    
    # Read Nifti file
    volume = read_nifti_file(file)
    
    # Extract skull from 3D volume
    volume = remove_skull(volume)
    
    # Cut 3D volume
    #volume = cut_volume(volume)
    
    # Resize width, height and depth
    volume = resize_volume(volume)
    
    # Normalize pixel intensity
    volume = normalize(volume)
    
    return volume


def load_cnn(model_name):

    #  Load model
    model = load_model(path + "Results/" + model_name + ".h5", 
                       custom_objects = {'f1': f1})

    # Define optimizer
    optimizer = Adam(learning_rate = 0.001, decay = 1e-6)

    # Compile model
    model.compile(loss = "binary_crossentropy",
                  optimizer = optimizer,
                  metrics = [BinaryAccuracy(), f1])

    return model


def get_activation_maps(model, volume):

    # Layer to visualize
    layer_name = 'conv3d_31'
    conv_layer = model.get_layer(layer_name)

    # Create a graph that outputs target convolution and output
    grad_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])

    # Compute GRADIENT
    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(volume)
        loss = predictions[:, 0]

    # Extract filters and gradients
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    # Average gradients spatially
    weights = tf.reduce_mean(grads, axis = (0, 1, 2))

    # Build a ponderated map of filters according to gradients importance
    conv_heatmap = np.zeros(output.shape[0:3], dtype = np.float32)

    for index, weight in enumerate(weights):
        conv_heatmap += weight * output[:, :, :, index]

    # Reshape ponderated map of filters
    conv_heatmap = resize(conv_heatmap, volume.shape[1:])
    conv_heatmap = np.maximum(conv_heatmap, 0)
    conv_heatmap = (conv_heatmap - conv_heatmap.min()) / (conv_heatmap.max() - conv_heatmap.min())
    
    return conv_heatmap



if __name__ == "__main__":

    start = time.time()

    path = "../"
    
    sys.path.append(path)

    # Get inputs
    mri_file = sys.argv[1]
    print("[+] MRI file to be processed:", os.path.basename(mri_file))
    
    ## RUN CODE 

    # Get volume cleaned
    volume = process_scan(mri_file)
    print("[+] MRI cleaned")

    # Load neural network
    model_name = "3d_model_v4"
    model = load_cnn(model_name)
    print("[+] CNN model loaded")

    # Reshape volume    
    volume = volume[10:120, 30:160, 15:95]
    volume = np.reshape(volume, (1,) + volume.shape)
    
    # Apply neural network to model
    prediction = model.predict(volume)
    if prediction > 0.5:
        print("RESULT: Model has predicted that the MRI contains signs of Alzheimer")
    else:
        print("RESULT: Model has predicted that the MRI is cognitively normal")

    # Get activation map
    conv_heatmap = get_activation_maps(model, volume)

    end = time.time()
    print("\nRun time: ", end-start)

