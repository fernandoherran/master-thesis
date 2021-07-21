# Import libraries
########################

import sys
import os
import time
from app_functions import *

# Define main process
########################

if __name__ == '__main__':

    start = time.time()

    path = '../../'
    sys.path.append(path)

    # Get inputs
    mri_file = sys.argv[1]
    print('[+] MRI file to be processed:', os.path.basename(mri_file))
    
    ### RUN CODE ###

    # Read and preprocess NiFTI file
    volume = process_scan(mri_file)
    print('[+] MRI preprocessed')

    # Load CNN model
    model_name = 'cnn_model'
    model = load_cnn(model_name, path)
    print('[+] CNN model loaded')

    # Reshape volume    
    volume = np.reshape(volume, (1,) + volume.shape)
    
    # Apply CNN to the MRI file
    print('[+] Predicting class...')
    prediction = model.predict(volume)
    
    # Extract activation map from the CNN
    print('[+] Extracting activation maps...')
    conv_heatmap = get_activation_map(model, volume, layer_name = 'conv3d_23')
    
    # Show predicted class
    if prediction > 0.5:
        print(f'\nRESULT: Model has predicted that the MRI contains signs of ALZHEIMER\'S DISEASE with a probability of {prediction[0][0] * 100:.2f} %')
    else:
        print(f'\nRESULT: Model has predicted that the MRI is COGNITIVELY NORMAL with a probability of {prediction[0][0] * 100:.2f} %')

    end = time.time()
    print('\nRun time: ', end-start)