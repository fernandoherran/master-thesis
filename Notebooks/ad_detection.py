# Import libraries
import sys
import os
import time
from aux_functions.app_functions import *

# Define process
if __name__ == '__main__':

    start = time.time()

    path = '../'
    sys.path.append(path)

    # Get inputs
    mri_file = sys.argv[1]
    print('[+] MRI file to be processed:', os.path.basename(mri_file))
    
    ### RUN CODE ###

    # Read and preprocess NiFTI file
    volume = process_scan(mri_file)
    print('[+] MRI preprocessed')

    # Load neural network
    model_name = '3d_model_v4'
    model = load_cnn(model_name, path)
    print('[+] CNN model loaded')

    # Reshape volume    
    volume = volume[10:120, 30:160, 15:95]
    volume = np.reshape(volume, (1,) + volume.shape)
    
    # Apply neural network to model
    print('[+] Prediction class...')
    prediction = model.predict(volume)
    
    # Get activation map
    print('[+] Extracting activation maps ...')
    conv_heatmap = get_activation_map(model, volume)
    
    # Show class prediction
    if prediction > 0.5:
        print(f'RESULT: Model has predicted that the MRI contains signs of Alzheimer with a probability of {prediction[0][0] * 100:.2f} %')
    else:
        print(f'RESULT: Model has predicted that the MRI is cognitively normal with a probability of {prediction[0][0] * 100:.2f} %')

    end = time.time()
    print('\nRun time: ', end-start)