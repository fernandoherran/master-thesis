# Import libraries
########################

import matplotlib.pyplot as plt

# Define functions 
########################

def neuro_plot(image, view_ = 'first', slice_ = 180):
    '''
    Function used to display views of a 3D image
    Input: 3D image, view (first, second, third or all), slice (integer or vector if view 'all' is selected
    Output: figure with image displayes
    '''
     
    if view_ == 'first':
        fig, axes = plt.subplots(figsize = (10,8))
        axes.imshow(image[slice_, :, :], cmap = 'gray', origin = 'lower')
    
    if view_ == 'second':   
        fig, axes = plt.subplots(figsize = (10,8))
        axes.imshow(image[:, slice_, :].T, cmap = 'gray', origin = 'lower')
    
    if view_ == 'third':
        fig, axes = plt.subplots(figsize = (10,8))
        axes.imshow(image[:, :, slice_].T, cmap = 'gray', origin = 'lower')
        
    if view_ == 'all':
        fig, axes = plt.subplots(1, 3, figsize=(25,8))
        
        axes[0].imshow(image[slice_[0], :, :], cmap = 'gray', origin = 'lower')
        
        axes[1].imshow(image[:, slice_[1], :].T, cmap = 'gray', origin = 'lower')

        axes[2].imshow(image[:, :, slice_[2]].T, cmap = 'gray', origin = 'lower')
