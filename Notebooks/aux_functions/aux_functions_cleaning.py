# Import libraries
########################
import os
import gzip
import shutil
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from deepbrain import Extractor

# Define functions 
########################

def load_images(folder):
    """ 
    Function used to load Nifti files from a folder.
    Input: folder path
    Output: list with images, list with titles, list with images shapes
    """
    
    images = [] # List where to save the nitfi 3D images
    titles = [] # List where to save the name of the 3D images 
    shapes = [] # List where to save the shapes of the 3D images
    
    count_images = 0 # Initialize images counter

    for file in os.listdir(folder):

        image = "no_image"

        # Avoid trigerring .DS_Store (when use macOS)
        if file.startswith('.DS_Store'):
            continue

        # Load Nifti file
        if file.endswith('.nii.gz'):
            
            image = nib.load(os.path.join(folder, file))

            # Add 3D image to images list
            images.append(image)
            
            # Add name of the file to titles list
            titles.append(file)

        # Check shape of the image
        if image.get_fdata().shape not in shapes:
            shapes.append(image.get_fdata().shape)
            shapes.append(1)
        else:
            index = shapes.index(image.get_fdata().shape)
            shapes[index + 1] +=1
        
        # Update images counter
        count_images += 1
        
        # Print counter status
        if(count_images % 20 == 0):
            print(f"        {count_images} images loaded")
    
    return images, titles, shapes


def extract_slice(image, title = "None"):
    """
    Function to plot a collection of slices from the 3 points of views: axial, coronal and sagittal.
    Inputs: 
        - image = image to plot
        - num_slices = number of slices for each plane
    """
    
    ############################
    ### INITIAL CALCULATIONS ###
    ############################
    
    # Figure parameters (rows and columns)
    num_rows = 2
    num_columns = 3
    num_slices = num_rows * num_columns
    cut_value_slice = 0.4 # 30% of the axis shape
    
    # Get image data
    data = image.get_fdata()
    rotated = False
    
    if data.shape[0] == data.shape[1]:
        
        # Retrieve shape of the image shape
        shape = data.shape 
    
    elif data.shape[1] == data.shape[2]:
        
        # Exchange axis 0 and 2
        data = np.swapaxes(data, 0, 2) 
        
        # Retrieve shape of the image shape
        shape = data.shape 
        
        rotated = True
    
    # Set cuts for each axis of the slice
    cut_value_ver = [ int(shape[0] * 0.06), int(shape[0] * (1 - 0.3)) ]
    cut_value_horiz = [ int(shape[2] * 0.1), int(shape[2] * (1 - 0.1)) ]

    # Get random slices of coronal plane
    x_1, x_2  = int(shape[1] * cut_value_slice), int(shape[1] * (1 - cut_value_slice))
    step_ = int((x_2 - x_1) / 7)
    slices_ = list(range(x_1, x_2, step_))[1:]

    ######################
    ### GET BRAIN MASS ###
    ######################
    
    # Initialize brain tissue extractor
    ext = Extractor()
    
    # Calculate probability of being brain tissue
    prob = ext.run(data) 

    # Extract mask with probaility higher than 0.5
    mask = prob > 0.5
    
    ##############
    ### FIGURE ###
    ##############
    
    # Get heights and widths
    heights = [i * (cut_value_ver[1] - cut_value_ver[0]) for i in [1] * num_rows]
    widths = num_columns * [cut_value_horiz[1] - cut_value_horiz[0]]

    # Set-up figure width and height
    fig_width = 7.0
    fig_height = fig_width * sum(heights) / sum(widths)

    # Set figure
    fig, axes = plt.subplots(num_rows, num_columns, 
                            figsize = (fig_width, fig_height),
                            gridspec_kw = {"height_ratios": heights})
    
    # Add 2D slices of coronal plane to each figure subplot
    count_slice = 0
    
    for i in range(num_rows):
        for j in range(num_columns):
            
            slice_index = slices_[count_slice]
            
            # Get 2D slice
            slice_original = np.array(data[:, slice_index, :])

            # Extract slice after skull-stripping 
            slice_cleaned = return_clean_slice(slice_original, slice_index, mask, 2)

            # Cut margins slice
            
            if rotated == True:
                slice_plot = ndimage.rotate(slice_cleaned.T, 90)  ##### AQUIIIIIIIII
                slice_plot = slice_plot[cut_value_ver[0]:cut_value_ver[1], cut_value_horiz[0]:cut_value_horiz[1]]
                
            else:
                slice_plot = slice_cleaned[cut_value_ver[0]:cut_value_ver[1], cut_value_horiz[0]:cut_value_horiz[1]]
                
            # Add slice to figure
            axes[i,j].imshow(slice_plot, cmap="gray")
            axes[i,j].axis("off")
            
            count_slice += 1
    
    # Adjust space between figure subplots
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1);
    
    # Save figure
    plt.savefig(title)
    
    # Close figure
    plt.close(fig)


def return_clean_slice(slice_original, slice_index, mask, border_voxels = 3):
    
    slice_cleaned = np.zeros((slice_original.shape[0], slice_original.shape[1]))

    for index_i, i in enumerate(slice_original):
        for index_k, k in enumerate(i):

            if (index_k > (border_voxels - 1)) & \
               (index_k < (len(i) - border_voxels)) & \
               (index_i > (border_voxels - 1)) & \
               (index_i < (slice_original.shape[0] - border_voxels)):

                if mask[index_i,  slice_index, index_k] == True:
                    slice_cleaned[index_i, index_k] = slice_original[index_i, index_k]

                elif (mask[index_i,  slice_index, index_k] == False) & (mask[index_i,  slice_index, index_k + border_voxels] == True):
                    slice_cleaned[index_i, index_k] = slice_original[index_i, index_k]

                elif (mask[index_i,  slice_index, index_k] == False) & (mask[index_i,  slice_index, index_k - border_voxels] == True):           
                    slice_cleaned[index_i, index_k] = slice_original[index_i, index_k]

                elif (mask[index_i,  slice_index, index_k] == False) & (mask[index_i + border_voxels, slice_index, index_k] == True):           
                    slice_cleaned[index_i, index_k] = slice_original[index_i, index_k]

                elif (mask[index_i,  slice_index, index_k] == False) & (mask[index_i - border_voxels, slice_index, index_k] == True):  
                    slice_cleaned[index_i, index_k] = slice_original[index_i, index_k]
        
    return slice_cleaned


def multiple_slices_OLD(image, title):
    """
    Function to plot a collection of slices from the 3 points of views: axial, coronal and sagittal.
    Inputs: 
        - image = image to plot
        - num_slices = number of slices for each plane
    """
    
    # Figure parameters (rows and columns)
    num_rows = 3
    num_columns = 5
    
    # Retrieve image shape
    shape = image.get_fdata().shape
    
    # Set-up heights and widths
    heights = [shape[1], shape[0], shape[0]]
    widths = num_columns * [shape[2]]
    
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)

    # Get image data
    data = image.get_fdata()
    
    # Get random slices
    random_axial_slices = [randint(round(data.shape[0]*0.3), data.shape[0] - round(data.shape[0]*0.3)) for p in range(0, num_columns)]
    random_coronal_slices = [randint(round(data.shape[1]*0.3), data.shape[1] - round(data.shape[1]*0.3)) for p in range(0, num_columns)]
    random_sagittal_slices = [randint(round(data.shape[2]*0.3), data.shape[2] - round(data.shape[2]*0.3)) for p in range(0, num_columns)]
    
    axial_slices = []
    coronal_slices = []
    sagittal_slices = []
    
    for slice_ in random_axial_slices:
        axial_slices.append(data[slice_,:,:])
    
    for slice_ in random_coronal_slices:
        coronal_slices.append(data[:,slice_,:])
    
    width_third_low = round((shape[1] - shape[2]) / 2)
    width_third_high = shape[1] - width_third_low
    for slice_ in random_sagittal_slices:
        sagittal_slices.append(data[:,(width_third_low):(width_third_high),slice_])
    
    # Set figure
    fig, axes = plt.subplots(num_rows, num_columns, 
                            figsize=(fig_width, fig_height),
                            gridspec_kw={"height_ratios": heights})
    
    for i in range(num_rows):
        for j in range(num_columns):
            if i == 0:
                slice_plot = axial_slices[j]
            elif i == 1:
                slice_plot = coronal_slices[j]
            else:
                slice_plot = sagittal_slices[j]
                
            axes[i,j].imshow(slice_plot, cmap="gray", origin="lower")
            axes[i,j].axis("off")
    
    # Adjust space between subplots
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1);
    #fig.tight_layout()
    
    # Save figure
    plt.savefig(title)
    plt.close(fig)
    
    # Plot figure
    #plt.show()
