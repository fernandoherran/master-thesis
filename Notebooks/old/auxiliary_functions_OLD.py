import os
import gzip
import shutil
import matplotlib.pyplot as plt
import nibabel as nib

def extract_images(root, new_root):
    '''
    Function to extract nifti images from a folder (root), zip them and save them in new fodler (new_root)
    ''' 
    
    # Create new directory to move files in case it doesn´t exist
    if not os.path.exists(new_root):
        os.makedirs(new_root)

    # Get directions of the file and move them to new folder
    for folder in sorted(os.listdir(root)):

        # Avoid trigerring .DS_Store
        if folder.startswith('.'):
            continue

        directions = []
        directions.append(os.path.join(root,folder))
        all_files = False

        while all_files == False:
            for index, path in enumerate(directions):

                if (os.path.isfile(path)) == False:

                    for subfolder in sorted(os.listdir(path)):

                        if subfolder.startswith('.'):
                            continue

                        directions.append(os.path.join(path,subfolder))

                    directions.remove(path)

                if index == (len(directions) -1):
                    is_not_file = False

                    for item in directions:
                        if (os.path.isfile(item)) == True:
                            continue
                        else:
                            is_not_file = True

                    if is_not_file == False:
                        all_files = True
                    else:
                        break
    
        # Copy files, compress them and move them to new folder
        for direction in directions:
            new_direction = os.path.join(new_root,direction.split("/")[-1]) + ".gz"

            # Check if file already exist in the new folder
            if os.path.exists(new_direction) == False:
                
                # Copy image file to new folder
                with open(direction, 'rb') as f_in:
                    with gzip.open(new_direction, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                print(f"File already exists in the new folder: {new_direction}")
              
        print(folder)
        print("*" * 30)
    
    
def load_images(root):
    """ 
    Function to load nifti images from a folder
    """
    
    images = []
    titles = []
    shapes = []
    n_images = 0

    for file in os.listdir(root):

        image = "no_image"

        # Avoid trigerring .DS_Store
        if file.startswith('.DS_Store'):
            continue

        # Load nifti image
        if file.endswith('.nii.gz'):
            image = nib.load(os.path.join(root,file))

            # Add image and file name to list
            images.append(image)
            titles.append(file)

        # Check shape of the image
        if image.get_fdata().shape not in shapes:
            shapes.append(image.get_fdata().shape)
            shapes.append(1)
        else:
            index = shapes.index(image.get_fdata().shape)
            shapes[index + 1] +=1

        n_images += 1
        if(n_images % 20 == 0):
            print(f"{n_images} images loaded")
    
    return images, titles, shapes


def axial_plot(nii_img, length = 180):
    """ 
    Function to display axial view of a NII image
    """
    
    fig, axes = plt.subplots(figsize=(10,8))
    axes.imshow(nii_img.get_fdata()[:, length, :].T, cmap="gray", origin="lower")

    
def neuro_plot(nii_img, view_ = "axial", slice_ = 180):
    """ 
    Function to display views of a NII image
        Inputs: nii image, view, slice_
        view = [sagittal, axial, coronal, all]
        slice_ = integer to visualize only one plane; or list to visualize all planes
    """
    
    
    if view_ == "sagittal":
        fig, axes = plt.subplots(figsize=(10,8))
        axes.imshow(nii_img.get_fdata()[slice_, :, :], cmap="gray", origin="lower")
        axes.set_title("Sagittal view", fontsize = 14)
    
    if view_ == "axial":   
        fig, axes = plt.subplots(figsize=(10,8))
        axes.imshow(nii_img.get_fdata()[:, slice_, :].T, cmap="gray", origin="lower")
        axes.set_title("Axial view", fontsize = 14)
    
    if view_ == "coronal":
        fig, axes = plt.subplots(figsize=(10,8))
        axes.imshow(nii_img.get_fdata()[:, :, slice_].T, cmap="gray", origin="lower")
        axes.set_title("Coronal view", fontsize = 14)
        
    
    if view_ == "all":
        fig, axes = plt.subplots(1, 3, figsize=(25,8))
        
        axes[0].imshow(nii_img.get_fdata()[slice_[0], :, :], cmap="gray", origin="lower")
        axes[0].set_title("Sagittal view", fontsize = 14)
        
        axes[1].imshow(nii_img.get_fdata()[:, slice_[1], :].T, cmap="gray", origin="lower")
        axes[1].set_title("Axial view", fontsize = 14)
        
        axes[2].imshow(nii_img.get_fdata()[:, :, slice_[2]].T, cmap="gray", origin="lower")
        axes[2].set_title("Coronal view", fontsize = 14)


def multiple_slices(image, title):
    
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


def plot_history(history):
    '''
    Function to plot the accuracy & lost history of the traing and validation data.
    Inputs: history
    '''
    
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    x = range(1, len(acc) + 1)

    # Plot history curves
    figure, axes = plt.subplots(1,2,figsize = (12,6))
    
    axes[0].plot(x, acc, 'b', label='Training acc')
    axes[0].plot(x, val_acc, 'r', label='Validation acc')
    axes[0].set_title('Training and validation accuracy', fontsize=18, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=14)
    axes[0].set_ylabel("Accuracy", fontsize=14)
    axes[0].legend(fontsize=12, loc="best")
    axes[0].grid(linestyle='-', linewidth=1, alpha = 0.5)
    axes[0].set_ylim(bottom=0)
    
    axes[1].plot(x, loss, 'b', label='Training loss')
    axes[1].plot(x, val_loss, 'r', label='Validation loss')
    axes[1].set_title('Training and validation loss', fontsize=18, fontweight="bold")
    axes[1].set_xlabel("Epoch",fontsize=14)
    axes[1].set_ylabel("Loss", fontsize=14)
    axes[1].legend(fontsize=12, loc="best")
    axes[1].grid(linestyle='-', linewidth=1, alpha = 0.5)
    axes[1].set_ylim(bottom=0)


def plot_metric(history, metric):
    history_dict = history.history
    values = history_dict[metric]
    if 'val_' + metric in history_dict.keys():  
        val_values = history_dict['val_' + metric]

    epochs = range(1, len(values) + 1)

    if 'val_' + metric in history_dict.keys():  
        plt.plot(epochs, val_values, label='Validation')
    plt.semilogy(epochs, values, label='Training')

    if 'val_' + metric in history_dict.keys():  
        plt.title('Training and validation %s' % metric)
    else:
        plt.title('Training %s' % metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid()


def plot_roc_curve(model, X_train, X_test, y_train, y_test, save_fig = False):
    '''
    Function used to plot the training & testing roc curve.
    Inputs: model, X_train, X_test, y_train, y_test
    Figure can be saved specifying save_fig = True in the inputs.
    '''

    class_1_probs_train = model.predict(X_train)
    class_1_probs_test = model.predict(X_test)

    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, class_1_probs_train)
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, class_1_probs_test)

    print("[+] Train AUC score = {:0.3f}".format(roc_auc_score(y_train, class_1_probs_train)))
    print("[+] Test AUC score = {:0.3f}".format(roc_auc_score(y_test, class_1_probs_test)))

    # Plot roc curve
    figure, axes = plt.subplots(figsize = (6,6))

    axes.plot(fpr_train,tpr_train, "-", linewidth=1.5)
    axes.plot(fpr_test,tpr_test, "-", linewidth=1.5)

    axes.set_title("Roc curve",fontsize=18,fontweight="bold");
    axes.set_xlabel("False positive rate", fontsize=14)
    axes.set_ylabel("True positive rate", fontsize=14)
    axes.legend(["Train","Test"],fontsize=12, loc="best")
    axes.grid(linestyle='-', linewidth=1, alpha = 0.5)
    axes.text(0.6, 0.04,
              'Train AUC score: {:0.2f} \nTest AUC score: {:0.2f}'.format(roc_auc_score(y_train, class_1_probs_train),
                                                                          roc_auc_score(y_test, class_1_probs_test)), 
              style='italic', 
              fontsize=12, 
              bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10});

    if save_fig == True:
        
        root_results = path + "/results"
        
        # Check if "results" folder exists
        if not os.path.exists(root_results):
            os.mkdir(root_results)
        
        plt.savefig(root_results + '/roc_curve.png', dpi = 500, transparent = False)


def build_model_old(input_shape):
    '''
    Function to build a convolutional neural network.
    Inputs: input shape
    '''

    # Fix random seed for reproducibility
    np.random.seed(123)
    tf.random.set_seed(123) 

    # Start model
    model = Models.Sequential()
    
    # Input layer
    model.add(Layers.Conv2D(50, kernel_size=(3,3),activation='relu',input_shape = input_shape))
    
    # Hidden layers 
    model.add(Layers.Conv2D(18,kernel_size=(3,3),activation='relu'))
    model.add(Layers.MaxPool2D(5,5))
    model.add(Layers.Conv2D(18,kernel_size=(3,3),activation='relu'))
    model.add(Layers.Conv2D(14,kernel_size=(3,3),activation='relu'))
    model.add(Layers.Conv2D(10,kernel_size=(3,3),activation='relu'))
    model.add(Layers.Conv2D(5,kernel_size=(3,3),activation='relu'))
    model.add(Layers.MaxPool2D(5,5))
    model.add(Layers.Flatten())
    model.add(Layers.Dense(18,activation='relu'))
    model.add(Layers.Dense(10,activation='relu'))
    model.add(Layers.Dense(5,activation='relu'))
    model.add(Layers.Dropout(rate=0.5))
    
    # Output layer
    model.add(Layers.Dense(1, activation = 'sigmoid'))
        
    # Compile model
    model.compile(loss = 'binary_crossentropy', 
                  optimizer = 'adam', 
                  metrics = ['binary_accuracy']) 
    
    return model