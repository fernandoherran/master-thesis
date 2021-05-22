import os
import gzip
import shutil
import matplotlib.pyplot as plt
import nibabel as nib


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

