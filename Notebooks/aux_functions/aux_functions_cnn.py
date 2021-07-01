# Import libraries
########################
import pandas as pd
import numpy as np
from functools import reduce
import os

# Visualization packages
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(context='notebook')
sns.set_style("ticks")

# Tensorflow packages
import tensorflow as tf
from tensorflow.keras import backend as K

# Metrics packages
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score


# Define functions 
########################

def load_tfr_dataset(tfr_dir = "../Datasets/TFRecords/", pattern ="*_volumes.tfrecords",files = None):
    '''
    Function used to load a Tensorflow dataset.
    Inputs: TFRecords directory, TFRecords pattern or list of patterns.
    Output: Tensorflow dataset
    '''
    
    def parse_tfr_element(element):
    
        # Define TFRecord dictionary
        data = {'height': tf.io.FixedLenFeature([], tf.int64),
                'width':tf.io.FixedLenFeature([], tf.int64),
                'depth':tf.io.FixedLenFeature([], tf.int64),
                'raw_image' : tf.io.FixedLenFeature([], tf.string),
                'label':tf.io.FixedLenFeature([], tf.int64)}
        
        # Read TFRecord content
        content = tf.io.parse_single_example(element, data)
      
        raw_image = content['raw_image']
        label = content['label']
      
        feature = tf.io.parse_tensor(raw_image, out_type = tf.float32)
        
        # Reshape feature and label 
        feature = tf.reshape(feature, shape = [110, 130, 80])
        label = tf.reshape(label, shape = [1])
        
        return (feature, label)

    # Get files matching the pattern
    if files == None:
        files = tf.io.gfile.glob(tfr_dir+pattern)
    
    AUTO = tf.data.AUTOTUNE

    # Create the dataset
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads = AUTO)

    # Map function
    dataset = dataset.map(parse_tfr_element, num_parallel_calls = AUTO)
    
    return dataset
    

def get_true_labels(dataset):
    '''
    Function used to return the true labels of a Tensorflow dataset.
    Input: dataset
    Output: list with true labels.
    '''

    return list(reduce(lambda a, b: np.concatenate((a, b), axis=0), [y.numpy() for x, y in dataset]))


def get_predicted_labels(model, dataset):
    '''
    Function used to return the predicted labels of a Tensorflow dataset.
    Inputs: neural network model, dataset
    Output: list with predicted labels.
    '''
    
    predicted_labels = list(reduce(lambda a, b: np.concatenate((a, b), axis=0), model.predict(dataset)))
    predicted_labels = list(map(lambda x: 1 if x > 0.5 else 0, predicted_labels))
    
    return predicted_labels


def plot_history(history, save_fig = False):
    '''
    Function to plot the metrics history of the traing and validation data.
    Inputs: history.
    Output: figure with metrics history
    Figure can be saved specifying path on save_fig
    '''
    
    if type(history) is dict:
        pass
    else:
        history = history.history
    
    limit_ = int(len(list(history.keys()))/2)
    keys_ = list(history.keys())[:limit_]

    # Plot history curves
    figure, axes = plt.subplots(1, len(keys_), figsize = (15,6))

    for index_, key_ in enumerate(keys_):

        train_plot = history[key_]
        val_plot = history["val_" + key_]

        x = range(1, len(train_plot) + 1)

        if key_ != 'loss':
          axes[index_].set_ylim(top=1.05)


        axes[index_].plot(x, train_plot, 'b', label='Train')
        axes[index_].plot(x, val_plot, 'r', label='Validation')
        axes[index_].set_xlabel("Epoch", fontsize=14)
        axes[index_].set_title(key_, fontsize=14, fontweight="bold")
        axes[index_].grid(linestyle='-', linewidth=1, alpha = 0.5)
        axes[index_].set_ylim(bottom=0)
    
    axes[0].legend(fontsize=12, loc="best")
    
    if save_fig != False:
        
        root = '/'.join(save_fig.split('/')[0:-1])
        
        # Check if "results" folder exists
        if not os.path.exists(root ):
            os.mkdir(root )
        
        plt.savefig(save_fig, dpi = 500, transparent = False)
        

def get_evaluation(y, y_predict, save_fig):
    '''
    Function to plot the confusion matrix
    Inputs: true labels, predicted labels
    Output: figure with confusion matrix and the metrics annotated
    Figure can be saved specifying path on save_fig
    '''
    
    # Get Metrics
    accuracy = accuracy_score(y, y_predict)
    precision = precision_score(y, y_predict)
    recall = recall_score(y, y_predict)
    f1 = f1_score(y, y_predict)
    stats_text = "[+] Accuracy = {:0.2f}\n[+] Precision = {:0.2f}\n[+] Recall = {:0.2f}\n[+] F1 Score = {:0.2f}".format(
                accuracy,precision,recall,f1)

    # Calculate confusion matrix
    cm_labels = ["Healty", "Alzheimer"]
    cm = confusion_matrix(y, y_predict)
    cm = pd.DataFrame(cm, index = cm_labels, columns = cm_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(8,5))
    ax = sns.heatmap(cm, annot = True, fmt = "", cmap = "Blues", cbar = False)    
    plt.xlabel('Predicted label', fontsize = 14, fontweight = 'bold')
    plt.ylabel('True label', fontsize = 14, fontweight = 'bold') 
    plt.tight_layout()
    
    plt.text(0.8, 1.15,
             stats_text, 
             style = 'normal', 
             fontsize = 13, 
             bbox = {'facecolor': 'ghostwhite', 'alpha':1 , 'pad': 8})
    
    if save_fig != False:
        
        root = '/'.join(save_fig.split('/')[0:-1])
        
        # Check if "results" folder exists
        if not os.path.exists(root ):
            os.mkdir(root )
        
        plt.savefig(save_fig, dpi = 500, transparent = False)
        
    plt.show()


def plot_roc_curve(model, train_dataset, test_dataset, y_train, y_test, save_fig = False):
    '''
    Function used to plot the training & testing roc curve.
    Inputs: model, train_dataset, test_dataset, y_train, y_test
    Output: figure with roc curve
    Figure can be saved specifying path on save_fig
    '''

    class_1_probs_train = list(reduce(lambda a, b: np.concatenate((a, b), axis=0), model.predict(train_dataset)))
    class_1_probs_test = list(reduce(lambda a, b: np.concatenate((a, b), axis=0), model.predict(test_dataset)))

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
    axes.legend(["Train","Test"],fontsize=12, loc="center right")
    axes.grid(linestyle='-', linewidth=1, alpha = 0.5)
    axes.text(0.6, 0.04,
              'Train AUC score: {:0.2f} \nTest AUC score: {:0.2f}'.format(roc_auc_score(y_train, class_1_probs_train),
                                                                          roc_auc_score(y_test, class_1_probs_test)), 
              style='italic', 
              fontsize=12, 
              bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10});

    if save_fig != False:
        
        root = '/'.join(save_fig.split('/')[0:-1])
        
        # Check if "results" folder exists
        if not os.path.exists(root ):
            os.mkdir(root )
        
        plt.savefig(save_fig, dpi = 500, transparent = False)


def get_metrics(y_true, y_predict):
    '''
    Function used to return the metrics of a dataset.
    Input: true labels, predicted labels
    Output: accuracy, precision, recall, f1 socre
    '''
    
    accuracy = accuracy_score(y_true, y_predict)
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    f1 = f1_score(y_true, y_predict)

    return accuracy, precision, recall, f1
    

def f1(y_true, y_pred):
    '''
    Function used to calculate f1 score during the model training.
    '''
    
    def recall(y_true, y_pred):
        '''
        Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        '''
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        '''
        Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        '''
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
        
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))