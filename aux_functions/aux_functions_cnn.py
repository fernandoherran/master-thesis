# Visualization packages
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(context='notebook')
sns.set_style("ticks")

# Tensorflow packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense, Dropout
from keras import backend as K

# Metrics packages
from sklearn.metrics import roc_curve, roc_auc_score


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


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


def plot_history_2(history):
    '''
    Function to plot the accuracy & lost history of the traing and validation data.
    Inputs: history
    '''
    
    acc = history.history['f1']
    val_acc = history.history['val_f1']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    x = range(1, len(acc) + 1)

    # Plot history curves
    figure, axes = plt.subplots(1, 2, figsize = (12,6))
    
    axes[0].plot(x, acc, 'b', label='Training')
    axes[0].plot(x, val_acc, 'r', label='Validation')
    axes[0].set_title('Training and validation F1 score', fontsize=18, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=14)
    axes[0].set_ylabel("Accuracy", fontsize=14)
    axes[0].legend(fontsize=12, loc="best")
    axes[0].grid(linestyle='-', linewidth=1, alpha = 0.5)
    axes[0].set_ylim(bottom=0)
    
    axes[1].plot(x, loss, 'b', label='Training')
    axes[1].plot(x, val_loss, 'r', label='Validation')
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


def build_model_OLD(input_shape):
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