from keras import backend as K
import telegram_send
import os
import h5py
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot
import seaborn as sn
import time

##################
##    UTILS     ##
##################

# print the number of available cpus
gpus = K.tensorflow_backend._get_available_gpus()
print('Available gpus', gpus)

def terminate():
    telegram_send.send(['Training has finished'])
    # shutsdown aws instance
    # call(['poweroff'])

# generate output dir + filename
def outdir(model_name):
    # check if dir exists, otherwise create
    basedir = './output/{}'.format(model_name)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    return '{}/{}'.format(basedir, model_name)

# load datasets
def load_dataset(name):
    # import datasets
    with h5py.File('datasets/train{}.h5'.format(name), 'r') as hf:
        X_train = hf['x'][:]
        Y_train = hf['y'][:]
    with h5py.File('datasets/test{}.h5'.format(name), 'r') as hf:
        X_test = hf['x'][:]
        Y_test = hf['y'][:]
    
    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
                   
    return X_train, X_test, Y_train, Y_test
                   
def show_stats(start_time, preds):
    print()
    print('Total time =', round(time.time() - start_time))
    print ("Loss =", str(preds[0]))
    print ("Test Accuracy =", str(preds[1]))
    print ("F-score =", str(preds[2]))

def ohe_to_label(ohe_labels):
    Y = [np.argmax(t) for t in ohe_labels]
    return Y

# plot and save the confusion matrix
def conf_matrix(Y_true, Y_pred, class_conversion, model_name, save = False):
    cm = confusion_matrix(Y_true, Y_pred)
    fig, ax = pyplot.subplots(figsize=(12,12))
    sn.heatmap(np.divide(cm, np.sum(cm, axis=1).reshape(-1,1)), annot=True, ax=ax)
    ax.yaxis.set_ticklabels(class_conversion.values())
    ax.xaxis.set_ticklabels(class_conversion.values())
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    if save:
        fig.savefig('{}.png'.format(outdir(model_name)))
        np.divide(cm, np.sum(cm, axis=1)).round(2)
        cla = classification_report(Y_true, Y_pred, target_names=class_conversion.values())
        with open('{}.txt'.format(outdir(model_name)), 'w') as text_file:
            text_file.write(cla)

def export_model(model, out_name):
    # serialize model to JSON
    model_json = model.to_json()
    with open('{}-model.json'.format(outdir(out_name)), 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('{}-weights.h5'.format(outdir(out_name)))
            
# some custom measures to be computed at each epoch for the models
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
