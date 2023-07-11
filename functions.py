#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:48:36 2022

@author: adriana
"""

# BEFORE START -----------------------------------------------------------------
# General defined functions
# It is necessary to locate this file in the exec script path to import these functions
# ______________________________________________________________________________

# INSTALL ----------------------------------------------------------------------
# pip install tensorflow
# pip install tensorflow-gpu
# pip install tensorflow-addons # to use Cohen Kappa
# pip install sklearn
# pip install scipy
# pip install pydot - plot model
# https://graphviz.gitlab.io/download/ install
# ______________________________________________________________________________

# LIBRARIES --------------------------------------------------------------------
import os
import csv
import copy
import time
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import numpy as np
from numpy.linalg import norm
import sklearn
from sklearn.utils import class_weight
import scipy.io as sio
from scipy.stats import mode
from itertools import product
from warnings import simplefilter
import logging
import absl.logging
# ______________________________________________________________________________

# CONFIGURATION ----------------------------------------------------------------
simplefilter(action='ignore', category=FutureWarning) # ignore all future warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR) # suppress warnings 
absl.logging.set_verbosity(absl.logging.ERROR) # suppress absl warning (found untraced functions such as lstm_cell_layer) <-- can be safetly ignored

print('Hash Keras: ', hash("keras")) # check keras hash seed
# ______________________________________________________________________________
# ______________________________________________________________________________


# ··············································································
# DATASET FUNCTIONS
# ··············································································
def filesDirectory (path):
    '''Iterates through directories to build two numpy.arrays with every file directory and its label.'''
    
    # define an empty list to append data
    data = []
    labels = []
    
    # iterate through directories
    with os.scandir(path) as iterator:
        for entry in iterator:
            if (os.path.isdir(entry) == True):
                neWpath = os.path.join(path, entry.name)
                os.chdir(neWpath)
                files = os.listdir()        
                for f in files:
                    full_path = os.path.join(neWpath, f)
                    data = np.append(data, full_path)
                    labels = np.append(labels, entry.name)
    
    return data, labels

def datasets (data, labels, seed):
    '''Gives train and test datasets shuffled with same random_state=seed,
    proportion 80-20 taken into account classes distribution with stratify parameter.''' 
  
    train_data, test_data, train_labels, test_labels = sklearn.model_selection.train_test_split(
        data, labels, test_size=0.2, shuffle=True, random_state=seed, stratify=labels) 
   
    return train_data, test_data, train_labels, test_labels

def load_matfile (filename, split):
    ''' load matfiles from tensor'''

    if split == 1:
        filename_ = filename.numpy()
        mat = np.array(sio.loadmat(filename_)['tempData'])
        return mat
    
    else:
        filename_ = filename.numpy()
        mat = np.array(np.split(sio.loadmat(filename_)['tempData'], split, axis=1)) # load mat file
        return mat

class My_Custom_Generator(keras.utils.Sequence):
    '''Creates a generator to load data into batches.'''
        
    def __init__(self, data, labels, batch_size, seed, split, shuffle=False):
        # define variables
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.indices = np.arange(self.data.shape[0]) # create an index to shuffle samples
        self.shuffle = shuffle # default = False
        self.split = split # default = '1' - without split data into submatrices
        self.seed = seed
        
    def __len__(self):
        
        return (np.ceil(len(self.data) / float(self.batch_size))).astype(np.int32) 
           
    def __getitem__(self, idx) :
        
        # __getitem__(idx)[0] gives X = data
        # __getitem__(idx)[1] gives Y = labels
        # batch_x = self.data[idx * self.batch_size : (idx+1) * self.batch_size]
        # batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.data[inds]
        batch_y = self.labels[inds]
        
        if self.split == 1:
            return np.array([sio.loadmat(file)['tempData'] for file in batch_x]), np.array(batch_y) # returns X and Y structured in an object
        else:
            return np.array([np.split(sio.loadmat(file)['tempData'], self.split, axis=1) for file in batch_x]), np.array(batch_y) # returns X and Y structured in an object, each mat file is splitted into 3 subfiles

    def on_epoch_end(self):       
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(self.indices) 
# ______________________________________________________________________________

# TF.DATASETS ------------------------------------------------------------------       
def get_data (filename, split):
    ''' load mat file from each item in dataset'''
    # must be convertible to a tf function, thus it is mandatory to apply tf.py_function() to our load_mat function
    return tf.py_function(load_matfile, inp=[filename, split], Tout=tf.float32)    

def process_data (filename, labels):
    
    data = get_data(filename, split=5)   
    # data = get_data(filename, split=1) 
    return data, labels
  
def _fixup_shape (filename, labels):
    ''' set tensor shapes '''
    filename.set_shape([5,4,3000])
    # filename.set_shape([4,3000])  
    labels.set_shape([5])
    return filename, labels 
    
def ds_generator (data, labels, buffer_size, batch_size, seed, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(_fixup_shape, num_parallel_calls=tf.data.AUTOTUNE)
    # dataset = dataset.cache()
    
    if shuffle == True:
        # dataset = dataset.repeat(30)
        dataset = dataset.shuffle(buffer_size, seed, reshuffle_each_iteration=True) # important to make the suffle size large enough or else shuffling will not be effective.
        # the shuffle order should be different for each epoch. The dataset should be pseudorandomly reshuffled each time it is iterated over.
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2) # prefetch 2 batches of 100 samples each
    # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) 
    # prefetch operates on the elements of the input dataset. It has no concept of examples vs. batches. 
    # examples.prefetch(2) will prefetch two elements (2 examples), while examples.batch(20).prefetch(2) will prefetch 2 elements (2 batches, of 20 examples each).  
    
    # ------- Optional 
    # options = tf.data.Options()
    # options.experimental_threading.max_intra_op_parallelism = 1
    # data = data.with_options(options)
    
    return dataset

def process_data_v4 (filename, labels, split):
    
    data = tf.py_function(load_matfile, inp=[filename, split], Tout=tf.float16)
    data.set_shape([5,4,3000])
    labels.set_shape([5])
    return data, labels

def ds_generator_v5 (data, labels, split, shuffle_buffer_size, batch_size, seed, shuffle=False, catched=False, sub_sample=-1):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))   
    dataset = dataset.map(lambda x,y: process_data_v4(x, y, split), num_parallel_calls=tf.data.AUTOTUNE)
    if catched == True:
        dataset = dataset.cache()
    
    if shuffle == True:
        dataset = dataset.shuffle(shuffle_buffer_size, seed, reshuffle_each_iteration=True) # important to make the suffle size large enough or else shuffling will not be effective.

    if sub_sample != -1:
        dataset = dataset.take(min(len(data), round(sub_sample)))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE) # prefetch

    return dataset

def ds_generator_v6 (data, labels, split, shuffle_buffer_size, batch_size, seed, shuffle=False, catched = False, sub_sample = -1):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    if catched == True:
        dataset = dataset.cache()
    
    if shuffle == True:
        dataset = dataset.shuffle(shuffle_buffer_size, seed, reshuffle_each_iteration=True) # important to make the suffle size large enough or else shuffling will not be effective.

    if sub_sample != -1:
        dataset = dataset.take(min(len(data), round(sub_sample)))   
    
    dataset = dataset.map(lambda x,y: process_data_v4(x, y, split), num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
# ______________________________________________________________________________

# TFRECORDS --------------------------------------------------------------------          
def parse_tfrecord (example):
    feature_description = {
            "mat": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.string)}
    
    example = tf.io.parse_single_example(example, feature_description)
    
    example['mat'] = tf.io.parse_tensor(example["mat"], out_type=tf.float32)
    example['label'] = tf.io.parse_tensor(example["label"], out_type=tf.float32)

    return example['mat'], example['label']    

def tfr_dataset_generator(data, buffer_size, batch_size, seed, shuffle=False):
    
   dataset = tf.data.TFRecordDataset(data)
   
   dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE) 
   dataset = dataset.cache()

   if shuffle == True:
      dataset = dataset.shuffle(buffer_size, seed, reshuffle_each_iteration=True)
   
   dataset = dataset.batch(batch_size)
   dataset = dataset.prefetch(2)
   
   return dataset
# ______________________________________________________________________________       
# ______________________________________________________________________________ 


# ··············································································
# CALLBACKS
# ··············································································
class IntervalEvaluation(keras.callbacks.Callback): 
    '''Evaluates the NN with the validation dataset every setted frequency and stops trainning with a patience batch parameter.
    Saves training and validation results (batch, loss, accuracy and kappa) in a csv file'''
    
    def __init__(self, model, training_batch_generator, validation_batch_generator, patience, frequency, tr_logs, val_logs):

        self.validation_batch_generator = validation_batch_generator
        self.best_value = None
        self.min_values = []
        self.patience = patience
        self.frequency = frequency
        self.model = model
        self.training_batch_generator = training_batch_generator
        self.idx = 0
        self.tr_logs = tr_logs
        self.val_logs = val_logs
        self.metrics = [tf.metrics.Accuracy(), tfa.metrics.CohenKappa(num_classes=5)]
        self.best_weights = None
        
        self.total_batches = len(self.training_batch_generator) # total number of batches
        print('\nTotal batches: ', self.total_batches)
        self.step = round(len(self.training_batch_generator)*self.frequency) # step frequency for validation
        print('Frequency step: ', self.step) 
           
    def on_batch_end(self, batch, logs=None):
        
        if batch % self.step == 0:
            
            copi = copy.deepcopy(self.model)
            metrics = copi.evaluate(self.validation_batch_generator) # batch evaluation 
                      
            if (self.best_value == None):
               self.best_value = metrics[0] 
              
            if metrics[0] < self.best_value: # condition:if metric is lower than best_value, best_value=metric and reset min_values list.
               self.best_value = metrics[0]
               self.best_weights = self.model.get_weights() 
               self.min_values = [] 

            elif metrics[0] >= self.best_value:
               self.min_values.append(metrics[0]) # accumulate worst loss values

            with open(self.val_logs, 'a') as file:
                file.write(str(self.idx)+','+str(metrics[0])+','+str(metrics[1])+','+str(metrics[2]))
                file.write('\n')                   
            
            if len(self.min_values) == self.patience:
                self.model.stop_training = True # stops training
                self.model.set_weights(self.best_weights) 

        with open(self.tr_logs, 'a') as file:
            file.write(str(self.idx)+','+str(logs['loss'])+','+str(logs['accuracy'])+','+str(logs['Cohen Kappa']))
            file.write('\n') 
              
        
        self.idx += 1
 # ______________________________________________________________________________
       
# LOSS -------------------------------------------------------------------------
# Loss function for optimization process - designed to be minimized
def loss_function(weights, ensemble, batch_generator, metrics): 
    # important: must be in the form f(x, *args), where x is the weights argument. 
    normalized = normalize(weights)	# normalize weights
    predictions = ensemble_predictions(ensemble, normalized, batch_generator)
    performance = get_performance(metrics, batch_generator.labels, predictions)      
	  # calculate error rate
    return 1.0 - performance[1]        
# ______________________________________________________________________________

# LEARNING RATE SCHEDULE -------------------------------------------------------      
# class MyLRSchedule(keras.callbacks.Callback):
#     '''Drop the learning rate by given drop every given epochs (drop_frequency).'''
#     def __init__ (self, model, drop_freq, drop):
#         self.model = model
#         self.drop_freq = drop_freq
#         self.drop = drop
#         self.adj_epoch = drop_freq
        
#     def on_epoch_end(self, epoch, logs=None):
        
#         if epoch + 1 == self.adj_epoch: # adjust the learning rate
#             lr = float(tf.keras.backend.get_value(self.model.optimizer.lr)) # get the current learning rate
#             new_lr = lr * self.drop
#             self.adj_epoch += self.drop_freq
            
#             print('\nNew adj_epoch: ', self.adj_epoch)
#             print('\nOn end epoch ', epoch+1, ' lr was adjusted from ', lr, ' to ', new_lr)
            
#             tf.keras.backend.set_value(self.model.optimizer.lr, new_lr) # set the learning rate in the optimizer            
# ______________________________________________________________________________       
# ______________________________________________________________________________ 


# ··············································································
# PERFORMANCE
# ··············································································
def external_evaluation (main_path, e_folder, model, evaluation, num_classes, batch_size, seed, split):
    '''Gives model performance using TS datasets and using full data from each database.
    To change between one evaluation or another type evaluation="ts" if you want the performance
    using test_dataset or evaluation="full" if you want the performance using full data.'''
    
    dicc = {}
    
    if evaluation == 'ts':
        
        for root, dirs, files in os.walk(main_path):
            if root.endswith(e_folder):
                os.chdir(root)
                data = np.load('test_data.npy')
                labels = np.load('test_labels.npy')
                # Build test dataset generator
                batch_generator = My_Custom_Generator(data, labels, batch_size, seed, split) 
                # --------------- TEST
                # Evaluate model with full dataset
                results = model.evaluate(batch_generator, verbose=0)
                dicc[root[41:]] = results
    
        return dicc
           
    if evaluation == 'full':
        
        for root, dirs, files in os.walk(main_path):
            if root.endswith(e_folder):
                os.chdir(root)
                data = np.load('full_data.npy')
                labels = np.load('full_labels.npy') 
                # Build test dataset generator
                batch_generator = My_Custom_Generator(data, labels, batch_size, seed, split) 
                # --------------- TEST
                # Evaluate model with full dataset
                results = model.evaluate(batch_generator, verbose=0)
                dicc[root[41:]] = results

        return dicc

def get_performance (metrics, true_labels, predictions):
    '''It computes model performance'''
    
    performance = []
    for metric in metrics:
        metric.update_state(true_labels, predictions) # accumulates confusion matrix statistics
        perf = metric.result().numpy() # gives performance value into np format
        performance.append(perf) # append kappa and accuracy in a list
        metric.reset_state()
        
    return performance
# ______________________________________________________________________________  
 
# PREDICTIONS ------------------------------------------------------------------
def get_predictions(ensemble, batch_generator, method):
    '''It computes ensemble predictions and gives the final result using different methods.
            - Max Voting: the predictions which we get from the majority of the models are used as the final prediction
            - Averaging: we take an average of predictions from all the models and use it to make the final prediction
            
        '''

    if method == 'averaging':
        
        preds = []
        for model in ensemble:
            out = model.predict(batch_generator) # predictions probability by class (5 values for each data)
            preds.append(out)  # append predictions for every model
        
        predictions = np.average(preds, axis=0) # compute the average along the specified axis (0 = by row)
        pred_class = np.argmax(predictions, axis=1) # index of averaged max value = predicted class (1 = by column)
            
        return pred_class
    
    if method == 'voting':
        
        preds = []
        for model in ensemble:
            out = model.predict(batch_generator) # predictions probability by class
            max_prob_class = np.argmax(out, axis=1) # get array index of max value => class (by column)
            preds.append(max_prob_class) # append predictions for every model (dims: len(test_data), num_models)
        
        final_preds, counts = mode(preds, axis=0) # returns an array of the most common value in the array. 
        # counts: count of the modal value (by row)
        
        return final_preds 
 
# To use weighted method: all models are assigned different weights defining the importance of each model for prediction
def ensemble_predictions(ensemble, weights, batch_generator):
    '''Make an ensemble prediction for multi-class classification'''
	  # make predictions
    preds = [model.predict(batch_generator) for model in ensemble]
	  # weighted sum across ensemble members
    weighted_preds = np.tensordot(preds, weights, axes=(0,0))
    
	  # argmax across classes
    pred_class = np.argmax(weighted_preds, axis=1) # get predicted classes
    pred_class = keras.utils.to_categorical(pred_class, num_classes=5) # classes to categorical
    
    return pred_class
# ______________________________________________________________________________  

# WEIGHTS ----------------------------------------------------------------------
def get_weights (labels):
    '''Compute weights by class to pay more attention to samples from an under-represented class'''
    cat_inverse = np.argmax(labels, axis=1) # labels categorical inverse
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(cat_inverse), y=cat_inverse)
    norm_weights= normalize(weights)

    dicc={}

    for i in np.unique(cat_inverse):
        dicc[i]=norm_weights[i]
    
    return dicc
# ______________________________________________________________________________  

# NORMALIZE --------------------------------------------------------------------
def normalize(weights):
        
	# calculate l1 vector norm
    result = norm(weights, 1)
	# check for a vector of all zeros
    if result == 0.0:
        return weights
	# return normalized vector (unit norm)
    return weights / result
# ______________________________________________________________________________  

# GRID SEARCH ------------------------------------------------------------------
def grid_search (ensemble, batch_generator, w, metrics):
    ''' Search the best weights that give the best ensemble performance.'''
    
    best_score, best_weights = 0, []
    
    # iterate all possible combinations (cartesian product)
    for weights in product(w, repeat=len(ensemble)): 
    	# skip if all weights are equal
        if len(set(weights)) == 1:
            continue
    	# normalize weight vector
        weights = normalize(weights)
        
        # make predictions
        predictions = ensemble_predictions(ensemble, weights, batch_generator) # with validation dataset
    	# evaluate weights 
        score = get_performance(metrics, batch_generator.labels, predictions) # accuracy and kappa using validation dataset
        
        if score[1] > best_score: # compare kappa metric
            best_score, best_weights = score[1], weights
    
    # one more time for case in which weights are equal
    weights = normalize(np.ones(len(ensemble)))
    predictions = ensemble_predictions(ensemble, weights, batch_generator) # with validation dataset
	  # evaluate weights 
    score = get_performance(metrics, batch_generator.labels, predictions) # accuracy and kappa using validation dataset
    
    if score[1] > best_score: # compare kappa metric
        best_score, best_weights = score[1], weights
    
    return best_score, best_weights
# ______________________________________________________________________________  
# ______________________________________________________________________________ 


# ··············································································
# SAVE RESULTS
# ··············································································
def get_run_logdir (root_logdir, csv=False):
    run_id = time.strftime("run_%Y-%m-%d")
    if csv:
        run_id = time.strftime("run_%Y-%m-%d.csv")
    
    return os.path.join(root_logdir, run_id)
   
def dicc_to_csv (filename, dicc, path):
    '''It writes dictionaries' results into csv format. 
    Dictionary and file name as input.'''
    
    with open(filename, 'w+') as file:
        file.write('Model'+','+path)
        file.write('\n')
        file.write('Database'+','+'loss'+','+'accuracy'+','+'kappa')
        file.write('\n')
        for key, value in dicc.items():
            file.write(str(key)+','+str(value[0])+','+str(value[1])+','+str(value[2]))
            file.write('\n')
    
    return file

def event_to_csv (tensorboard_event, tags, filename):
    '''Tensorboard event to csv. Generates a csv file with epoch logs 
    (loss, accuracy and kappa) from tensorboard event.'''
    
    dicc = { i : [] for i in tags }

    for e in tf.compat.v1.train.summary_iterator(tensorboard_event):
        for v in e.summary.value:
            for key, value in dicc.items():
                if v.tag == key:
                    num = float(tf.make_ndarray(v.tensor))
                    dicc[key].append(num)
                                         
    with open(filename, 'w+') as file:
        writer = csv.writer(file)
        writer.writerow(dicc.keys())
        writer.writerows(zip(*dicc.values()))
        
    return file
# ______________________________________________________________________________       
# ______________________________________________________________________________ 
