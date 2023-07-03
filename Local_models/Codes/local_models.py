#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:27:45 2022

@author: adriana
"""

# ··············································································
# BEFORE START 
# ··············································································
# · How to run:  python local_models.py
# TODO · IMPORTANT: make sure functions.py is located in the same folder as this file   
# This code builds local models (CNN-LSTM architecture) using their corresponding TR, VAL and TS datasets
#
# FIRST BLOCK: global parameters setting, upload data and create datasets
# weight_cross parameter decides if the training takes into account data distribution
#
# SECOND BLOCK: model definition and compilation #TODO: add or remove callbacks according to user preference
# Tensorboard callbacks return training and validation metrics (stored in tensorboard folder) - test results are saved in a different csv file (test folder)
#
# THIRD BLOCK: model training and evaluation
# Callback tensorboard includes only training metrics - run using: tensorboard --logdir <path_to_log> command line
# Callback csvlogger includes training and validation metrics per epoch in csv format
# ··············································································

# LIBRARIES ····································································
import random as python_random
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import numpy as np
import time
import os
import functions # it is necessary to be located in "functions.py" path to import it
# ··············································································

# VERSIONS ·····································································
# python 3.9.7
# tensorflow 2.9.1
# tensorflow addons 0.17.1
# keras 2.9.0
# numpy 1.21.5
# ··············································································
    

# ··············································································
# GLOBAL PARAMETERS 
# Change them depending on user preferences
# ··············································································
INDEX = 0
SEED = 0

# Configuration ··················
# It's not necessary to set seeds for individual initializers in the code if we do these steps 
# because their seeds are determined by the combination of the seeds setted above.
np.random.seed(SEED)
tf.random.set_seed(SEED)
python_random.seed(SEED)
# ································

SERVER = False

if SERVER:
    PARENT_PATH = os.path.abspath('path') 
else:    
    PARENT_PATH = os.path.abspath('path')

# ··· Model configuration
MODEL_TYPE = 'lstm' # (cnn or lstm) # TODO · Change "process_data" and "_fixup_shape" in functions.py 
NUM_CLASSES = 5
EPOCHS = 100
FILTERS = 8
UNITS = 100 
LR = 0.001 
MOMENTUM = 0.9 
PATIENCE = 5
LOSS = keras.losses.CategoricalCrossentropy()
# VAL_FREQ = 0.2 # step validation frequency - less value → more frequency <--- only in IntervalEvaluation()
# DROP = 0.5 # lr decay 
# DROP_FREQ = 4 # how often is the learning rate reduced (by epoch)
WEIGHT_CROSS = False
BATCH_NORM = False
DROPOUT = True
METRICS = ['accuracy', tfa.metrics.CohenKappa(num_classes=NUM_CLASSES, name='Cohen Kappa')]
OPTIMIZER = keras.optimizers.SGD(learning_rate=LR, momentum=MOMENTUM) 

# ··· Datasets configuration
DATABASES = ['ISRUC', 'SHHS', 'DREAMS', 'Telemetry', 'HMC-APR-2018', 'Dublin']
E_FOLDER = 'e5-noFil-EpochNorm'
GENERATOR_VERSION = 'v2' # v1 → ds_generator_v5(), v2 → My_Custom_Generator() or v3 → tfr_dataset_generator() | definitions can be found in functions.py
SPLIT = int(E_FOLDER[1])
BATCH_SIZE = 100
TR_SHUFFLE = True
TS_SHUFFLE = False
# ··············································································


# ··············································································
# MAIN CODE     
# ··············································································
for folder in os.listdir(PARENT_PATH): 
    neWpath = os.path.join(PARENT_PATH, folder) 
    
    for root, dirs, files in os.walk(neWpath):              
        if root.endswith(E_FOLDER + '/datasets'):
            os.chdir(root)

            # ··· Create datasets using preselected generator
            if GENERATOR_VERSION == 'v1': # v1 → ds.generator_v5()
            
                # ··· Load datasets
                if SERVER:
                    train_data = np.load('server_train_data.npy')
                    test_data = np.load('server_test_data.npy')
                    val_data = np.load('server_val_data.npy')
                else:
                    train_data = np.load('train_data.npy')
                    test_data = np.load('test_data.npy')
                    val_data = np.load('val_data.npy')
                    
                train_labels = np.load('train_labels.npy')
                test_labels = np.load('test_labels.npy')
                val_labels = np.load('val_labels.npy')
                
                train_buffer_size = len(train_data)
                test_buffer_size = len(test_data)
                val_buffer_size = len(val_data)
        
                # ··· Build dataset for each database
                if BATCH_SIZE == 1:
                    train_dataset = functions.ds_generator_v5(train_data, train_labels, SPLIT, train_buffer_size, train_buffer_size, SEED, shuffle=True)
                    test_dataset = functions.ds_generator_v5(test_data, test_labels, SPLIT, test_buffer_size, test_buffer_size, SEED)
                    val_dataset = functions.ds_generator_v5(val_data, val_labels, SPLIT, val_buffer_size, val_buffer_size, SEED)
                else:
                    train_dataset = functions.ds_generator_v5(train_data, train_labels, SPLIT, train_buffer_size, BATCH_SIZE, SEED, shuffle=True)
                    test_dataset = functions.ds_generator_v5(test_data, test_labels, SPLIT, test_buffer_size, BATCH_SIZE, SEED)
                    val_dataset = functions.ds_generator_v5(val_data, val_labels, SPLIT, val_buffer_size, BATCH_SIZE, SEED)
            
            if GENERATOR_VERSION == 'v2': # v2 → My_custom_generator()
                
                # ··· Load datasets
                if SERVER:
                    train_data = np.load('server_train_data.npy')
                    test_data = np.load('server_test_data.npy')
                    val_data = np.load('server_val_data.npy')
                else:
                    train_data = np.load('train_data.npy')
                    test_data = np.load('test_data.npy')
                    val_data = np.load('val_data.npy')
                    
                train_labels = np.load('train_labels.npy')
                test_labels = np.load('test_labels.npy')
                val_labels = np.load('val_labels.npy')
                
                train_buffer_size = len(train_data)
                test_buffer_size = len(test_data)
                val_buffer_size = len(val_data)
        
                # ··· Build dataset for each database
                if BATCH_SIZE == 1:
                    train_dataset = functions.My_Custom_Generator(train_data, train_labels, train_buffer_size, SEED, SPLIT, shuffle=True) 
                    val_dataset = functions.My_Custom_Generator(val_data, val_labels, val_buffer_size, SEED, SPLIT) 
                    test_dataset = functions.My_Custom_Generator(test_data, test_labels, test_buffer_size, SEED, SPLIT)     
                else:
                    train_dataset = functions.My_Custom_Generator(train_data, train_labels, BATCH_SIZE, SEED, SPLIT, shuffle=True) 
                    val_dataset = functions.My_Custom_Generator(val_data, val_labels, BATCH_SIZE, SEED, SPLIT) 
                    test_dataset = functions.My_Custom_Generator(test_data, test_labels, BATCH_SIZE, SEED, SPLIT)     
                                
            if GENERATOR_VERSION == 'v3': # v3 → tfr_dataset_generator()

                # ··· Load datasets
                train_data = tf.data.Dataset.list_files('/*.tfrecord')
                val_data = tf.data.Dataset.list_files('/*.tfrecord')
                test_data = tf.data.Dataset.list_files('/*.tfrecord')
                
                tr_buffer_size = len(train_data)
                val_buffer_size = len(val_data)
                ts_buffer_size = len(test_data)
                
                # ··· Build dataset for each database
                train_dataset = functions.tfr_dataset_generator(train_data, tr_buffer_size, BATCH_SIZE, SEED, shuffle=True) # shuffles the order of batches on epoch end
                val_dataset = functions.tfr_dataset_generator(val_data, val_buffer_size, BATCH_SIZE, SEED) # shuffle=False (default) 
                test_dataset = functions.tfr_dataset_generator(test_data, ts_buffer_size, BATCH_SIZE, SEED) 
    # ·········································································              
    print('··········································\n')
    print('Local model: M({0})'.format(folder))
    print('{0} configuration: split = {1}, batchnorm = {2}, dropout = {3}, batch_size = {4}, learning_rate = {5}, momentum = {6}, loss = {7}, weight_cross = {8}, patience = {9}, epochs = {10}'.format(MODEL_TYPE,
            SPLIT, BATCH_NORM, DROPOUT, BATCH_SIZE, LR, MOMENTUM, LOSS.name, WEIGHT_CROSS, PATIENCE, EPOCHS))
    print('··········································') 
    # ·········································································

    # LOGS ····································································   
    RUN_LOGS = functions.get_run_logdir(os.path.join(neWpath, E_FOLDER, 'logs/tensorboard')) # tensorboard runs (train and validation events)
    TR_VAL_CSV = functions.get_run_logdir(os.path.join(neWpath, E_FOLDER, 'logs/tensorboard'), csv=True) # tensorboard runs (train and valitadion csv)
    TS_LOGS = functions.get_run_logdir(os.path.join(neWpath, E_FOLDER, 'logs/test'), csv=True) # test logs folder and files
    CHECKS = functions.get_run_logdir(os.path.join(neWpath, E_FOLDER, 'models/checkpoints')) # checkpoints
    MODEL_NAME = functions.get_run_logdir(os.path.join(neWpath, E_FOLDER, 'models')) # final model          
    # TR_LOGS = functions.get_run_logdir(os.path.join(root, 'logs/train'), csv=True) # training logs folder -----> only with IntervalEvaluation()
    # VAL_LOGS = functions.get_run_logdir(os.path.join(root, 'logs/validation'), csv=True) # validation logs folder -----> only with IntervalEvaluation()
    if WEIGHT_CROSS:
        weights = functions.get_weights(train_labels) # get weights by class to pay more attention to samples from an under-represented class
    # ·········································································

    # COMPILATION PARAMETERS ··················································
    EARLY_STOP = keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1) # controls how training should stop
    CHECK_POINT = keras.callbacks.ModelCheckpoint(filepath=CHECKS, monitor='val_loss', save_best_only=False, save_freq='epoch') # stores best models during training
    TENSORBOARD = keras.callbacks.TensorBoard(log_dir=RUN_LOGS, update_freq='batch', profile_batch='1,20') # creates one folder by execution with two subfolders with train and validation events
    CSV_LOGGER = keras.callbacks.CSVLogger(TR_VAL_CSV) # training and validation results by epoch in csv format 
    # ·········································································  

    # ·········································································
    # MODEL 
    # ·········································································
    def create_keras_model():
        '''Creates neural network. CNN-LSTM configuration'''
        # ··· Create neuronal network
        cnn = keras.Sequential();
        
        # N operational blocks
        n = 3 # number of operational blocks
        
        for block in range(1, n+1):
            if block == 1:
                cnn.add(keras.layers.Conv1D(FILTERS, kernel_size=100, strides=1, padding='same', input_shape = (4, 3000, 1), trainable=True)) # input 4x3000 matrix (2xEEG, EMG, EOG)
                
            if block > 1:
                FILTERS = FILTERS*2
                cnn.add(keras.layers.Conv1D(FILTERS, kernel_size=100, strides=1, padding='same', trainable=True))
            
            cnn.add(keras.layers.Activation('relu', trainable=True))
            if BATCH_NORM:
                cnn.add(keras.layers.BatchNormalization(trainable=True))    
            cnn.add(keras.layers.AveragePooling2D(pool_size=[1,2], strides=[1,2], trainable=True))
        
        # Output block  
        cnn.add(keras.layers.Flatten(trainable=True))
        cnn.add(keras.layers.Dense(50, activation='relu', trainable=True))
        if DROPOUT:
            cnn.add(keras.layers.Dropout(0.5)) # applies dropout to the input, randomly selected 
            # noise_shape=(batch_size, timesteps, features) - None default. Is a 1D tensor representing the shape of the binary dropout mask. 
            # If we set any parameter to "1", the same dropout will be applied to all 
        
        if MODEL_TYPE == 'lstm':
            # --- Define LSTM 
            lstm = keras.Sequential()
            lstm.add(keras.layers.TimeDistributed(cnn, input_shape=(SPLIT, 4, 3000, 1)))
            lstm.add(keras.layers.LSTM(units=UNITS, return_sequences=False))
            lstm.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))  
            return lstm
        
        else:
            cnn.add(keras.layers.Dense(NUM_CLASSES, activation='softmax')) 
            return cnn
    # ·········································································
    print('Creating keras model...')
    model = create_keras_model()
    model.summary()
    # ·········································································

    # EVALUATION BY BATCHES ···················································
    # IVAL = functions.IntervalEvaluation(model, train_dataset, val_dataset, PATIENCE, VAL_FREQ, TR_LOGS, VAL_LOGS) # validate each X batches (VAL_FREQ) and controls training
    # ·········································································

    # LEARNING RATE SCHEDULE···················································
    # Change learning rate during training
    # lr_schedule = functions.MyLRSchedule(model, DROP_FREQ, DROP) # add in CALLBACKLIST
    # ·········································································

    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
    CALLBACKLIST = [EARLY_STOP, CHECK_POINT, CSV_LOGGER]  
    # ·········································································

    # TRAIN ···································································
    print('Starting training...')
    start = time.time() # gives time in seconds

    if WEIGHT_CROSS:
        history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, class_weight=weights, callbacks=CALLBACKLIST)
        # class_weight: dictionary mapping class indices (integers) to a weight (float) value, useful to tell the model to "pay more attention" to samples from an under-represented class. 
    else:
        history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=CALLBACKLIST)

    stop = time.time()
    print('··········································')
    print('Time: {0:.2f} min'.format((stop-start)/60))
    print('··········································')
    # ·········································································

    # TEST ····································································
    print('Testing...')
    tr_results = model.evaluate(train_dataset, verbose=0) # progression bar not shown
    val_results = model.evaluate(val_dataset, verbose=0)
    ts_results = model.evaluate(test_dataset, verbose=0) 
        
    print('\n* Local performance ')
    print('Train metrics: loss = {0}, acc = {1}, kappa = {2}'.format(tr_results[0], tr_results[1], tr_results[2]))
    print('Validation metrics: loss = {0}, acc = {1}, kappa = {2}'.format(val_results[0], val_results[1], val_results[2]))
    print('Local test metrics: loss = {0}, acc = {1}, kappa = {2}'.format(ts_results[0], ts_results[1], ts_results[2]))
    # ·········································································

    # ·········································································
    # SAVE RESULTS 
    # ·········································································
    model.save(MODEL_NAME) 

    with open (TS_LOGS, 'w+') as file:
        file.write('Database'+','+ folder)      
        file.write('\nTraining time: {} seconds'.format(stop-start))
        file.write('\nDataset' +','+'loss'+','+'accuracy'+','+'kappa')
        file.write('\ntrain'+','+str(tr_results[0])+','+str(tr_results[1])+','+str(tr_results[2])+'\n')
        file.write('\nvalidation'+','+str(val_results[0])+','+str(val_results[1])+','+str(val_results[2])+'\n')
        file.write('\ntest'+','+str(ts_results[0])+','+str(ts_results[1])+','+str(ts_results[2])+'\n')
        file.write('\n')
    # ·········································································

    # CLEAR SESSION ···························································
    tf.keras.backend.clear_session() # resets all state generated by Keras
    # ·········································································
