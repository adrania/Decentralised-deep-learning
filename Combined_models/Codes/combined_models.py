#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:26:32 2023

@author: adriana
"""

# ··············································································
# BEFORE START 
# ··············································································
# Run this code using "exec.sh"
# change "exec.sh" code in python line to use one script or another
# How to run:   ./exec.sh <path_to_Combine_folder>   
# TODO · Make sure "functions.py" is located in the same folder as "exec.sh" file
#    
# This code builds combined models using one-leave-out strategy in order to perform generalization using external databases
# · First block: global parameters setting, upload data and create datasets (just train and validation)
# · Second block: model definition and compilation.  TODO: uncomment callback's lines if necessary
# · Third block: model training and evaluation using train and validation datasets
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
import functions
# ··············································································

# ··············································································
# GLOBAL PARAMETERS 
# ··············································································
ROOT = os.getcwd()
os.chdir(ROOT)

SEED = 0 

# Configuration ····················
# Set seeds to obtain reproducible results - global seeds
# It's not necessary to set seeds for individual initializers in the code if we do these steps
# because their seeds are determined by the combination of the seeds set above.
np.random.seed(SEED)
tf.random.set_seed(SEED)
python_random.seed(SEED)
# ··································

# ··· Model configuration
EPOCHS = 30
BATCH_SIZE = 100
SPLIT = 5 
NUM_CLASSES = 5 
FILTERS = 8
UNITS = 100 
LR = 0.001
MOMENTUM = 0.9
LOSS = keras.losses.CategoricalCrossentropy()
BATCH_NORM = False
WEIGHT_CROSS = False # if True results are computed using weights by class distribution 
PATIENCE = 5
MONITOR = 'val_loss' # validation on epoch end
# VAL_FREQ = 0.2 # step validation frequency - less value → more frequency ---> only in IntervalEvaluation()
# DROP = 0.5 # lr decay ---> only if lr_schedule
# DROP_FREQ = 4 # how often is the learning rate reduced (by epoch) ---> only if lr_schedule

# ··· Datasets configuration
TS_SHUFFLE = False

# ··· Logs
RUN_LOGS = functions.get_run_logdir(os.path.join(ROOT, 'logs/tensorboard')) # tensorboard runs (train and validation events)
TR_VAL_CSV = functions.get_run_logdir(os.path.join(ROOT, 'logs/tensorboard'), csv=True) # tensorboard runs into csv
TS_LOGS = functions.get_run_logdir(os.path.join(ROOT, 'logs/test'), csv=True) # final evaluations of train and validation datasets
CHECKS = functions.get_run_logdir(os.path.join(ROOT, 'models/checkpoints')) # checkpoints
MODEL_NAME = functions.get_run_logdir(os.path.join(ROOT, 'models')) # final model 
# TR_LOGS = functions.get_run_logdir(os.path.join(root, 'logs/train'), csv=True) # training logs folder ---> only with IntervalEvaluation()
# VAL_LOGS = functions.get_run_logdir(os.path.join(root, 'logs/validation'), csv=True) # validation logs folder ---> only with IntervalEvaluation()
# ··············································································

# ··············································································
# MAIN CODE
# ··············································································
# ··· Load datasets
train_data = np.load('datasets/train_data.npy')
train_labels = np.load('datasets/train_labels.npy')
val_data = np.load('datasets/val_data.npy')
val_labels = np.load('datasets/val_labels.npy')

# ··· Build datasets
training_batch_generator = functions.My_Custom_Generator(train_data, train_labels, BATCH_SIZE, SEED, SPLIT) 
validation_batch_generator = functions.My_Custom_Generator(val_data, val_labels, BATCH_SIZE, SEED, SPLIT, shuffle=TS_SHUFFLE) 

# ··· Weights by class: to pay more attention to samples from an under-represented class
# dictionary form (las claves son los indices de posicion - se multiplica el peso por su posicion correspondiente en el label)
weights = functions.get_weights(train_labels) 
# ··············································································

# ··············································································
# MODEL 
# ··············································································    
cnn = keras.Sequential();

# Operational blocks
n = 3 # number of operational blocks

for block in range(1, n+1):
    if block == 1:
        cnn.add(keras.layers.Conv1D(FILTERS, kernel_size=100, strides=1, padding='same', input_shape = (4, 3000, 1))) # input as a 4x3000 matrix (2xEEG, EMG, EOG)
        
    if block > 1:
        FILTERS = FILTERS*2
        cnn.add(keras.layers.Conv1D(FILTERS, kernel_size=100, strides=1, padding='same'))
    
    cnn.add(keras.layers.Activation('relu'))
    if BATCH_NORM:
        cnn.add(keras.layers.BatchNormalization())
    cnn.add(keras.layers.AveragePooling2D(pool_size=[1,2], strides=[1,2]))
  
# Output block  
cnn.add(keras.layers.Flatten())
cnn.add(keras.layers.Dense(50, activation='relu'))

# LSTM
lstm = keras.Sequential()
lstm.add(keras.layers.TimeDistributed(cnn, input_shape=(SPLIT, 4, 3000, 1)))
lstm.add(keras.layers.LSTM(units=UNITS, return_sequences=False))
lstm.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
# ··············································································
cnn.summary()
lstm.summary()
# ··············································································

# Compilation parameters ·······················································
metrics = ['accuracy', tfa.metrics.CohenKappa(num_classes=NUM_CLASSES, name='Cohen Kappa')] # CohenKappa: sparse_label=False default, if True => multilabel problems
loss = LOSS
optimizer = keras.optimizers.SGD(learning_rate=LR, momentum=MOMENTUM) # Stochastic gradient descent

# ··· Callbacks
early_stop = keras.callbacks.EarlyStopping(monitor=MONITOR, patience=PATIENCE, restore_best_weights=True) # controla cuando debe terminar el entrenamiento
# ival = functions.IntervalEvaluation(lstm, training_batch_generator, validation_batch_generator, patience, val_freq, tr_logs, val_logs) # valida cada X batches (val_freq) y controla el entrenamiento
check_point = keras.callbacks.ModelCheckpoint(filepath=CHECKS, monitor=MONITOR, save_best_only=False, save_freq='epoch') # va almacenando los mejores modelos durante el entrenamiento
tensorboard = keras.callbacks.TensorBoard(log_dir=RUN_LOGS, update_freq='batch') # crea una carpeta por ejecucion con dos subcarpetas que continen los eventos de train y validacion
csvlogger = keras.callbacks.CSVLogger(TR_VAL_CSV) # almacena en un csv los resultados de entrenamiento y validacion por epoch, corresponden a la performance del modelo en el momento del entrenamiento en el que se encuentra
# si queremos los resultados por batch tenemos que iterar sobre los eventos de tensorboard con el script tensorboard.py

# ··· Change learning rate during training
# lr_schedule = functions.MyLRSchedule(lstm, drop_freq, drop) #TODO: add this parameter in model.fit callbacks

callbacklist = [early_stop, check_point, tensorboard, csvlogger]

# ··· Compile model
lstm.compile(optimizer=optimizer, loss=loss, metrics=metrics)
# ··············································································

# ··············································································
# TRAINING 
# ··············································································
start = time.time() # gives time in seconds
if WEIGHT_CROSS == True:
    history = lstm.fit(training_batch_generator, epochs=EPOCHS, validation_data=validation_batch_generator, class_weight=weights, callbacks=callbacklist, verbose=0)
    # class_weight parameter → dictionary mapping class indices (integers) to a weight (float) value, useful to tell the model to "pay more attention" to samples from an under-represented class. 
else:
    history = lstm.fit(training_batch_generator, epochs=EPOCHS, validation_data=validation_batch_generator, callbacks=callbacklist, verbose=0)
stop = time.time()
# ··············································································

# ··············································································
# TEST 
# ··············································································
tr_results = lstm.evaluate(training_batch_generator, verbose=0) 
val_results = lstm.evaluate(validation_batch_generator, verbose=0)
print('\n** LOCAL RESULTS')
print('Train metrics:\tloss={0}, acc = {1}, kappa = {2}'.format(tr_results[0], tr_results[1], tr_results[2]))
print('Validation metrics:\tloss={0}, acc = {1}, kappa = {2}'.format(val_results[0], val_results[1], val_results[2]))
# ··············································································

# SAVE RESULTS ·································································
# TODO: os.makedirs(os.path.dirname(ts_log)) # create test logs folder directory - only first iteration
with open (TS_LOGS, 'w+') as file:
    file.write('Database'+','+ ROOT[41:])
    file.write('\n')        
    file.write('Training time: {} seconds'.format(stop-start))
    file.write('\n')
    file.write('Dataset' +','+'loss'+','+'accuracy'+','+'kappa')
    file.write('\n')
    file.write('train'+','+str(tr_results[0])+','+str(tr_results[1])+','+str(tr_results[2])+'\n')
    file.write('validation'+','+str(val_results[0])+','+str(val_results[1])+','+str(val_results[2])+'\n')
    file.write('\n')

# ··· Save model
lstm.save(MODEL_NAME) 
# ··············································································

# ··············································································
# CLEAR SESSION
# ··············································································
tf.keras.backend.clear_session() # resets all state generated by Keras
# ··············································································
