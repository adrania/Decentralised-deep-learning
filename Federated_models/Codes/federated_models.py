#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 13:29:18 2022

@author: adriana
"""

# BEFORE START ·································································
# ··· Run: 
# (federated) python -u "pythoncode" > "logs_filename" 2>&1 & 
    # 2>&1: errors and logs in the same file 
    # -u: force the stdout and stderr streams to be unbuffered
# Dinamically check code state: tail -f "logs_filename" → RAM/GPU: htop
# ______________________________________________________________________________

# LIBRARIES ····································································
import tensorflow as tf
import tensorflow_addons as tfa 
import tensorflow_federated as tff
import keras
import os
import time
import datetime
import numpy as np 
from operator import itemgetter
from termcolor import colored

import functions # it is necessary to be located in "functions.py" path to import it
# TODO!! change process_data and _fixup_shape in functions.py when using lstm or cnn
# ______________________________________________________________________________

# VERSIONS ·····································································
# python 3.9.7
# tensorflow 2.9.1
# tensorflow addons 0.17.1
# tensorflow federated 0.34.0
# keras 2.9.0
# numpy 1.21.5
# ______________________________________________________________________________

# CONFIGURATION ································································
tf.debugging.set_log_device_placement(False) # if True prints operations and functions location (GPU or CPU)
# ______________________________________________________________________________


# ··············································································
# GLOBAL PARAMETERS 
# ··············································································
INDEX = 0 
SEED = 0

PARENT_PATH = os.path.abspath('path_to_databases') 
PATH_LOG = os.path.abspath('path_to_federated')

MODELS = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'] 
CLIENTS = ['ISRUC', 'SHHS', 'DREAMS', 'Telemetry', 'HMC-APR-2018', 'Dublin'] # databases
MODEL_TYPE = 'lstm' # (cnn or lstm) · TODO: change process_data and _fixup_shape in functions.py when using lstm or cnn
BATCH_NORM = False
DROPOUT = True
E_FOLDER = 'e5-noFil-EpochNorm'
NUM_CLIENTS = 5
REPLACEMENT = False
VAL_CATCHED = True
TR_CATCHED = True
TS_CATCHED = False
FULL_CATCHED = False
SUBSAMPLE = 2000 
SPLIT = 5
LEARNING_ALGORITHM = tff.learning.algorithms.build_fed_sgd
EPOCHS = 1
BATCH_SIZE = 100
NUM_CLASSES = 5
FILTERS = 8
UNITS = 100 
ROUNDS = 10000
LR = 0.001 
MOMENTUM = 0.9 
PATIENCE = 100
LOSS = keras.losses.CategoricalCrossentropy()
# ______________________________________________________________________________


# ··············································································
# MODEL 
# ··············································································
def create_keras_model():
   '''Creates neural network. CNN-LSTM configuration'''
   
   filters = FILTERS
   # --- Create neuronal network
   cnn = keras.Sequential();
   
   ## N operational blocks
   n = 3 # number of operational blocks
   
   for block in range(1, n+1):
       if block == 1:
           cnn.add(keras.layers.Conv1D(filters, kernel_size=100, strides=1, padding='same', input_shape = (4, 3000, 1), trainable=True)) 
           
       if block > 1:
           filters = filters*2
           cnn.add(keras.layers.Conv1D(filters, kernel_size=100, strides=1, padding='same', trainable=True))
       
       cnn.add(keras.layers.Activation('relu', trainable=True))
       if BATCH_NORM:
           cnn.add(keras.layers.BatchNormalization(trainable=True))    
       cnn.add(keras.layers.AveragePooling2D(pool_size=[1,2], strides=[1,2], trainable=True))
   
   ## Output block  
   cnn.add(keras.layers.Flatten(trainable=True))
   cnn.add(keras.layers.Dense(50, activation='relu', trainable=True))
   if DROPOUT:
       cnn.add(keras.layers.Dropout(0.5))
   
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
# ______________________________________________________________________________


# ··············································································
# FEDERATED LEARNING 
# ··············································································
# Server-client performance. Iterations in rounds
# Weighted mean using amount of data by client → data_balance
# Training using random client selection
# ______________________________________________________________________________

date = datetime.datetime.now().strftime("%x-%X") 
date = str(np.char.replace(date, '/', '.'))

# MAIN CODE ····································································
for server in MODELS:
    fed_datasets = [] # training datasets list
    val_datasets = [] # validation datasets list
    len_datasets = [] # training buffer sizes (balanced metrics computation)
    fed = [] # client list
    best_loss = None
    worst_loss = []
    best_weights = []
    best_clients_ids = []
    # ··· Model name and logs
    config = '{0}_{1}_{2}-c{3}-e{4}-b{5}-lr{6}-m{7}/'.format(str(server), str(MODEL_TYPE), E_FOLDER[0:2], str(NUM_CLIENTS), 
                str(EPOCHS), str(BATCH_SIZE), str(LR), str(MOMENTUM)) #LEARNING_ALGORITHM.__name__[-3:])
    client_tb_logs = os.path.join(PATH_LOG, config+date)
    
    # LOAD DATASETS ····························································     
    for folder in os.listdir(PARENT_PATH): 
        neWpath = os.path.join(PARENT_PATH, folder) 

        # Load training and validation datasets
        if (folder in CLIENTS and folder != CLIENTS[INDEX]) :
            for root, dirs, files in os.walk(neWpath):                  
                if root.endswith(E_FOLDER + '/datasets'):
                    os.chdir(root)
                    
                    # ··· Load data (training dataset) 5 clients
                    train_data = np.load('train_data.npy')
                    val_data = np.load('val_data.npy')
                    train_labels = np.load('train_labels.npy')
                    train_buffer_size = len(train_data)
                    val_labels = np.load('val_labels.npy')
                    val_buffer_size = len(val_data)
                
                    # ··· Build dataset for each database/client
                    train_dataset = functions.ds_generator_v6(train_data, train_labels, SPLIT, train_buffer_size, BATCH_SIZE, SEED, shuffle=True, catched=TR_CATCHED, sub_sample=SUBSAMPLE)
                    val_dataset = functions.ds_generator_v5(val_data, val_labels, SPLIT, val_buffer_size, BATCH_SIZE, SEED, catched=VAL_CATCHED)
                        
                    fed_datasets.append(train_dataset) # accumulate train datasets
                    val_datasets.append(val_dataset) # accumulate validation datasets
                    
                    len_datasets.append(train_buffer_size) # datasets len
                    fed.append(folder) # clients list
        
        # Load test dataset, unseen client = external evaluation 
        if folder == CLIENTS[INDEX]:
            for root, dirs, files in os.walk(neWpath):                   
                if root.endswith(E_FOLDER + '/datasets'):
                    os.chdir(root)
                    
                    # ··· Load data (test dataset)
                    test_data = np.load('test_data.npy') # local 
                    full_data = np.load('full_data.npy')
                    test_labels = np.load('test_labels.npy')
                    test_buffer_size = len(test_data)
                    full_labels = np.load('full_labels.npy')
                    full_buffer_size = len(full_data)
                
                    # ··· Build dataset for each database
                    test_dataset = functions.ds_generator_v6(test_data, test_labels, SPLIT, test_buffer_size, BATCH_SIZE, SEED, catched=TS_CATCHED)       
                    full_dataset = functions.ds_generator_v6(full_data, full_labels, SPLIT, full_buffer_size, BATCH_SIZE, SEED, catched=FULL_CATCHED)
        
    fed_dataset_ids = [i for i, data in enumerate(fed_datasets)] # datasets ids
    total_data = sum(len_datasets)
    data_balance = [i/total_data for i in len_datasets] # weights

    # ··········································································                                                    
    print('··········································\n')
    print(colored('FEDERATED MODEL: ', 'cyan'), server)
    print('Test dataset: ', CLIENTS[INDEX])
    print('Client list:\t', fed)
    print('Data weights: ', data_balance)
    print('Num selected clients:\t {0}, sub_sample = {1}, replace = {2}'.format(NUM_CLIENTS, SUBSAMPLE, REPLACEMENT))
    print('Configuration: model_type = {0}, batchnorm = {1}, dropout = {2}, batch_size = {3}, lr = {4}, momentum = {5}, patience = {6}, loss = {7}'.format(MODEL_TYPE,
            BATCH_NORM, DROPOUT, BATCH_SIZE, LR, MOMENTUM, PATIENCE, LOSS.name))
    print('Datasets catched: TR = {0}, VAL = {1}, TS = {2}'.format(TR_CATCHED, VAL_CATCHED, TS_CATCHED))
    print('Iterative process: learning algorithm = {0}, rounds = {1}, epochs = {2}'.format(LEARNING_ALGORITHM.__name__, ROUNDS, EPOCHS))
    print('··········································')
    # ·········································································· 

    # CREATE FEDERATED MODEL ···················································
    # This is the federated cycle. It creates the same model structure by client. Displayed metrics: AUC and ACC during "training" local process
    def model_fn():
        
        client_model = create_keras_model()
        model_fn = tff.learning.from_keras_model(client_model, input_spec=fed_datasets[0].element_spec, loss=LOSS, 
                metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.AUC()])                                                  
        return model_fn
    # ·········································································· 
    
    # ··········································································
    # ITERATIVE PROCESS 
    # ··········································································
    # Initialize iterative process using one learning algorithm (FedSGD, FedAvg...)
    # Next → push the server state to the clients: training on their local data, 
    # collecting and averaging model updates and producing a new server model
    # evaluation using keras method taking into account the amount of data
    # writes clients training metrics into tensorboard events
    # __________________________________________________________________________
    
    print('Creating iterative process...')
    iterative_process = LEARNING_ALGORITHM(
        model_fn,
        server_optimizer_fn = lambda: keras.optimizers.SGD(learning_rate=LR, momentum=MOMENTUM), use_experimental_simulation_loop=True) 
    
    # INITIALIZE FEDERATED ALGORITHM ··········································· 
    print('Initializing federated algorithm...')
    state = iterative_process.initialize() # construct the server state   

    # ··· Create Keras model
    # this is a copy of the model structure, we apply the corresponding final round state to this structure to display Kappa and ACC "training" metrics 
    # (ex. after-round state, which includes every client computed gradients)
    keras_model = create_keras_model() # define model structure
    keras_model.compile(optimizer=keras.optimizers.SGD(learning_rate=LR, 
                        momentum=MOMENTUM), loss=LOSS, 
                        metrics=['accuracy', tfa.metrics.CohenKappa(num_classes=NUM_CLASSES, name='Cohen Kappa')])
    
    # ··· Tensorboard visualization → clients training
    client_summary_writer = tf.summary.create_file_writer(client_tb_logs)
    
    # TRAINING ·································································
    print('Starting federated rounds...')
    print('··········································')
    fl_start = time.time() # state federated training time 
        
    with client_summary_writer.as_default():
        for round_num in range(0, ROUNDS):
            
            # CLIENT SELECTION ·················································
            # Select a random sample of clients on each round
            clients_sample_ids = np.random.choice(fed_dataset_ids, size=NUM_CLIENTS, replace=REPLACEMENT)
            train_sample = itemgetter(*clients_sample_ids)(fed_datasets)
            val_sample = itemgetter(*clients_sample_ids)(val_datasets)
            sample_weights = itemgetter(*clients_sample_ids)(data_balance)  
            # ··································································
            
            # CLIENTS TRAINING (tff) ···········································    
            # "result" variable reflects the performance of the model at the BEGINING of the training round
            start = time.time() # state clients training time
            result = iterative_process.next(state, train_sample)
            stop = time.time()
            state = result.state
            clients_train_metrics = result.metrics 
            print('··················')
            print(clients_train_metrics)
            print('*** TR ROUND {0} - {1:.1f}s: \tacc = {2:.4f}, auc = {3:.4f}, precision = {4:.4f}'.format(round_num, (stop-start), 
                    clients_train_metrics['client_work']['train']['categorical_accuracy'],clients_train_metrics['client_work']['train']['auc'], 
                    clients_train_metrics['client_work']['train']['loss']))
            
            # SERVER TRAINING EVALUATION (keras) ·······························
            # evaluate global model state after-round
            weights = iterative_process.get_model_weights(state) # get after-round model weights
            weights.assign_weights_to(keras_model) # assign server weights to keras_model
            keras_val_metrics = []
            keras_train_metrics = []
            
            start = time.time() # state server evaluation time
            for ds in train_sample:
                k_metric = keras_model.evaluate(ds, verbose=0)
                keras_train_metrics.append(k_metric)
            train_metrics = np.average(keras_train_metrics, axis=0, weights=sample_weights) # we take into account the amount of client data
            stop = time.time()
            print('Server training - {0:.1f}s: \tacc = {1:.4f}, kappa = {2:.4f}, loss = {3:.4f}'.format((stop-start), train_metrics[1], train_metrics[2], train_metrics[0]))


            # VALIDATION (keras) ···············································
            start = time.time()
            for ds in val_sample:
                k_metric = keras_model.evaluate(ds, verbose=0) 
                keras_val_metrics.append(k_metric)    
            val_metrics = np.average(keras_val_metrics, axis=0, weights=sample_weights)
            stop = time.time()
            print('Validation - {0:.1f}s: \tacc = {1:.4f}, kappa = {9:.4f}, loss = {11:.5f}'.format((stop-start), val_metrics[1], val_metrics[2], val_metrics[0]))
            # ··································································

            # EARLY STOPPING ···················································
            # using mean clients loss from validation datasets
            # we need to save clients ids to use them in the final averaged evaluation
            if best_loss == None:
                best_loss = val_metrics[0]
                best_weights = weights
                best_clients_ids = clients_sample_ids
                
            if val_metrics[0] < best_loss: # condition: if loss is lower than best_loss, best_loss = loss and reset worst_loss list.
                best_loss = val_metrics[0]
                best_weights = weights
                best_clients_ids = clients_sample_ids 
                worst_loss = [] 
                
            elif val_metrics[0] >= best_loss:
                worst_loss.append(val_metrics[0]) 
                
            if len(worst_loss) == PATIENCE: # if condition == True → stops training
                print('Patient threshold achieved ... ¡stop training!')    
                break
            # ··································································
            
            # ··· Write client "training" logs into tensorboard event 
            for name, value in clients_train_metrics['client_work']['train'].items():
                tf.summary.scalar(name, value, step=round_num)
            

    fl_stop = time.time()
    print('··········································')
    print('Time training and validation: {0:.2f} h'.format(((fl_stop-fl_start)/60)/60))
    print('··········································')
    # ··········································································

    # ··········································································
    # FINAL SERVER EVALUATION 
    # ··········································································
    print('** Final server evaluation **')

    # ··· Keras external evaluation
    # evaluate final server state in the unseen client
    best_weights.assign_weights_to(keras_model) # assign best weights to keras model 
    test_metrics = keras_model.evaluate(test_dataset, verbose=0) 
    full_metrics = keras_model.evaluate(full_dataset, verbose=0)
    
    # ··· Training and validation evaluation using MEAN
    # it is not necessary to use clients ids. Mean doesn't take into account database balance
    final_train_metrics = []
    final_val_metrics = []
    for tr_ds, val_ds in zip(fed_datasets, val_datasets):
        final_TR_metric = keras_model.evaluate(tr_ds, verbose=0)
        final_VL_metric = keras_model.evaluate(val_ds, verbose=0)
        final_train_metrics.append(final_TR_metric) 
        final_val_metrics.append(final_VL_metric)
    
    final_TR_mean = np.mean(final_train_metrics, axis=0)
    final_VL_mean = np.mean(final_val_metrics, axis=0)
    
    # ··· Training and validation evaluation using AVERAGE
    # it is necessary to use client ids. AVERAGE takes into account database balance
    # we use best_clients_ids which corresponds to the combination of databases selected when best model weights are achived.
    tr_ids_ds = itemgetter(*best_clients_ids)(fed_datasets)
    val_ids_ds = itemgetter(*best_clients_ids)(val_datasets)
    ids_we = itemgetter(*best_clients_ids)(data_balance) 
    
    final_train_metrics_avg = []
    final_val_metrics_avg = []
    for tr_ds, val_ds in zip(tr_ids_ds, val_ids_ds):
        final_TR_metric_avg = keras_model.evaluate(tr_ds, verbose=0)
        final_VL_metric_avg = keras_model.evaluate(val_ds, verbose=0)
        final_train_metrics_avg.append(final_TR_metric_avg) 
        final_val_metrics_avg.append(final_VL_metric_avg)
    
    final_TR_avg = np.average(final_train_metrics_avg, axis=0, weights=ids_we)
    final_VL_avg = np.average(final_val_metrics_avg, axis=0, weights=ids_we)
    

    print('Train dataset {0}:\tacc = {1:.4f}, kappa = {2:.4f}, loss = {3:.5f}'.format(fed, final_TR_mean[1], final_TR_mean[2], final_TR_mean[0]))
    print('Train dataset avg {0}:\tacc = {1:.4f}, kappa = {2:.4f}, loss = {3:.5f}'.format(fed, final_TR_avg[1], final_TR_avg[2], final_TR_avg[0]))
    print('Val dataset {0}:\tacc = {1:.4f}, kappa = {2:.4f}, loss = {3:.4f}'.format(fed, final_VL_mean[1], final_VL_mean[2], final_VL_mean[0]))
    print('Val dataset avg {0}:\tacc = {1:.4f}, kappa = {2:.4f}, loss = {3:.4f}'.format(fed, final_VL_avg[1], final_VL_avg[2], final_VL_avg[0]))
    print('External test dataset {0}:\tacc = {1:.4f}, kappa = {2:.4f}, loss = {3:.5f}'.format(CLIENTS[INDEX], test_metrics[1], test_metrics[2], test_metrics[0]))
    print('External full test database {0}:\tacc = {1:.4f}, kappa = {2:.4f}, loss = {3:.5f}'.format(CLIENTS[INDEX], full_metrics[1], full_metrics[2], full_metrics[0]))
    
    # Save model
    os.chdir(client_tb_logs)
    keras_model.save(server)

    INDEX += 1 
        
    # CLEAR SESSION  ···························································
    tf.keras.backend.clear_session() # resets all state generated by Keras
# ______________________________________________________________________________
