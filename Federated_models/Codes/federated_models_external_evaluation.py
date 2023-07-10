#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:39:18 2023

@author: adriana
"""

# BEFORE START ·································································
# · Run: python federated_model_external_evaluation.py
# This code calculate a full picture of federated model performance 
# It computes accuracy, precision, recall, f1 score and kappa metrics 
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
import functions # it is necessary to be located in "functions.py" path to import it
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
tf.debugging.set_log_device_placement(False) 
# ______________________________________________________________________________
 
# ··············································································
# GLOBAL PARAMETERS 
# ··············································································
SEED = 0
INDEX = 4

PARENT_PATH = os.path.abspath('path_to_federated')
DB_PATH = os.path.abspath('path_to_databases') 

MODELS = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'] 
CLIENTS = ['ISRUC', 'SHHS', 'DREAMS', 'Telemetry', 'HMC-APR-2018', 'Dublin'] 

# ··· Model → TODO · Change model name and best_clients_ids
FEDERATED_MODEL = 'model_name'
BEST_CLIENTS_IDS = ['best_clients_ids'] 

# ··· Datasets configuration
E_FOLDER = 'e5-noFil-EpochNorm'
SPLIT = int(E_FOLDER[1])
BATCH_SIZE = 100
# ______________________________________________________________________________


# FUNCTIONS ····································································
def predict_and_score_macro(dataset, true_labels, model):
    import sklearn.metrics

    # ··· Get dataset predictions 
    probabilities = model.predict(dataset, verbose = 0)
    predictions = np.argmax(probabilities, axis = 1) # by column
    categorical_preds = keras.utils.to_categorical(predictions) # transform into categorical 

    # ··· Calculate metrics (acc, precision, recall and f1_score)
    acc = sklearn.metrics.accuracy_score(true_labels, categorical_preds)
    precision = sklearn.metrics.precision_score(true_labels, categorical_preds, average='macro')
    recall = sklearn.metrics.recall_score(true_labels, categorical_preds, average='macro')
    f1_score = sklearn.metrics.f1_score(true_labels, categorical_preds, average='macro')

    return [acc, precision, recall, f1_score]

def multi_dataset (dataset, data_weights, model, evaluation_type):
    metrics = []

    for i, ds in enumerate(dataset):
        true_labels = np.concatenate([y for x, y in ds], axis = 0)
        
        # ··· Metrics computed with predict
        if evaluation_type == 'predict':
            metric = predict_and_score_macro(ds, true_labels, model)
        
        # ··· Metrics computed with evaluate
        if evaluation_type == 'evaluate':
            metric = model.evaluate(ds, verbose = 0)

        metrics.append(metric) # list with every client metrics
    
    # ··· Metrics average using MEAN
    # it is not necessary to use clients ids. Mean doesn't take into account database balance
    global_metric_mean = np.mean(metrics, axis=0)

    # ··· Metrics average using AVERAGE
    # it is necessary to use client ids. AVERAGE takes into account database balance
    # we use best_clients_ids which corresponds to the combination of databases selected when best model weights are achived.
    global_metric_average = np.average(metrics, axis=0, weights=data_weights)

    return metrics, global_metric_average, global_metric_mean
# ··············································································            

# ··············································································
# MAIN CODE 
# ··············································································
for server in MODELS:
    if server == 'F5':
        train_datasets = [] # train datasets
        val_datasets = [] # validation datasets
        test_datasets = [] # test datasets list
        len_datasets = []
        fed = []

        # LOAD DATASETS ···························································   
        for folder in os.listdir(DB_PATH): 
            neWpath = os.path.join(DB_PATH, folder)

            # ··· Load local datasets
            if (folder in CLIENTS and folder != CLIENTS[INDEX]) :
                for root, dirs, files in os.walk(neWpath):                  
                    if root.endswith(E_FOLDER + '/datasets'):
                        os.chdir(root)
                        
                        # ··· Load local data
                        train_data = np.load('train_data.npy')
                        val_data = np.load('val_data.npy')
                        test_data = np.load('test_data.npy')
                             
                        train_labels = np.load('train_labels.npy')
                        val_labels = np.load('val_labels.npy')
                        test_labels = np.load('test_labels.npy')
                        
                        # ··· Calculate the amount of data per dataset
                        train_buffer_size = len(train_data)
                        val_buffer_size = len(val_data)
                        test_buffer_size = len(test_data)
        
                        # ··· Build datasets
                        tr_dataset = functions.ds_generator_v6(train_data, train_labels, SPLIT, test_buffer_size, BATCH_SIZE, SEED)
                        vl_dataset = functions.ds_generator_v6(val_data, val_labels, SPLIT, test_buffer_size, BATCH_SIZE, SEED)
                        ts_dataset = functions.ds_generator_v6(test_data, test_labels, SPLIT, test_buffer_size, BATCH_SIZE, SEED)

                        train_datasets.append(tr_dataset)
                        val_datasets.append(vl_dataset)
                        test_datasets.append(ts_dataset) # accumulate datasets
                        
                        len_datasets.append(train_buffer_size)
                        fed.append(folder) # clients list
            
            # ··· Load external test dataset, unseen client = external evaluation 
            if folder == CLIENTS[INDEX]:
                for root, dirs, files in os.walk(neWpath):                   
                    if root.endswith(E_FOLDER + '/datasets'):
                        os.chdir(root)
                        
                        # ··· Load data (test dataset)
                        external_test_data = np.load('test_data.npy') # local 
                        external_full_data = np.load('full_data.npy')
                        external_test_labels = np.load('test_labels.npy')
                        external_full_labels = np.load('full_labels.npy')

                        # ··· Calculate the amount of data 
                        external_test_buffer_size = len(external_test_data)
                        external_full_buffer_size = len(external_full_data)
                    
                        # ··· Build dataset for each database
                        external_test_dataset = functions.ds_generator_v6(external_test_data, external_test_labels, SPLIT, external_test_buffer_size, BATCH_SIZE, SEED)       
                        external_full_dataset = functions.ds_generator_v6(external_full_data, external_full_labels, SPLIT, external_full_buffer_size, BATCH_SIZE, SEED)
        
        # ··········································································     
        tr_datasets_id = [i for i, data in enumerate(train_datasets)] # datasets ids
        total_data = sum(len_datasets)
        data_balance = [i/total_data for i in len_datasets] # weights

        # ··· Calculate client weights
        train_datasets = itemgetter(*BEST_CLIENTS_IDS)(train_datasets)
        val_datasets = itemgetter(*BEST_CLIENTS_IDS)(val_datasets)
        test_datasets = itemgetter(*BEST_CLIENTS_IDS)(test_datasets)
        data_weights = itemgetter(*BEST_CLIENTS_IDS)(data_balance)

        # ··········································································
        print('Federated model: ', server)
        print('· Out: ', CLIENTS[INDEX])
        print('· Client list: ', itemgetter(*BEST_CLIENTS_IDS)(fed))
        print('· Data weights: ', data_weights)
        # ··········································································
            
        # LOAD MODEL ·······························································    
        print('Loading model...')
        
        folder_name = str(server + FEDERATED_MODEL)
        
        for root, dirs, files in os.walk(PARENT_PATH):              
            if root.endswith(folder_name):
                print('· Model: ', folder_name)
                os.chdir(root)
                model = keras.models.load_model(server)
        # ··········································································

        # ··········································································
        # FINAL SERVER EVALUATION 
        # ··········································································
        print('Model evaluation...\n')

        # ··· Metrics from sklearn with model.predict
        p_clients_train_metrics, p_train_metrics_avg, p_train_metrics_mean = multi_dataset(train_datasets, data_weights, model, 'predict')
        p_clients_val_metrics, p_val_metrics_avg, p_val_metrics_mean = multi_dataset(val_datasets, data_weights, model, 'predict')
        p_clients_test_metrics, p_test_metrics_avg, p_test_metrics_mean = multi_dataset(test_datasets, data_weights, model, 'predict')

        p_external_test_metrics = predict_and_score_macro(external_test_dataset, external_test_labels, model)
        p_external_full_metrics = predict_and_score_macro(external_full_dataset, external_full_labels, model)

        # ··· Metrics from model.evaluate
        e_clients_train_metrics, e_train_metrics_avg, e_train_metrics_mean = multi_dataset(train_datasets, data_weights, model, 'evaluate')
        e_clients_val_metrics, e_val_metrics_avg, e_val_metrics_mean = multi_dataset(val_datasets, data_weights, model, 'evaluate')
        e_clients_test_metrics, e_test_metrics_avg, e_test_metrics_mean = multi_dataset(test_datasets, data_weights, model, 'evaluate')

        e_external_test_metrics = model.evaluate(external_test_dataset, verbose=0) 
        e_external_full_metrics = model.evaluate(external_full_dataset, verbose=0)

        print('* METRICS *')
        print('External test dataset: acc = {0:.4f}, precision = {1:.4f}, recall = {2:.4f}, f1_score = {3:.4f}, kappa = {4:.4f}, loss = {5:.4f}'.format(p_external_test_metrics[0], p_external_test_metrics[1], p_external_test_metrics[2], p_external_test_metrics[3], e_external_test_metrics[2], e_external_test_metrics[0]))       
        print('External full dataset: acc = {0:.4f}, precision = {1:.4f}, recall = {2:.4f}, f1_score = {3:.4f}, kappa = {4:.4f}, loss = {5:.4f}'.format(p_external_full_metrics[0], p_external_full_metrics[1], p_external_full_metrics[2], p_external_full_metrics[3], e_external_full_metrics[2], e_external_full_metrics[0]))    
        
        print('\n* Average ·······················')
        print('Train dataset: acc = {0:.4f}, precision = {1:.4f}, recall = {2:.4f}, f1_score = {3:.4f}, kappa = {4:.4f}, loss = {5:.4f}'.format(p_train_metrics_avg[0], p_train_metrics_avg[1], p_train_metrics_avg[2], p_train_metrics_avg[3], e_train_metrics_avg[2], e_train_metrics_avg[0]))
        print('Validation dataset: acc = {0:.4f}, precision = {1:.4f}, recall = {2:.4f}, f1_score = {3:.4f}, kappa = {4:.4f}, loss = {5:.4f}'.format(p_val_metrics_avg[0], p_val_metrics_avg[1], p_val_metrics_avg[2], p_val_metrics_avg[3], e_val_metrics_avg[2], e_val_metrics_avg[0]))
        print('Local test dataset: acc = {0:.4f}, precision = {1:.4f}, recall = {2:.4f}, f1_score = {3:.4f}, kappa = {4:.4f}, loss = {5:.4f}'.format(p_test_metrics_avg[0], p_test_metrics_avg[1], p_test_metrics_avg[2], p_test_metrics_avg[3], e_test_metrics_avg[2], e_test_metrics_avg[0]))       

        print('\n* Mean ··························')
        print('Train dataset: acc = {0:.4f}, precision = {1:.4f}, recall = {2:.4f}, f1_score = {3:.4f}, kappa = {4:.4f}, loss = {5:.4f}'.format(p_train_metrics_mean[0], p_train_metrics_mean[1], p_train_metrics_mean[2], p_train_metrics_mean[3], e_train_metrics_mean[2], e_train_metrics_mean[0]))
        print('Validation dataset: acc = {0:.4f}, precision = {1:.4f}, recall = {2:.4f}, f1_score = {3:.4f}, kappa = {4:.4f}, loss = {5:.4f}'.format(p_val_metrics_mean[0], p_val_metrics_mean[1], p_val_metrics_mean[2], p_val_metrics_mean[3], e_val_metrics_mean[2], e_val_metrics_mean[0]))
        print('Local test dataset: acc = {0:.4f}, precision = {1:.4f}, recall = {2:.4f}, f1_score = {3:.4f}, kappa = {4:.4f}, loss = {5:.4f}'.format(p_test_metrics_mean[0], p_test_metrics_mean[1], p_test_metrics_mean[2], p_test_metrics_mean[3], e_test_metrics_mean[2], e_test_metrics_mean[0]))       
