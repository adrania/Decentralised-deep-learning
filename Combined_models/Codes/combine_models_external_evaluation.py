#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  16 10:29:18 2023

@author: adriana
"""

# ··············································································
# BEFORE START 
# ··············································································
# · How to run:  python combined_models_external_evaluation.py
# TODO · IMPORTANT: make sure functions.py is located in the same folder as this file   
# This code evaluates combined models. It calculates accuracy, precision, recall, f1 score and Kappa metrics.
# ··············································································

# LIBRARIES ····································································
from tensorflow import keras
import os
import functions
import numpy as np
# ··············································································

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
# ··············································································

# ··············································································
# GLOBAL PARAMETERS 
# ··············································································
SEED = 0

DB_PATH = os.path.abspath('path_to_databases')
MAIN_PATH = os.path.abspath('path_to_models')
    
E_FOLDER = 'e5-noFil-EpochNorm'
DIC = {'C1': 'ISRUC' , 'C2':'SHHS', 'C3':'DREAMS', 'C4':'Telemetry', 'C5':'HMC-APR-2018', 'C6':'Dublin'}

BATCH_SIZE = 100
NUM_CLASSES = 5
SPLIT = int(E_FOLDER[1])

MODEL_NAME = 'run_2023-01-11' # TODO: change depending on which model we want to evaluate
# ··············································································

# ··············································································
# MAIN CODE
# ··············································································
for folder in os.listdir(MAIN_PATH): 
    print('\nCombined model: ', folder)
    print('Database out: ', DIC[folder])

    neWpath = os.path.join(MAIN_PATH, folder) 
    for root, dirs, files in os.walk(neWpath):
        
        # ··· Load data and configure datasets
        if root.endswith('/datasets'):
            print('Building local datasets...')
            os.chdir(root)
          
            train_data = np.load('train_data.npy')
            val_data = np.load('val_data.npy')
            test_data = np.load('test_data.npy')

            train_labels = np.load('train_labels.npy')
            val_labels = np.load('val_labels.npy')
            test_labels = np.load('test_labels.npy')

            train_buffer_size = len(train_data)
            val_buffer_size = len(val_data)
            test_buffer_size = len(test_data)
            print('· Train size: ',train_buffer_size)
            print('· Val size: ', val_buffer_size)
            print('· Test size: ',test_buffer_size)

            # ··· Build dataset for each database
            if BATCH_SIZE == 1:
                train_dataset = functions.My_Custom_Generator(train_data, train_labels, BATCH_SIZE, SEED, SPLIT)
                val_dataset = functions.My_Custom_Generator(val_data, val_labels, BATCH_SIZE, SEED, SPLIT)
                test_dataset = functions.My_Custom_Generator(test_data, test_labels, BATCH_SIZE, SEED, SPLIT)
            else:
                train_dataset = functions.My_Custom_Generator(train_data, train_labels, BATCH_SIZE, SEED, SPLIT)
                val_dataset = functions.My_Custom_Generator(val_data, val_labels, BATCH_SIZE, SEED, SPLIT)
                test_dataset = functions.My_Custom_Generator(test_data, test_labels, BATCH_SIZE, SEED, SPLIT)

        # ··· Load model 
        if root.endswith('/models'):
            os.chdir(root)
            print('Loading model...')
            model = keras.models.load_model(MODEL_NAME)

    # ··· Load external dataset
    print('Building external dataset...')
    external_path = os.path.abspath(DB_PATH + '/' + DIC[folder] + '/' + E_FOLDER + '/datasets')

    if SERVER:
        external_data = np.load(external_path + '/server_full_data.npy')
    else:
        external_data = np.load(external_path + '/full_data.npy')
    
    external_buffer_size = len(external_data)
    external_labels = np.load(external_path + '/full_labels.npy')

    external_dataset = functions.My_Custom_Generator(external_data, external_labels, BATCH_SIZE, SEED, SPLIT)
    print('· External {0} size: {1}'.format(DIC[folder], external_buffer_size))
    # ··············································································

    # MODEL EVALUATION ·····························································
    print('Evaluation...')
    train_metrics = predict_and_score_macro(train_dataset, train_labels, model)
    val_metrics = predict_and_score_macro(val_dataset, val_labels, model)
    test_metrics = predict_and_score_macro(test_dataset, test_labels, model)
    external_metrics = predict_and_score_macro(external_dataset, external_labels, model)

    eval_train_metrics = model.evaluate(train_dataset, verbose = 0)
    eval_val_metrics = model.evaluate(val_dataset, verbose = 0)
    eval_test_metrics = model.evaluate(test_dataset, verbose = 0)
    eval_external_metrics = model.evaluate(external_dataset, verbose = 0)
    # ··············································································

    print('···························')
    print('* METRICS *')
    print('External full dataset: acc = {0:.4f}, precision = {1:.4f}, recall = {2:.4f}, f1_score = {3:.4f}, kappa = {4:.4f}, loss = {5:.4f}'.format(external_metrics[0], external_metrics[1], external_metrics[2], external_metrics[3], eval_external_metrics[2], eval_external_metrics[0]))    
    print('Train dataset: acc = {0:.4f}, precision = {1:.4f}, recall = {2:.4f}, f1_score = {3:.4f}, kappa = {4:.4f}, loss = {5:.4f}'.format(train_metrics[0], train_metrics[1], train_metrics[2], train_metrics[3], eval_train_metrics[2], eval_train_metrics[0]))
    print('Validation dataset: acc = {0:.4f}, precision = {1:.4f}, recall = {2:.4f}, f1_score = {3:.4f}, kappa = {4:.4f}, loss = {5:.4f}'.format(val_metrics[0], val_metrics[1], val_metrics[2], val_metrics[3], eval_val_metrics[2], eval_val_metrics[0]))
    print('Local test dataset: acc = {0:.4f}, precision = {1:.4f}, recall = {2:.4f}, f1_score = {3:.4f}, kappa = {4:.4f}, loss = {5:.4f}'.format(test_metrics[0], test_metrics[1], test_metrics[2], test_metrics[3], eval_test_metrics[2], eval_test_metrics[0]))       
    print('···························')
