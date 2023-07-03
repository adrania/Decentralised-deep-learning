#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  16 13:29:18 2023

@author: adriana
"""

# ··············································································
# BEFORE START 
# ··············································································
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
INDEX = 0
SERVER = False 

if SERVER:
    DB_PATH = os.path.abspath('path') 
else:
    DB_PATH = os.path.abspath('path')

DB = ['ISRUC', 'SHHS', 'DREAMS', 'Telemetry', 'HMC-APR-2018', 'Dublin'] 
E_FOLDER = 'e5-noFil-EpochNorm'

BATCH_SIZE = 100
NUM_CLASSES = 5
SPLIT = int(E_FOLDER[1])

MODEL_NAME = 'run_2023-01-19' # TODO: change depending on which model we want to evaluate
# ··············································································

# ··············································································
# MAIN CODE
# ··············································································
for folder in os.listdir(DB_PATH):
    print('Local model: ', folder) 
    neWpath = os.path.join(DB_PATH, folder) 
    print(neWpath)
    for root, dirs, files in os.walk(neWpath):            
        # ··· Load model 
        if root.endswith(E_FOLDER + '/models'):
            os.chdir(root)
            print('Loading model...')
            print(root)
            model = keras.models.load_model(MODEL_NAME)
    for db in DB:
        # ··· Entramos en todas las demás DB donde vamos a evaluar
        if folder != db:
            external_path = os.path.join(DB_PATH, db)
            print('· Test database: ', db)
            print(external_path)
            for root, dirs, files in os.walk(external_path):                  
                if root.endswith(E_FOLDER + '/datasets'):
                    os.chdir(root)
                    # ··· Load external dataset
                    print('Building external dataset...')
                    if SERVER:
                        external_data = np.load('server_full_data.npy')
                    else:
                        external_data = np.load('full_data.npy')
                    
                    external_buffer_size = len(external_data)
                    external_labels = np.load('full_labels.npy')
    
                    external_dataset = functions.My_Custom_Generator(external_data, external_labels, BATCH_SIZE, SEED, SPLIT)
                    print(external_buffer_size)
                    # MODEL EVALUATION ·····························································
                    print('Evaluation...')
                    external_metrics = predict_and_score_macro(external_dataset, external_labels, model)
                    eval_external_metrics = model.evaluate(external_dataset, verbose = 0)
                    # ··············································································
    
                    print('* METRICS *')
                    print('acc = {0:.4f}, precision = {1:.4f}, recall = {2:.4f}, f1_score = {3:.4f}, kappa = {4:.4f}, loss = {5:.4f}'.format(external_metrics[0], external_metrics[1], external_metrics[2], external_metrics[3], eval_external_metrics[2], eval_external_metrics[0]))    
                    print('···························')

    # ··············································································
