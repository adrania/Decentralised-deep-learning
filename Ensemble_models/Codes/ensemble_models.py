#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  16 14:29:18 2023

@author: adriana
"""

# ··············································································
# BEFORE START 
# ··············································································
# Run: python ensemble_models.py
# TODO · Make sure functions.py is located in the same folder as exec.sh file
# This code evaluates each ensemble combination 
# ··············································································

# LIBRARIES ····································································
from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
import os
import numpy as np
import sklearn.metrics
import functions
# ··············································································

# FUNCTIONS ····································································
def score_macro (true_labels, predictions):
    import sklearn.metrics

    # ··· Calculate metrics (acc, precision, recall and f1_score)
    acc = sklearn.metrics.accuracy_score(true_labels, predictions)
    precision = sklearn.metrics.precision_score(true_labels, predictions, average='macro')
    recall = sklearn.metrics.recall_score(true_labels, predictions, average='macro')
    f1_score = sklearn.metrics.f1_score(true_labels, predictions, average='macro')

    return [acc, precision, recall, f1_score]
# ··············································································

# ··············································································
# GLOBAL PARAMETERS 
# ··············································································
SEED = 0

DB_PATH = os.path.abspath('path_to_databases') 

E_FOLDER = 'e5-noFil-EpochNorm'
DIC = {'Ens1': 'ISRUC' , 'Ens2':'SHHS', 'Ens3':'DREAMS', 'Ens4':'Telemetry', 'Ens5':'HMC-APR-2018', 'Ens6':'Dublin'}

BATCH_SIZE = 100
NUM_CLASSES = 5
SPLIT = int(E_FOLDER[1])
METHOD_1 = 'averaging'
METHOD_2 = 'voting'

METRICS = [tf.metrics.CategoricalAccuracy(), tfa.metrics.CohenKappa(num_classes=NUM_CLASSES)] # computed metrics

MODEL_NAME = 'run_2023-01-19' # TODO: change depending on which model we want to evaluate
# ··············································································

# ··············································································
# MAIN CODE
# ··············································································
for e, db_external in DIC.items(): # recorremos diccionario  ensemble + db out
    print('\nEnsemble model: ', e)
    ensemble = []
    ens = []
    
    for folder in os.listdir(DB_PATH): # path to databases

        # ··· Load ensemble data and models
        if folder != db_external: # si la carpeta es distinta de db_out 
            path = os.path.join(DB_PATH, folder) 
            os.chdir(path) # entramos en la carpeta
            for root, dirs, files in os.walk(path): # recorremos directorio     

                # ··· Load models 
                if root.endswith(E_FOLDER + '/models'): # si la carpeta termina en models 
                    os.chdir(root) # entramos
                    model = keras.models.load_model(MODEL_NAME) # cargamos el modelo 
            
            ensemble.append(model) # acumulamos modelos
            ens.append(folder) # acumulamos bases de datos en el ensemble

        # ··· Load external dataset
        if folder == db_external: # si es igual a db_out
            path = os.path.join(DB_PATH, folder) 
            os.chdir(path) # cambiamos de directorio y entramos
            for root, dirs, files in os.walk(path):    

                # ··· Load data and configure datasets
                if root.endswith(E_FOLDER + '/datasets'): # entramos en los datasets y cargamos full data from db out
                    os.chdir(root)
                    external_data = np.load('full_data.npy')
                    external_labels = np.load('full_labels.npy')
                    external_dataset = functions.My_Custom_Generator(external_data, external_labels, BATCH_SIZE, SEED, SPLIT)
    
    print('Databases in ensemble: {0} - {1} models'.format(ens, len(ensemble)))
    print('Database out of ensemble {0} - size: {1}'.format(db_external, len(external_data))) 
    # ··············································································

    # PREDICTIONS ··································································
    # ··· Averaging
    print('· Starting averaging method...')
    external_average_preds = functions.get_predictions(ensemble, external_dataset, method=METHOD_1) # compute the predictions for the ensemble using full_dataset
    external_average_preds_cat = keras.utils.to_categorical(external_average_preds, num_classes=NUM_CLASSES)

    # ··· Max voting
    print('· Starting max voting method...')
    external_voting_preds = functions.get_predictions(ensemble, external_dataset, method=METHOD_2) # compute the predictions for the ensemble using full_dataset
    external_voting_preds_cat = keras.utils.to_categorical(external_voting_preds, num_classes=NUM_CLASSES)
    external_voting_preds_cat_sq = np.squeeze(external_voting_preds_cat, axis=0) # remove axes of length one - axis is optional (only for max voting)
    
    # ··· Equal weights
    print('· Starting equal weights method...')
    equal_weights = [1.0/(len(ensemble)) for _ in range(len(ensemble))] # list (size: len(ensemble)) with equal values
    external_equal_preds = functions.ensemble_predictions(ensemble, equal_weights, external_dataset)
    
    # ··· Proportional weights
    print('· Starting proportional weights method...')
    if e == 'Ens1':
        weights = [0.13, 0.06, 0.33, 0.06, 0.42]
    if e == 'Ens2':
        weights = [0.14, 0.07, 0.29, 0.06, 0.44]
    if e == 'Ens3':
        weights = [0.11, 0.05, 0.22, 0.27, 0.35]
    if e == 'Ens4':
        weights = [0.05, 0.24, 0.29, 0.05, 0.37]
    if e == 'Ens5':
        weights = [0.15, 0.07, 0.32, 0.39, 0.07]
    if e == 'Ens6':
        weights = [0.11, 0.22, 0.27, 0.05, 0.35]

    external_proportional_preds = functions.ensemble_predictions(ensemble, weights, external_dataset)

    # ··· Nelder-Mead
    print('Starting Nelder-Mead method: asigning best weights...')

    # · Define bounds on each model, compute best_weigths and perform predictions
    # initial_weights = [1.0/(len(ensemble)) for _ in range(len(ensemble))] # list (size: len(ensemble)) with equal values 
    # bound_w = [(0.0, 1.0)  for _ in range(len(ensemble))] # return a list with tuples (0.0, 1.0) - len(ensemble) size
    # search_arg = (ensemble, val_dataset, METRICS)  # arguments to the loss function
    # options = {'maxiter':MAX_ITER, 'disp':True, 'adaptive':True, 'return_all':True}
    # result = scipy.optimize.minimize(functions.loss_function, initial_weights, args=search_arg, method=METHOD, bounds=bound_w, options=options)
    # best_weights = functions.normalize(result['x']) 

    if e == 'Ens1':
        nelder_weights = [0.1863, 0.1925, 0.2185, 0.1774, 0.2253]
    if e == 'Ens2':
        nelder_weights = [0.1873, 0.1806, 0.2193, 0.1821, 0.2307]	
    if e == 'Ens3':
        nelder_weights = [0.1812, 0.1529, 0.2242, 0.2038, 0.2379]	
    if e == 'Ens4':
        nelder_weights = [0.1598, 0.2002, 0.2240, 0.1853, 0.2307]	
    if e == 'Ens5':
        nelder_weights = [0.1594, 0.2020, 0.2302, 0.2235, 0.1848]	
    if e == 'Ens6':
        nelder_weights = [0.1899, 0.2184, 0.2029, 0.1567, 0.2320]	
	
    external_nelder_preds = functions.ensemble_predictions(ensemble, nelder_weights, external_dataset)
    # ··············································································

    # PERFORMANCE ··································································
    print('Evaluation...')

    external_average_metrics = score_macro(external_labels, external_average_preds_cat)
    external_voting_metrics = score_macro(external_labels, external_voting_preds_cat_sq)
    external_equal_metrics = score_macro(external_labels, external_equal_preds)
    external_proportional_metrics = score_macro(external_labels, external_proportional_preds)
    external_nelder_metrics = score_macro(external_labels, external_nelder_preds)

    external_average_metrics_eval = functions.get_performance(METRICS, external_labels, external_average_preds_cat)
    external_voting_metrics_eval = functions.get_performance(METRICS, external_labels, external_voting_preds_cat_sq)
    external_equal_metrics_eval = functions.get_performance(METRICS, external_labels, external_equal_preds)
    external_proportional_metrics_eval = functions.get_performance(METRICS, external_labels, external_proportional_preds)
    external_nelder_metrics_eval = functions.get_performance(METRICS, external_labels, external_nelder_preds)
    # ··············································································

    print('···························')
    print('* METRICS *')
    print('Averaging')
    print('· External dataset: acc = {0:.4f}, precision = {1:.4f}, recall = {2:.4f}, f1_score = {3:.4f}, kappa = {4:.4f}'.format(external_average_metrics[0], external_average_metrics[1], external_average_metrics[2], external_average_metrics[3], external_average_metrics_eval[1]))   
    print('Voting')
    print('· External dataset: acc = {0:.4f}, precision = {1:.4f}, recall = {2:.4f}, f1_score = {3:.4f}, kappa = {4:.4f}'.format(external_voting_metrics[0], external_voting_metrics[1], external_voting_metrics[2], external_voting_metrics[3], external_voting_metrics_eval[1]))      
    print('Equal weights')
    print('· External dataset: acc = {0:.4f}, precision = {1:.4f}, recall = {2:.4f}, f1_score = {3:.4f}, kappa = {4:.4f}'.format(external_equal_metrics[0], external_equal_metrics[1], external_equal_metrics[2], external_equal_metrics[3], external_equal_metrics_eval[1]))    
    print('Size-proportional')
    print('· External dataset: acc = {0:.4f}, precision = {1:.4f}, recall = {2:.4f}, f1_score = {3:.4f}, kappa = {4:.4f}'.format(external_proportional_metrics[0], external_proportional_metrics[1], external_proportional_metrics[2], external_proportional_metrics[3], external_proportional_metrics_eval[1]))    
    print('Nelder - Mead')
    print('· External dataset: acc = {0:.4f}, precision = {1:.4f}, recall = {2:.4f}, f1_score = {3:.4f}, kappa = {4:.4f}'.format(external_nelder_metrics[0], external_nelder_metrics[1], external_nelder_metrics[2], external_nelder_metrics[3], external_nelder_metrics_eval[1]))    
    print('···························')
