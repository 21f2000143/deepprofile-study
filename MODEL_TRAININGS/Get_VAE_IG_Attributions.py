###############################
#Script for running integrated gradients to get gene-level attributions of each node
###############################

import os
import numpy as np
import pandas as pd
import math 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import model_from_json
import sys

import csv
from pathlib import Path

# This is for absolute imports from the root repository
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
  ALL_CANCER_FILES
)
# Configure GPU memory growth (TF2 replacement for ConfigProto Session config)
try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
except Exception:
    pass

#Read all user inputs
cancer = sys.argv[1]
dimension = int(sys.argv[2])
start = int(sys.argv[3])
end = int(sys.argv[4])
pca_components_file = int(sys.argv[5])

print("CANCER " + str(cancer))
print("DIM " + str(dimension))
print("START " + str(start)) 
print("END " + str(end)) 

input_folder = ALL_CANCER_FILES + '/' + cancer + '/'
output_folder = ALL_CANCER_FILES + '/' + cancer + '/'

#Load PCA weights
pca_df = pd.read_table(input_folder + cancer + '_DATA_TOP2_JOINED_PCA_' + str(pca_components_file) + 'L_COMPONENTS.tsv', index_col = 0)
print("PCA COMPONENTS ",  pca_df.shape)
pca_components = pca_df.values

 #Read input data
input_df = pd.read_table(input_folder + cancer + '_DATA_TOP2_JOINED_PCA_' + str(pca_components_file) + 'L.tsv', index_col=0)
print("INPUT FILE ", input_df.shape)

###############################
# NOTE:
# This script operates ONLY on the encoder to obtain integrated gradients
# for latent dimensions w.r.t. input PCA features. We do NOT need a custom
# VAE loss here nor to re-define layers; we simply load the serialized
# encoder model + weights and use it directly under eager execution.
###############################

#Save the weight for each run
for vae_run in range(start, end):
    
    print("MODEL " + str(vae_run))
    
    # Load model architecture
    json_path = input_folder + 'VAE_' + cancer + '_encoder_' + str(dimension) + 'L_' + str(vae_run) + '.json'
    with open(json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    encoder = model_from_json(loaded_model_json)

    # Load weights (try new .weights.h5 suffix first, fallback to legacy .h5)
    weights_base = input_folder + 'VAE_' + cancer + '_encoder_' + str(dimension) + 'L_' + str(vae_run)
    tried_paths = [weights_base + '.weights.h5', weights_base + '.h5']
    loaded = False
    for w_path in tried_paths:
        if os.path.exists(w_path):
            encoder.load_weights(w_path)
            print(f"Loaded weights: {w_path}")
            loaded = True
            break
    if not loaded:
        raise FileNotFoundError(f"Could not find weights for encoder. Tried: {tried_paths}")

    # No compile needed for inference + gradients in eager mode
    input_df_training = input_df
    latent_dim = dimension
    batch_size = 128

    # Quick shape sanity
    print("Encoder input shape expected:", encoder.input_shape, "  Provided data:", input_df_training.shape)


    #Measure weights and save absolute value of importance, averaged over samples
    from IntegratedGradients import *

    ig = IntegratedGradients(encoder)

    overall_weights = np.zeros((pca_components.shape[0], dimension))

    #Go over each node
    for latent in range(dimension):
        print("Node " + str(latent + 1))
        weights = np.zeros((pca_components.shape[0], input_df_training.shape[0]))
        
        #Go over each sample
        for i in range(input_df_training.shape[0]):
            vals = ig.explain(input_df_training.values[i, :], latent)
            # Map back to gene space via PCA loadings
            new_vals = np.matmul(vals, pca_components.T)
            weights[:, i] = new_vals
            
        #Take absolute values avg over all samples 
        overall_weights[:, latent] = np.mean(np.abs(weights), axis = 1)

    ig_df = pd.DataFrame(overall_weights, index = pca_df.index)
    print("EXPLANATIONS DF ", ig_df.shape)
    
    ig_df.to_csv(output_folder + cancer + '_DATA_VAE_Cluster_Weights_TRAINING_' + str(dimension) + 'L_fold' + str(vae_run) + '.tsv', sep='\t', quoting = csv.QUOTE_NONE)
    
