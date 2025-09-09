###############################
# Bash script to run all steps for DeepProfile model training and evaluation
###############################
#!/bin/bash

# STEP 1: Creating PCs
python Create_PCs_for_DeepLearning_Models.py BRCA 1000

# STEP 2: Training VAE models
python Run_VAE_Models.py BRCA 5 0 100
python Run_VAE_Models.py BRCA 10 0 100
python Run_VAE_Models.py BRCA 25 0 100
python Run_VAE_Models.py BRCA 50 0 100
python Run_VAE_Models.py BRCA 75 0 100
python Run_VAE_Models.py BRCA 100 0 100

# STEP 3: Running IG for VAE models
python Get_VAE_IG_Attributions.py BRCA 5 0 100
python Get_VAE_IG_Attributions.py BRCA 10 0 100
python Get_VAE_IG_Attributions.py BRCA 25 0 100
python Get_VAE_IG_Attributions.py BRCA 50 0 100
python Get_VAE_IG_Attributions.py BRCA 75 0 100
python Get_VAE_IG_Attributions.py BRCA 100 0 100

# STEP 4: Learning ensemble labels
python Create_Ensemble_Labels.py BRCA 150

# STEP 5: Creating DeepProfile ensemble training embedding
python Create_DeepProfile_Training_Embeddings.py BRCA

# STEP 6: Creating DeepProfile ensemble gene attribution matrices
python Create_DeepProfile_Ensemble_Weights.py BRCA


