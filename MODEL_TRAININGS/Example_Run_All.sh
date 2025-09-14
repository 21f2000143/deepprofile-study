###############################
# Bash script to run all steps for DeepProfiling for one cancer type: Currently set to BRCA
###############################
#!/bin/bash

# Verified PCs for each size of the datasets which cover 80% variance
# AML 6534 -> 1000 PCs
# BLADDER 371 -> 250 PCs
# BRAIN 4282 -> 1000 PCs
# BRCA 11963 -> 1000 PCs
# CERVICAL 443 -> 250 PCs
# COLON 5616 -> 1000 PCs
# HEAD_NECK 643 -> 500 PCs
# KIDNEY 2293 -> 1000 PCs
# LIVER 1937 -> 1000 PCs
# LUNG 4869 -> 1000 PCs
# OV 2714 -> 1000 PCs
# PANCREAS 602 -> 500 PCs
# PROSTATE 1195 -> 1000 PCs
# SARCOMA 2330 -> 1000 PCs
# SKIN 1240 -> 1000 PCs
# STOMACH 1742 -> 1000 PCs
# THYROID 776 -> 500 PCs
# UTERINE 661 -> 500 PCs
# # STEP 1: Creating PCs | Pass the cancer type and number of PCA components you want
# python Create_PCs_for_DeepLearning_Models.py BRCA 1000

# # STEP 2: Training VAE models
# python Run_VAE_Models.py BRCA 5 0 100 1000
# python Run_VAE_Models.py BRCA 10 0 100 1000
# python Run_VAE_Models.py BRCA 25 0 100 1000
# python Run_VAE_Models.py BRCA 50 0 100 1000
# python Run_VAE_Models.py BRCA 75 0 100 1000
# python Run_VAE_Models.py BRCA 100 0 100 1000

# # STEP 3: Running IG for VAE models
# python Get_VAE_IG_Attributions.py BRCA 5 0 100 1000
python Get_VAE_IG_Attributions.py BRCA 10 0 100 1000
python Get_VAE_IG_Attributions.py BRCA 25 0 100 1000
python Get_VAE_IG_Attributions.py BRCA 50 0 100 1000
python Get_VAE_IG_Attributions.py BRCA 75 0 100 1000
python Get_VAE_IG_Attributions.py BRCA 100 0 100 1000

# STEP 4: Learning ensemble labels
python Create_Ensemble_Labels.py BRCA 150

# STEP 5: Creating DeepProfile ensemble training embedding
python Create_DeepProfile_Training_Embeddings.py BRCA

# STEP 6: Creating DeepProfile ensemble gene attribution matrices
python Create_DeepProfile_Ensemble_Weights.py BRCA


