###############################
#Example for generating healthy tissue embeddings for a cancer type
###############################

# Preprocess data
python Preprocess_Gtex_Rnaseq_Expressions.py BRCA
python Create_Gtex_Rnaseq_PCs.py BRCA

# Create DeepProfile embeddings
python Encode_GTEX_Data_with_VAE.py BRCA 5 0 100
python Encode_GTEX_Data_with_VAE.py BRCA 10 0 100
python Encode_GTEX_Data_with_VAE.py BRCA 25 0 100
python Encode_GTEX_Data_with_VAE.py BRCA 50 0 100
python Encode_GTEX_Data_with_VAE.py BRCA 75 0 100
python Encode_GTEX_Data_with_VAE.py BRCA 100 0 100

python Create_DeepProfile_GTEX_Embeddings.py BRCA

# Train healthy tissue classifiers
python Normal_Tissue_Classifier.py BRCA