###############################
# Bash script to run all the steps for BRCA
###############################

python Create_PCA_Data.py BRCA
python Create_ICA_Data.py BRCA
python Create_RP_Data.py BRCA
python Train_AE_Models.py BRCA
python Get_AE_IG_Attributions.py BRCA 0
python Train_DAE_Models.py BRCA 
python Get_DAE_IG_Attributions.py BRCA 0
