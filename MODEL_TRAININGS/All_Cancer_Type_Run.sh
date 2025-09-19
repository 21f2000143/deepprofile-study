#!/bin/bash

declare -A cancers

cancers[AML]=1000
cancers[BRAIN]=1000
cancers[BRCA]=1000
cancers[COLON]=1000
cancers[KIDNEY]=1000
cancers[LIVER]=1000
cancers[LUNG]=1000
cancers[OV]=1000
cancers[PROSTATE]=1000
cancers[SARCOMA]=1000
cancers[SKIN]=1000   # Melanoma = SKIN
cancers[STOMACH]=1000
cancers[BLADDER]=250
cancers[CERVICAL]=250
cancers[HEAD_NECK]=500
cancers[PANCREAS]=500
cancers[THYROID]=500
cancers[UTERINE]=500

CANCERS=('AML' 'BRAIN' 'COLON' 'KIDNEY' 'LIVER' 'LUNG' 'OV' 'PROSTATE' 'SARCOMA' 'SKIN' 'STOMACH' 'BLADDER' 'CERVICAL' 'HEAD_NECK' 'PANCREAS' 'THYROID' 'UTERINE' 'BRCA')

for CANCER in "${CANCERS[@]}"; do
    SESSION="$CANCER"
    SIZE="${cancers[$CANCER]}"

    echo "▶️ Launching tmux session: $SESSION (size=$SIZE)"

    tmux new-session -d -s "$SESSION" \
    "python Get_VAE_IG_Attributions.py $CANCER 5 0 100 $SIZE; \
     python Get_VAE_IG_Attributions.py $CANCER 10 0 100 $SIZE; \
     python Get_VAE_IG_Attributions.py $CANCER 25 0 100 $SIZE; \
     python Get_VAE_IG_Attributions.py $CANCER 50 0 100 $SIZE; \
     python Get_VAE_IG_Attributions.py $CANCER 75 0 100 $SIZE; \
     python Get_VAE_IG_Attributions.py $CANCER 100 0 100 $SIZE; \
     python Create_Ensemble_Labels.py $CANCER 150; \
     python Create_DeepProfile_Training_Embeddings.py $CANCER; \
     python Create_DeepProfile_Ensemble_Weights.py $CANCER"
done
