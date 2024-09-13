#!/bin/bash

source activate bsh_prompt

DATA=/opt/data/private/OOD_data # your directory of dataset
TRAINER=PROMPT_STYLER
CFG=b128_ep50_domainnet   # config file
SEED=3

DATASET=$1
BACKBONE=$2     # backbone name
DOMAIN=$3
GPU=$4

# bash scripts/prompt_styler/main_ps.sh domainnet RN50

# bash scripts/prompt_styler/main_ps.sh domainnet ViT-B/16

if [ "$DATASET" = "pacs" ]; then
  ALL_DOMAIN=('a' 'c' 'p' 's')
elif [ "$DATASET" = "vlcs" ]; then
  ALL_DOMAIN=('c' 'l' 'p' 's')
elif [ "$DATASET" = "office_home" ]; then
  ALL_DOMAIN=('a' 'c' 'p' 'r')
elif [ "$DATASET" = "terra_incognita" ]; then
  ALL_DOMAIN=('l38' 'l43' 'l46' 'l100')
elif [ "$DATASET" = "domainnet" ]; then
  ALL_DOMAIN=('c' 'i' 'p' 'q' 'r' 's')
  
fi

DIR=output/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/${DOMAIN}/seed_${SEED}

if [ -d "$DIR" ]; then
  echo "Results are available in ${DIR}, so skip this job"
else
  echo "Run this job and save the output to ${DIR}"
  
  python train.py \
    --backbone ${BACKBONE} \
    --target-domains ${DOMAIN} \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --seed ${SEED} \
    --gpu ${GPU}

fi
