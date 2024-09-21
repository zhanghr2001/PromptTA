#!/bin/bash

conda activate prompt_ta

DATA=/opt/data/private/OOD_data # your directory of dataset
TRAINER=PROMPT_TA
CFG=$2   # config file
SEEDS=(1 2 3)

DATASET=$1
BACKBONE=$3     # backbone name
GPU=$4

# bash scripts/t-sne/main_all.sh pacs b128_ep50_pacs RN50 0


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

      
python train.py \
  --backbone ${BACKBONE} \
  --target-domains ${ALL_DOMAIN[0]} \
  --root ${DATA} \
  --trainer ${TRAINER} \
  --dataset-config-file configs/datasets/${DATASET}.yaml \
  --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
  --output-dir output_tsne \
  --gpu ${GPU} \
  --tSNE
