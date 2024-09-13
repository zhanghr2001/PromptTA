#!/bin/bash

source activate bsh_prompt

DATA=/opt/data/private/OOD_data # your directory of dataset
TRAINER=PROMPT_TA
CFG=$2   # config file
SEEDS=(1 2 3)

DATASET=$1
BACKBONE=$3     # backbone name
GPU=$4

# bash scripts/prompt_ta/main_ta_all.sh pacs b128_ep50_pacs RN50 0
# bash scripts/prompt_ta/main_ta_all.sh vlcs b128_ep50_vlcs RN50 1
# bash scripts/prompt_ta/main_ta_all.sh office_home b128_ep50_officehome RN50 3
# bash scripts/prompt_ta/main_ta_all.sh domainnet b128_ep50_domainnet RN50 0

# bash scripts/prompt_ta/main_ta_all.sh pacs b128_ep50_pacs ViT-B/16 0
# bash scripts/prompt_ta/main_ta_all.sh vlcs b128_ep50_vlcs ViT-B/16 1
# bash scripts/prompt_ta/main_ta_all.sh office_home b128_ep50_officehome ViT-B/16 1
# bash scripts/prompt_ta/main_ta_all.sh domainnet b128_ep50_domainnet ViT-B/16 1

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


for DOMAIN in "${ALL_DOMAIN[@]}"
do
  for SEED in ${SEEDS[@]}
  do
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
  done
done
