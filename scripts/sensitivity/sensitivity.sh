#!/bin/bash

source activate bsh_prompt

DATA=/opt/data/private/OOD_data # your directory of dataset
TRAINER=PROMPT_TA
CFG=$2   # config file
SEEDS=(1 2 3)

alpha=(0.5 1.0 2.0 5.0)
beta=(0.5 1.0 2.0 5.0)

DATASET=$1
BACKBONE=$3     # backbone name
GPU=$4

# bash scripts/sensitivity/sensitivity.sh pacs b128_ep50_pacs RN50 0
# bash scripts/sensitivity/sensitivity.sh vlcs b128_ep50_vlcs RN50 1
# bash scripts/sensitivity/sensitivity.sh office_home b128_ep50_officehome RN50 3
# bash scripts/sensitivity/sensitivity.sh domainnet b128_ep50_domainnet RN50 0

# bash scripts/sensitivity/sensitivity.sh pacs b128_ep50_pacs ViT-B/16 0
# bash scripts/sensitivity/sensitivity.sh vlcs b128_ep50_vlcs ViT-B/16 1
# bash scripts/sensitivity/sensitivity.sh office_home b128_ep50_officehome ViT-B/16 1
# bash scripts/sensitivity/sensitivity.sh domainnet b128_ep50_domainnet ViT-B/16 1

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

for a in ${alpha[@]}
do
  for b in ${beta[@]}
  do
    for DOMAIN in "${ALL_DOMAIN[@]}"
    do
      for SEED in ${SEEDS[@]}
      do
        DIR=output_sensitivity/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/alpha${a}_beta${b}/${DOMAIN}/seed_${SEED}

        if [ -d "$DIR" ]; then
          echo "Results are available in ${DIR}, so skip this job"
        else
          echo "Run this job and save the output to ${DIR}"
          
          python train_sensitivity.py \
            --backbone ${BACKBONE} \
            --target-domains ${DOMAIN} \
            --root ${DATA} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --seed ${SEED} \
            --gpu ${GPU} \
            --alpha ${a} \
            --beta ${b}

        fi
      done
    done
  done
done
