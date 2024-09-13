#!/bin/bash

source activate bsh_prompt

DATA=/opt/data/private/OOD_data # your directory of dataset
TRAINER=CLIP_ZS_C
CFG=b128_ep50   # config file
SEED=1

DATASET=$1
BACKBONE=$2     # backbone name
GPU=$3

# bash scripts/clip/main_clip_c.sh pacs RN50 0
# bash scripts/clip/main_clip_c.sh vlcs RN50 1
# bash scripts/clip/main_clip_c.sh office_home RN50 3
# bash scripts/clip/main_clip_c.sh domainnet RN50 0

# bash scripts/clip/main_clip_c.sh pacs ViT-B/16 0
# bash scripts/clip/main_clip_c.sh vlcs ViT-B/16 2
# bash scripts/clip/main_clip_c.sh office_home ViT-B/16 4
# bash scripts/clip/main_clip_c.sh domainnet ViT-B/16 0

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
