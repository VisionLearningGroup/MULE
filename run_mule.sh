#!/bin/bash

export PYTHONUNBUFFERED="True"

SPLIT=$1
GPU_ID=$2
DATASET=$3
TAG=$4

case ${DATASET} in
  multi30k)
    LANGUAGES="en,de,fr,cs"
    DOMAIN_ADAPT=1e-6
    MAX_SENTENCE_LENGTH=40
    ;;
  coco)
    LANGUAGES="en,cn,jp"
    DOMAIN_ADAPT=1e-5
    MAX_SENTENCE_LENGTH=60
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

if [ "${SPLIT}" = "train" ]; then
    CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --max_sentence_length ${MAX_SENTENCE_LENGTH} \
    --domain_adapt ${DOMAIN_ADAPT} \
    --univ_pretrain \
    --dataset ${DATASET} \
    --split ${SPLIT} \
    --save_dir models/${DATASET}/${TAG} \
    --languages ${LANGUAGES}
else
    CUDA_VISIBLE_DEVICES=${GPU_ID} python test.py \
    --dataset ${DATASET} \
    --split ${SPLIT} \
    --save_dir models/${TAG} \
    --languages ${LANGUAGES} \
    --restore_path models/${DATASET}/${TAG}/two_branch-ckpt-$5.meta
fi

